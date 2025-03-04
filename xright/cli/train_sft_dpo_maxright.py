import argparse
import math
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import torch

from transformers.trainer import get_scheduler

from xright.datasets import RewardDataset, SFTDataset
from xright.models import Actor
from xright.trainer import SFT_DPO_MAXRIGHT_Trainer
from xright.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    # process the referece pareto values needed for stopping criteria of MOO methods
    if args.ref_pareto_sft=="":
        ref_pareto_front = [[0.0, 0.0]]
    else:
        ref_pareto_sft = [float(_) for _ in args.ref_pareto_sft.split()]
        ref_pareto_dpo= [float(_) for _ in args.ref_pareto_dpo.split()]

        assert len(ref_pareto_sft)== len(ref_pareto_dpo)

        ref_pareto_front = []
        for _sft, _dpo in zip(ref_pareto_sft, ref_pareto_dpo):
            ref_pareto_front.append([_sft, _dpo])

    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    if torch.cuda.is_available():
        model.to('cuda')
    ref_model = Actor(
        args.ref_pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
    )
    if torch.cuda.is_available():
        ref_model.to('cuda')
    if args.ref_offload:
        ref_model._offload = True

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    get_tokenizer(args.pretrain, ref_model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # DPO prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.rlhf_dataset,
        args.rlhf_dataset_probs,
        strategy,
        args.seed,
        max_count=args.rlhf_max_samples,
        stopping_strategy="all_exhausted",
        train_split=args.rlhf_train_split,
        eval_split=args.rlhf_eval_split,
    )
    train_data = train_data.select(range(min(args.rlhf_max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.rlhf_max_samples, len(eval_data))))
    train_dataset = RewardDataset(
        train_data, tokenizer, args.rlhf_max_len, strategy, input_template=args.input_template, is_dpo=True
    )
    eval_dataset = RewardDataset(
        eval_data, tokenizer, args.rlhf_max_len, strategy, input_template=args.input_template, is_dpo=True
    )

    rlhf_train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.rlhf_micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )

    rlhf_eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.rlhf_micro_train_batch_size, True, False, eval_dataset.collate_fn
    )

# SFT prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.sft_dataset,
        args.sft_dataset_probs,
        strategy,
        args.seed,
        max_count=args.sft_max_samples,
        train_split=args.sft_train_split,
        eval_split=args.sft_eval_split,
    )
    train_data = train_data.select(range(min(args.sft_max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.sft_max_samples, len(eval_data))))
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.sft_max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    eval_dataset = SFTDataset(
        eval_data,
        tokenizer,
        args.sft_max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    sft_train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.sft_micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    sft_eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.sft_micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )

    # scheduler

    # constant learning rate
    scheduler = get_scheduler(
        "constant",
        optim
    )

    # strategy prepare
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    if args.load_checkpoint:
        # ! Only printing, but not actually loading?
        strategy.print("Load checkpoint: ", args.save_path)
        # strategy.load_checkpoint(args.save_path + '/rm_model.pt')

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = SFT_DPO_MAXRIGHT_Trainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        sft_train_dataloader=sft_train_dataloader,
        rlhf_train_dataloader=rlhf_train_dataloader,
        sft_eval_dataloader=sft_eval_dataloader,
        rlhf_eval_dataloader=rlhf_eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        beta=args.beta,
        max_epochs=args.max_epochs,
        pretrain_mode=args.pretrain_mode,
        lambd=args.lambd,
        sft_opt=args.sft_opt,
        dpo_opt=args.dpo_opt,
        eps=args.eps,
        ideal_dist=args.ideal_dist,
        ref_pareto_front=ref_pareto_front
    )

    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB

    # DeepSpeed
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)

    # DPO
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    # custom dataset
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--ref_pretrain", type=str, default=None)
    parser.add_argument("--rlhf_dataset", type=str, default=None)
    parser.add_argument("--rlhf_dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--rlhf_train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--rlhf_eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--rlhf_micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--rlhf_max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--rlhf_max_len", type=int, default=512) 

    # SFT
    parser.add_argument("--sft_dataset", type=str, default=None)
    parser.add_argument("--sft_dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--sft_train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--sft_eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--sft_micro_train_batch_size", type=int, default=8, help="batch size per GPU")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    parser.add_argument("--sft_tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--sft_max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--sft_max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Mixing objectives
    parser.add_argument("--lambd", type=float, default=0.5, help="coefficient for mixing SFT and DPO objectives")
    parser.add_argument("--max_eval_steps", type=int, default=-1, help="steps until the next max evaluation of objectives")
    parser.add_argument("--sft_opt", type=float, default=0.0, help="optimum SFT objective value")
    parser.add_argument("--dpo_opt", type=float, default=0.0, help="optimum SFT objective value")
    parser.add_argument("--eps", type=float, default=1e-4, help="stopping threshold tolerance")
    parser.add_argument("--ideal_dist", type=float, default=0.0, help="distance to the ideal point (in loss space) from the comparable alternating method model")
    parser.add_argument("--ref_pareto_sft", type=str, default="", help="set of reference pareto front values for sft loss")
    parser.add_argument("--ref_pareto_dpo", type=str, default="", help="set of reference pareto front values for dpo loss")

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="xright_maxright")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    train(args)
