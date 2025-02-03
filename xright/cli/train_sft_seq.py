import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from xright.datasets import SFTDataset, RewardDataset
from xright.models import Actor
from xright.trainer import SFT_Seq_Trainer
from xright.utils import blending_datasets, get_strategy, get_tokenizer

import torch

def train(args):
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
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        model_name='model'
    )

    torch.cuda.is_available():
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
        model_name='ref_model'
    )

    torch.cuda.is_available():
        ref_model.to('cuda')

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    get_tokenizer(args.pretrain, ref_model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)


    # handles printing only once
    strategy.print(model)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

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

    train_dataloader = strategy.setup_dataloader(
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


    # DPO prepare for data and dataset

    _, eval_data = blending_datasets(
        args.rlhf_dataset,
        args.rlhf_dataset_probs,
        strategy,
        args.seed,
        max_count=args.rlhf_max_samples,
        stopping_strategy="all_exhausted",
        train_split=args.rlhf_train_split,
        eval_split=args.rlhf_eval_split,
    )

    eval_data = eval_data.select(range(min(args.rlhf_max_samples, len(eval_data))))

    eval_dataset = RewardDataset(
        eval_data, tokenizer, args.rlhf_max_len, strategy, input_template=args.input_template, is_dpo=True
    )

    rlhf_eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.rlhf_micro_train_batch_size, True, False, eval_dataset.collate_fn
    )
    # scheduler

    # constant scheduler rate
    scheduler = get_scheduler(
        "constant",
        optim
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # prepare models
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    # load checkpoint
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = SFT_Seq_Trainer(
        model=model,
        ref_model=ref_model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        sft_eval_dataloader=sft_eval_dataloader,
        rlhf_eval_dataloader=rlhf_eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
    )

    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size SFT")
    parser.add_argument("--sft_micro_train_batch_size", type=int, default=8, help="batch size per GPU for SFT")
    parser.add_argument("--rlhf_micro_train_batch_size", type=int, default=8, help="batch size per GPU for RLHF")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    
    # SFT
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    # custom dataset
    parser.add_argument("--sft_dataset", type=str, default=None)
    parser.add_argument("--sft_dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--sft_train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--sft_eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--sft_tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--sft_max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--sft_max_len", type=int, default=2048, help="Max tokens for the samples")

    # DPO
    parser.add_argument("--ref_pretrain", type=str, default=None)
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--rlhf_beta", type=float, default=0.1)
    parser.add_argument("--rlhf_ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--rlhf_label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    # custom dataset
    # parser.add_argument("--ref_pretrain", type=str, default=None)
    parser.add_argument("--rlhf_dataset", type=str, default=None)
    parser.add_argument("--rlhf_dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--rlhf_train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--rlhf_eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--rlhf_max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--rlhf_max_len", type=int, default=512)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    
    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="xright_sft_seq")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain
    train(args)
