import math
from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from xright.datasets.packing_utils import patch_for_block_diag_attn

from xright.models import DPOLoss, GPTLMLoss
import time

# for recording gpu memory used
import subprocess as sp
import numpy as np


class SFT_DPO_Mix_Trainer(ABC):
    """
        Trainer to use while training a model for both SFT and DPO objectives simultaneously

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        ref_model,
        strategy,
        tokenizer,
        optim: Optimizer,
        sft_train_dataloader,
        rlhf_train_dataloader,
        sft_eval_dataloader,
        rlhf_eval_dataloader,
        scheduler,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
        pretrain_mode: bool = False,
        lambd=0.5,
        sft_opt=0.0,
        dpo_opt=0.0,
        eps=1e-4,
        ideal_dist=0.0,
        ref_pareto_front=[[0.0, 0.0]]
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.ref_model = ref_model
        # for alternating update of SFT and DPO objectives
        self.sft_train_dataloader = sft_train_dataloader
        self.rlhf_train_dataloader = rlhf_train_dataloader
        # for simultaneous evaluation of SFT and DPO objectives
        self.sft_eval_dataloader = sft_eval_dataloader
        self.rlhf_eval_dataloader = rlhf_eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        # for DPO regularization
        self.beta = beta

        # loss functions for DPO and SFT objectives
        self.rlhf_loss_fn = DPOLoss(self.beta, self.args.label_smoothing, self.args.ipo)
        self.sft_loss_fn = GPTLMLoss()

        # For SFT
        self.packing_samples = strategy.args.packing_samples
        self.pretrain_mode = pretrain_mode

        # packing samples using Flash Attention 2
        if self.packing_samples:
            assert strategy.args.flash_attn, "Only support `--packing_samples` with Flash Attention 2."
            model_type = getattr(strategy._unwrap_model(model).config, "model_type", None)
            patch_for_block_diag_attn(model_type)

        # For mixing objectives. lambd=0 corresponds to DPO, lambd=1 corresponds to DPO
        self.lambd=lambd
        self.sft_opt = sft_opt
        self.dpo_opt = dpo_opt
        self.eps = eps
        self.ideal_dist = ideal_dist
        self.ref_pareto_front = ref_pareto_front

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.sft_train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        for epoch in range(self.epochs):
            # Setup iterable train_dataloader
            sft_iter_train_dataloader = iter(self.sft_train_dataloader)
            rlhf_iter_train_dataloader = iter(self.rlhf_train_dataloader)
            step_bar = tqdm(
                range(self.sft_train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.sft_train_dataloader.sampler, DistributedSampler):
                self.sft_train_dataloader.sampler.set_epoch(epoch)

            if isinstance(self.rlhf_train_dataloader.sampler, DistributedSampler):
                self.rlhf_train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            self.ref_model.eval()
            acc_mean = 0
            loss_mean = 0

            # Train SFT and DPO simultaneously, so implement a train_dataloader agnostic loop
            # assuming both train_dataloaders have the same length
            for step in range(self.sft_train_dataloader.__len__()):
                # DEBUG
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # SFT
                prompts_id_lens, inputs, attention_masks, infos = next(sft_iter_train_dataloader)

                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                output = self.model(
                    inputs, attention_mask=attention_mask, return_output=True, packing_samples=self.packing_samples
                )

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.sft_loss_fn.IGNORE_INDEX,
                )
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompts_id_lens):
                            labels[0][index : index + source_len] = self.sft_loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompts_id_lens):
                            label[:source_len] = self.sft_loss_fn.IGNORE_INDEX

                gpt_loss = self.sft_loss_fn(output.logits, labels)
                loss_0 = gpt_loss + aux_loss * self.args.aux_loss_coef
                # self.strategy.backward(loss, self.model, self.optimizer)
                # self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict_0 = {"gpt_loss": gpt_loss.item(), "loss_mean": loss_mean}
                if self.aux_loss:
                    logs_dict_0["aux_loss"] = aux_loss.item()

                # DPO
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = next(rlhf_iter_train_dataloader)
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_logps, rejected_logps, aux_loss = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )
                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps, _ = self.concatenated_forward(
                        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )

                # loss function
                preference_loss, chosen_reward, reject_reward = self.rlhf_loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss_1 = preference_loss + aux_loss * self.args.aux_loss_coef


                acc_mean = acc_mean * 0.9 + 0.1 * (chosen_reward > reject_reward).float().mean().item()
                loss_mean = loss_mean * 0.9 + 0.1 * loss_1.item()
                # dpo logs
                logs_dict_1 = {
                    "preference_loss": preference_loss.item(),
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "acc_mean": acc_mean,
                    "loss_mean": loss_mean,
                }
               # DEBUG
                if self.strategy.is_rank_0():
                    self._log_gpu_memory_usage('before backward')
                    self._log_gpu_memory_from_nvdia_smi('before backward')
                t = time.time()
                self.strategy.backward((1-self.lambd) * loss_0 + self.lambd * loss_1, self.model, self.optimizer)
                # DEBUG
                if self.strategy.is_rank_0():
                    self._log_time_elapsed('backward', time.time()-t)
                # DEBUG
                if self.strategy.is_rank_0():
                    self._log_gpu_memory_usage('after backward')
                    self._log_gpu_memory_from_nvdia_smi('after backward')
                t = time.time()
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                # DEBUG
                if self.strategy.is_rank_0():
                    self._log_time_elapsed('optimizer_step', time.time()-t)


                 
                # logs/checkpoints/evaluate
                sft_eval_loss = self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict_0, eval='sft')
                rlhf_eval_loss = self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict_1, eval='rlhf')

                if sft_eval_loss is not None and rlhf_eval_loss is not None:
                    if sft_eval_loss-self.sft_opt <= self.eps or rlhf_eval_loss-self.dpo_opt <= self.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if sft_eval_loss - ref_loss_pair[0] <= self.eps and rlhf_eval_loss - ref_loss_pair[1] <= self.eps:
                                OPTIM_ACHIEVED = True
                                break               

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, eval='sft'):
        # logs
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        eval_loss = None
        if global_step % args.eval_steps == 0 or global_step==1:
            if eval=='sft':
                eval_loss = self.sft_evaluate(self.sft_eval_dataloader, global_step)
            if eval=='rlhf':
                eval_loss = self.rlhf_evaluate(self.rlhf_eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0 and eval=='rlhf':
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)
        
        return eval_loss

    def rlhf_evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            acc_sum = 0
            loss_sum = 0
            times = 0
            for chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens in eval_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_logps, rejected_logps, _ = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )
                reference_chosen_logps, reference_rejected_logps, _ = self.concatenated_forward(
                    self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )
                loss, chosen_reward, reject_reward = self.rlhf_loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                acc_sum += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                times += 1

                logs = {
                    "dpo_loss": loss_sum / times,
                    "dpo_acc": acc_sum / times,
                }
                logs = self.strategy.all_reduce(logs)
                step_bar.set_postfix(logs)
                step_bar.update()

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state

        return logs["eval/dpo_loss"] if self._wandb is not None and self.strategy.is_rank_0() else logs["dpo_loss"]

    def sft_evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_lens, inputs, attention_masks, infos in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                output = self.model(
                    inputs, attention_mask=attention_mask, return_output=True, packing_samples=self.packing_samples
                )

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.sft_loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompts_id_lens):
                            labels[0][index : index + source_len] = self.sft_loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompts_id_lens):
                            label[:source_len] = self.sft_loss_fn.IGNORE_INDEX

                loss = self.sft_loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"sft_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state

        return logs["eval/sft_loss"] if self._wandb is not None and self.strategy.is_rank_0() else logs["sft_loss"]

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps = self._get_batch_logps(all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False)
        chosen_logps = all_logps[: chosen_ids.shape[0]]
        rejected_logps = all_logps[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        else:
            return (per_token_logps * loss_masks).sum(-1)

    # DEBUG
    def _log_gpu_memory_usage(self, checkpoint):
        max_allocated = torch.cuda.max_memory_allocated()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        self._wandb.log({
            f"GPU Memory Allocated @ {checkpoint} (MB)": allocated / (1024 ** 2),
            f"Max. GPU Memory Allocated @ {checkpoint} (MB)": max_allocated / (1024 ** 2),
            # f"GPU Memory Reserved @ {checkpoint} (MB)": reserved / (1024 ** 2)
        })

    # DEBUG
    def _log_time_elapsed(self, checkpoint, t):

        self._wandb.log({
            f"Time @ {checkpoint} (s)": t,
            # f"GPU Memory Reserved @ {checkpoint} (MB)": reserved / (1024 ** 2)
        })

    # DEBUG
    def _log_gpu_memory_from_nvdia_smi(self, checkpoint, gpus=[0, 1]):
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        memory_use_values = np.array([int(x.split()[0]) for i, x in enumerate(memory_use_info)])[gpus]

        self._wandb.log({
            f"NVIDIA-SMI Max. Memory-Usage @ {checkpoint} (MB)": np.max(memory_use_values),
            f"NVIDIA-SMI Avg. Memory-Usage @ {checkpoint} (MB)": np.mean(memory_use_values)
        })
        
        for i, mem in enumerate(memory_use_values):
            self._wandb.log({
                f"NVIDIA-SMI GPU {i} Memory-Usage @ {checkpoint} (MB)": mem,
            })                