# only GPUs compatible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ===========================================================Joint training for SFT and RLHF===========================================================

LAMBD_SET="0.25 0.5 0.75" 
beta=0.1
lambd=0.5
learning_rate=5e-5
max_len=2048 
train_batch_size=4 # set to no. GPUS used
sft_max_samples=24000 
rlhf_max_samples=6000 
sft_micro_train_batch_size=4 
rlhf_micro_train_batch_size=1 
max_epochs=4
max_eval_steps=10 
sft_opt=1.02426
dpo_opt=0.04935
ref_pareto_sft="1.1484 1.1173 1.0917"
ref_pareto_dpo="0.2382 0.3534 0.5009"

# ----------- ALRIGHT (datasets:vicgalle/alpaca-gpt4, Dahoas/rm-hh-rlhf) -----------

for lambd in $LAMBD_SET; do
   deepspeed --module xright.cli.train_sft_dpo_alright \
      --save_path ./checkpoint/Meta-Llama-3-8B-beta=$beta-learning_rate=$learning_rate-sft_dpo_alright-lambd-$lambd \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps 100 \
      --rlhf_dataset Dahoas/rm-hh-rlhf \
      --train_batch_size $train_batch_size \
      --rlhf_micro_train_batch_size $rlhf_micro_train_batch_size \
      --rlhf_max_samples $rlhf_max_samples \
      --rlhf_max_len $max_len \
      --rlhf_eval_split dummy \
      --sft_dataset vicgalle/alpaca-gpt4 \
      --sft_max_samples $sft_max_samples \
      --sft_max_len $max_len \
      --sft_micro_train_batch_size $sft_micro_train_batch_size \
      --input_key instruction \
      --lambd $lambd \
      --sft_opt $sft_opt \
      --dpo_opt $dpo_opt \
      --ref_pareto_sft "" \
      --ref_pareto_dpo "" \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --ref_pretrain ./checkpoint/Meta-Llama-3-8B-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-sft-optim \
      --bf16 \
      --max_epochs $max_epochs \
      --zero_stage 2 \
      --beta $beta \
      --flash_attn \
      --learning_rate $learning_rate \
      --lora_rank 16 \
      --lora_alpha 16 \
      --use_wandb WANDB_KEY \
      --target_module q_proj v_proj \
      --gradient_checkpointing
done

# ----------- MAXRIGHT (datasets:vicgalle/alpaca-gpt4, Dahoas/rm-hh-rlhf) -----------

for lambd in $LAMBD_SET; do
   deepspeed --module xright.cli.train_sft_dpo_maxright \
      --save_path ./checkpoint/Meta-Llama-3-8B-beta=$beta-learning_rate=$learning_rate-sft_dpo_maxright-lambd-$lambd \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps 100 \
      --rlhf_dataset Dahoas/rm-hh-rlhf \
      --train_batch_size $train_batch_size \
      --rlhf_micro_train_batch_size $rlhf_micro_train_batch_size \
      --rlhf_max_samples $rlhf_max_samples \
      --rlhf_max_len $max_len \
      --rlhf_eval_split dummy \
      --sft_dataset vicgalle/alpaca-gpt4 \
      --sft_max_samples $sft_max_samples \
      --sft_max_len $max_len \
      --sft_micro_train_batch_size $sft_micro_train_batch_size \
      --input_key instruction \
      --lambd $lambd \
      --sft_opt $sft_opt \
      --dpo_opt $dpo_opt \
      --ref_pareto_sft "$ref_pareto_sft" \
      --ref_pareto_dpo "$ref_pareto_dpo" \
      --max_eval_steps $max_eval_steps \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --ref_pretrain ./checkpoint/Meta-Llama-3-8B-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-sft-optim \
      --bf16 \
      --max_epochs $max_epochs \
      --zero_stage 2 \
      --beta $beta \
      --flash_attn \
      --learning_rate $learning_rate \
      --lora_rank 16 \
      --lora_alpha 16 \
      --use_wandb WANDB_KEY \
      --target_module q_proj v_proj \
      --gradient_checkpointing
done

# ----------- Mix (datasets:vicgalle/alpaca-gpt4, Dahoas/rm-hh-rlhf) -----------

for lambd in $LAMBD_SET; do
   deepspeed --module xright.cli.train_sft_dpo_mix \
      --save_path ./checkpoint/Meta-Llama-3-8B-beta=$beta-learning_rate=$learning_rate-sft_dpo_mix-lambd-$lambd \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps 100 \
      --rlhf_dataset Dahoas/rm-hh-rlhf \
      --train_batch_size $train_batch_size \
      --rlhf_micro_train_batch_size $rlhf_micro_train_batch_size \
      --rlhf_max_samples $rlhf_max_samples \
      --rlhf_max_len $max_len \
      --rlhf_eval_split dummy \
      --sft_dataset vicgalle/alpaca-gpt4 \
      --sft_max_samples $sft_max_samples \
      --sft_max_len $max_len \
      --sft_micro_train_batch_size $sft_micro_train_batch_size \
      --input_key instruction \
      --lambd $lambd \
      --sft_opt $sft_opt \
      --dpo_opt $dpo_opt \
      --ref_pareto_sft "$ref_pareto_sft" \
      --ref_pareto_dpo "$ref_pareto_dpo" \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --ref_pretrain ./checkpoint/Meta-Llama-3-8B-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-sft-optim \
      --bf16 \
      --max_epochs $max_epochs \
      --zero_stage 2 \
      --beta $beta \
      --flash_attn \
      --learning_rate $learning_rate \
      --lora_rank 16 \
      --lora_alpha 16 \
      --use_wandb WANDB_KEY \
      --target_module q_proj v_proj \
      --gradient_checkpointing
done
