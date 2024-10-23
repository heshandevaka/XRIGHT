# only GPUs compatible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ===========================================================Train models using independent objectives to find approx. objective optima===========================================================

beta=0.1
learning_rate=5e-5
max_len=2048
train_batch_size=4 
sft_max_samples=24000  
rlhf_max_samples=6000 
sft_micro_train_batch_size=4 
rlhf_micro_train_batch_size=1
max_epochs=1

# ----------------------------------------- SFT (dataset:vicgalle/alpaca-gpt4) -----------------------------------------
deepspeed --module xright.cli.train_sft_seq \
   --save_path ./checkpoint/Meta-Llama-3-8B-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-sft-optim \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size $train_batch_size \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --bf16 \
   --max_epochs $max_epochs \
   --zero_stage 2 \
   --rlhf_dataset Dahoas/rm-hh-rlhf \
   --rlhf_max_samples $rlhf_max_samples \
   --rlhf_max_len $max_len \
   --rlhf_beta $beta \
   --rlhf_eval_split dummy \
   --rlhf_micro_train_batch_size $rlhf_micro_train_batch_size \
   --sft_dataset vicgalle/alpaca-gpt4 \
   --sft_max_samples $sft_max_samples \
   --sft_max_len $max_len \
   --sft_micro_train_batch_size $sft_micro_train_batch_size \
   --flash_attn \
   --learning_rate $learning_rate \
   --lora_rank 16 \
   --lora_alpha 16 \
   --use_wandb WANDB_KEY \
   --target_module q_proj v_proj \
   --gradient_checkpointing

# ----------------------------------------- DPO (dataset:Dahoas/rm-hh-rlhf) -----------------------------------------    
deepspeed --module xright.cli.train_dpo_seq \
   --save_path ./checkpoint/Meta-Llama-3-8B-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-dpo-sft-ref-optim \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
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
   --gradient_checkpointing \

