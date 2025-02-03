# only first two GPUs are compatible
export CUDA_VISIBLE_DEVICES=0,1

# ===========================================================Sequential training for SFT and RLHF===========================================================
NA_SET="1 2 3 4 5"
beta=0.1
lambd=0.5
learning_rate=5e-5
max_len=2048 
train_batch_size=2 # set to no. GPUS used
sft_max_samples=24000  
rlhf_max_samples=8000  
sft_micro_train_batch_size=12  
rlhf_micro_train_batch_size=4  
max_epochs=6


for na in $NA_SET; do
   # ----------------------------------------- SFT (dataset:vicgalle/alpaca-gpt4) -----------------------------------------
   deepspeed --module xright.cli.train_sft_seq \
      --save_path ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-seq-sft-temp \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps 100 \
      --train_batch_size $train_batch_size \
      --pretrain EleutherAI/pythia-1b \
      --ref_pretrain ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-sft-optim \
      --bf16 \
      --max_epochs $((6-$na)) \
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
      --input_key instruction \
      --flash_attn \
      --learning_rate $learning_rate \
      --lora_rank 32 \
      --lora_alpha 32 \
      --use_wandb WANDB_KEY \
      --target_module query_key_value
   # ----------------------------------------- DPO (preference data trained reference model, dataset:Dahoas/rm-hh-rlhf) -----------------------------------------
   #   
   deepspeed --module xright.cli.train_dpo_seq \
      --save_path ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-seq-dpo-temp \
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
      --pretrain ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-seq-sft-temp \
      --ref_pretrain ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-sft-optim \
      --bf16 \
      --max_epochs $na \
      --zero_stage 2 \
      --beta $beta \
      --flash_attn \
      --learning_rate $learning_rate \
      --lora_rank 32 \
      --lora_alpha 32 \
      --use_wandb WANDB_KEY \
      --target_module query_key_value

done