# only first two GPUs are compatible
export CUDA_VISIBLE_DEVICES=0,1

# ===========================================================Joint training for SFT and RLHF===========================================================

LAMBD_SET="0.01 0.25 0.5 0.75 0.99"
beta=0.1
learning_rate=5e-5
max_len=2048 
train_batch_size=2 # set to no. GPUS used
sft_max_samples=24000  
rlhf_max_samples=8000  
sft_micro_train_batch_size=12  
rlhf_micro_train_batch_size=4  
max_epochs=6
max_eval_steps=10 
sft_opt=1.4980
dpo_opt=0.0647
ref_pareto_sft='1.8898 1.631 1.5889 1.5474 1.4998'
ref_pareto_dpo='0.0659 0.1753 0.3331 0.5559 1.0433'


# ----------- ALRIGHT (datasets:vicgalle/alpaca-gpt4, Dahoas/rm-hh-rlhf) -----------

for lambd in $LAMBD_SET; do
   deepspeed --module xright.cli.train_sft_dpo_alright \
      --save_path ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-sft_dpo_alright-temp \
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
      --lambd $lambd \
      --sft_opt $sft_opt \
      --dpo_opt $dpo_opt \
      --ref_pareto_sft "" \
      --ref_pareto_dpo "" \
      --pretrain EleutherAI/pythia-1b \
      --ref_pretrain ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-sft-pref-optim  \
      --bf16 \
      --max_epochs $max_epochs \
      --zero_stage 2 \
      --beta $beta \
      --flash_attn \
      --learning_rate $learning_rate \
      --lora_rank 32 \
      --lora_alpha 32 \
      --use_wandb WANDB_KEY \
      --target_module query_key_value
done

# ----------- MAXRIGHT (datasets:vicgalle/alpaca-gpt4, Dahoas/rm-hh-rlhf) -----------

for lambd in $LAMBD_SET; do
   deepspeed --module xright.cli.train_sft_dpo_maxright \
      --save_path ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-sft_dpo_maxright-temp \
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
      --lambd $lambd \
      --sft_opt $sft_opt \
      --dpo_opt $dpo_opt \
      --ref_pareto_sft "$ref_pareto_sft" \
      --ref_pareto_dpo "$ref_pareto_dpo" \
      --max_eval_steps $max_eval_steps \
      --pretrain EleutherAI/pythia-1b \
      --ref_pretrain ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-sft-pref-optim  \
      --bf16 \
      --max_epochs $max_epochs \
      --zero_stage 2 \
      --beta $beta \
      --flash_attn \
      --learning_rate $learning_rate \
      --lora_rank 32 \
      --lora_alpha 32 \
      --use_wandb WANDB_KEY \
      --target_module query_key_value
done

# ----------- Mix (datasets:vicgalle/alpaca-gpt4, Dahoas/rm-hh-rlhf) -----------

for lambd in $LAMBD_SET; do
   deepspeed --module xright.cli.train_sft_dpo_mix \
      --save_path ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-sft_dpo_mix-temp \
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
      --lambd $lambd \
      --sft_opt $sft_opt \
      --dpo_opt $dpo_opt \
      --ref_pareto_sft "$ref_pareto_sft" \
      --ref_pareto_dpo "$ref_pareto_dpo" \
      --pretrain EleutherAI/pythia-1b \
      --ref_pretrain ./checkpoint/pythia-1b-beta=$beta-learning_rate=$learning_rate-epoch=$max_epochs-sft-pref-optim  \
      --bf16 \
      --max_epochs $max_epochs \
      --zero_stage 2 \
      --beta $beta \
      --flash_attn \
      --learning_rate $learning_rate \
      --lora_rank 32 \
      --lora_alpha 32 \
      --use_wandb WANDB_KEY \
      --target_module query_key_value
done