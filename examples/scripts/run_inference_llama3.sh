# only these GPUs are compatible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ===========================================================Inference with trained models===========================================================

ALGOS_1="seq-dpo seq-sft"
ALGOS_2="sft_dpo_alright sft_dpo_maxright sft_dpo_mix" 
NA_LIST="1 2 3"
LAMBD_LIST="0.25 0.5 0.75"

max_len=2048 
max_samples=640
micro_batch_size=16 
model=Meta-Llama-3-8B 

# for seq algorithms
for algo in $ALGOS_1; do
   for na in $NA_LIST; do
      deepspeed --module xright.cli.batch_inference \
         --eval_task generate_alpaca_eval \
         --dataset ./model_outputs/rm_hh_rlhf_test_subset_formatted.json \
         --dataset_split train \
         --input_key instruction \
         --output_key output \
         --micro_batch_size $micro_batch_size \
         --max_samples $max_samples \
         --max_len $max_len \
         --pretrain ./checkpoint/$model-beta=0.1-learning_rate=5e-5-$algo-na-$na \
         --output_path ./model_outputs/$model-$algo-na=$na-rm-hh-rlhf-test-out.json \
         --bf16 \
         --zero_stage 0 \
         --flash_attn
      done
done

# for moo algorithms
for algo in $ALGOS_2; do
   for lambd in $LAMBD_LIST; do
      deepspeed --module xright.cli.batch_inference \
         --eval_task generate_alpaca_eval \
         --dataset ./model_outputs/rm_hh_rlhf_test_subset_formatted.json \
         --dataset_split train \
         --input_key instruction \
         --output_key output \
         --micro_batch_size $micro_batch_size \
         --max_samples $max_samples \
         --max_len $max_len \
         --pretrain ./checkpoint/$model-beta=0.1-learning_rate=5e-5-$algo-lambd-$lambd \
         --output_path ./model_outputs/$model-$algo-lambd=$lambd-rm-hh-rlhf-test-out.json \
         --bf16 \
         --zero_stage 0 \
         --flash_attn
   done
done

