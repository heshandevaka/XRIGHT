# ===========================================================Evaluation with MMLU benchmark===========================================================

ALGOS_1="seq-dpo seq-sft"
ALGOS_2="sft_dpo_alright sft_dpo_maxright sft_dpo_mix" 
NA_LIST="1 2 3"
LAMBD_LIST="0.25 0.5 0.75"

# for seq algorithms
for algo in $ALGOS_1; do
   for na_or_lambd in $NA_LIST; do
        python3 examples/MMLU_deep_eval_llama3.py --algo $algo --na_or_lambd $na_or_lambd
    done
 done

 # for moo algorithms
for algo in $ALGOS_2; do
   for na_or_lambd in $LAMBD_LIST; do
        python3 examples/MMLU_deep_eval_llama3.py --algo $algo --na_or_lambd $na_or_lambd
    done
 done