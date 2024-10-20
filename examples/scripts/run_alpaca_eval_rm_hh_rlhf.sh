set -x 
export OPENAI_API_KEY='your-OpenAI-key'
export PATH=$HOME/.local/bin/:$PATH

ALGOS_1="seq-dpo seq-sft"
ALGOS_2="sft_dpo_alright sft_dpo_maxright sft_dpo_mix" 
NA_LIST="1 2 3"
LAMBD_LIST="0.25 0.5 0.75"
model=Meta-Llama-3-8B

for algo in $ALGOS_1; do
   for na in $NA_LIST; do
        echo $model-$algo-na=$na
        alpaca_eval --model_outputs ./model_outputs/$model-$algo-na=$na-rm-hh-rlhf-test-out.json --reference_outputs ./xright/rm_hh_rlhf_test_subset_formatted.json  --annotators_config alpaca_eval_gpt4_turbo_fn
    done
done

for algo in $ALGOS_2; do
   for lambd in $LAMBD_LIST; do
    echo $model-$algo-lambd=$lambd
    alpaca_eval --model_outputs ./model_outputs/$model-$algo-lambd=$lambd-rm-hh-rlhf-test-out.json --reference_outputs ./xright/rm_hh_rlhf_test_subset_formatted.json  --annotators_config alpaca_eval_gpt4_turbo_fn
    done
done