import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import json
import transformers
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

import random
import numpy as np
from datetime import datetime
import os

import argparse


parser = argparse.ArgumentParser(description="Deep Eval MMLU benchmark for trained LLAMA3 models")

parser.add_argument('--filename', type=str, default='MMLU_results', help='The name of the output file')
parser.add_argument('--algo', type=str, default='meta-llama/Meta-Llama-3-8B', help='The algo used to train the model')
parser.add_argument('--na_or_lambd', type=str, default="1", help='The variant of the algo')

args = parser.parse_args()


# setting seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False  



class CustomLlama3_8B(DeepEvalBaseLLM):
    def __init__(self, pretrain, adapter=None):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_4bit = AutoModelForCausalLM.from_pretrained(
            pretrain,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrain
        )

        if adapter is not None:
            print()
            print(f"Loading adapter from {adapter} ....")
            model_4bit = PeftModel.from_pretrained(model_4bit, adapter)
            model_4bit = model_4bit.merge_and_unload()    
            print("Done!")
            print()     

        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )

        # Output and load valid JSON
        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt) :]
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Llama-3 8B"

if args.algo=='seq-dpo' or args.algo=='seq-sft':
    pretrain=f"./checkpoint/Meta-Llama-3-8B-beta=0.1-learning_rate=5e-5-{args.algo}-na-{args.na_or_lambd}"
else:
    pretrain=f"./checkpoint/Meta-Llama-3-8B-beta=0.1-learning_rate=5e-5-{args.algo}-lambd-{args.na_or_lambd}"


llama3 = CustomLlama3_8B(pretrain, pretrain)


from deepeval.benchmarks import MMLU 
from deepeval.benchmarks.mmlu.task import MMLUTask

# Create and evaluate using MMLU benchmark
benchmark = MMLU(n_shots=1)
results = benchmark.evaluate(model=llama3)

# Log results
current_time = datetime.now()
loggable_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
if not os.path.exists('../MMLU_results/'):
    os.makedirs('../MMLU_results/')
with open(f'../MMLU_results/{args.filename}_{args.algo}_{args.na_or_lambd}.txt', 'a') as text_file:
    text_file.write(f"\n{loggable_time}\n")
    text_file.write(f"{pretrain}\n")
    text_file.write(f"Overall Score:\n{benchmark.overall_score}\n")
    for index, row in benchmark.task_scores.iterrows():
        task = row['Task']
        score = row['Score']
        text_file.write(f"{task}:{score}\n")

print("Overall Score: ", results)
print(benchmark.overall_score)
print(benchmark.task_scores)
preds = benchmark.predictions
print("Sample Input: ", preds.iloc[1]["Input"])
print("Sample Predictions: ", preds.iloc[1]["Prediction"])