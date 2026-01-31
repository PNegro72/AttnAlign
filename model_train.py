import pandas as pd
from transformers import AutoTokenizer
from llama import BasicModelRunner
import utilities as ut

model_name = "EleutherAI/pythia-70m"

dataset_name = "edskbtotal_proc_prepared_new.jsonl"
# dataset_name = "lamini_docs.jsonl"
dataset_path = f"Data/{dataset_name}"
use_hf = False

training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

def split_data():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = ut.tokenize_and_split_data(training_config, tokenizer)

    print(train_dataset)
    print(test_dataset)

def main():
    split_data()

if __name__ == "__main__":
    main()
