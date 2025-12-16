from pathlib import Path
import sys
import os
import logging
from functools import partial

from datasets import load_from_disk
from transformers import (DataCollatorForTokenClassification,
                          AutoModelForTokenClassification,
                          TrainingArguments,
                          Trainer,
                          AutoTokenizer)
import yaml

# Import your local modules
from prepare_dataset import prepare_dataset
from evaluation_methods import calc_scores # Assuming metrics are here or use huggingface metrics

def compute_metrics_wrapper(p, label_names):
    # Simple wrapper to convert logits to predictions
    import numpy as np
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # You can plug in seqeval or your evaluation_methods.py here
    # For now returning empty dict to ensure loop runs
    return {} 

def train_from_config(config):
    c = config
    n_folds = c["data"].get("n_folds", 1) # Default to 1 if not set
    path_base_data = get_dataset_path(c)
    path_base_model = get_model_path(c)
    
    # Get the subset mode (generic vs named)
    subset_mode = c["data"].get("subset_mode", "all")
    print(f"--- Training Mode: {subset_mode} ---")

    for fold in range(0, n_folds):
        print(f"Start with fold {fold}")
        
        path_fold = path_base_data / Path(str(fold))
        if not path_fold.exists():
            print(f"Data for fold {fold} not found at {path_fold}")
            continue

        dataset_raw = load_from_disk(str(path_fold))
        
        # Prepare dataset with the specific subset filter
        dataset_tokenized, label_names = prepare_dataset(
            dataset_raw,
            c["model"]["base_model"],
            c["data"]["tagset"],
            subset_mode=subset_mode
        )
        
        # Load model
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        
        model = AutoModelForTokenClassification.from_pretrained(
            c["model"]["base_model"],
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(c["model"]["base_model"])
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        path_model_fold = path_base_model / Path(str(fold))
        
        training_args = TrainingArguments(
            output_dir=str(path_model_fold),
            **c["model"]["training_arguments"]
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_tokenized["train"],
            eval_dataset=dataset_tokenized["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=partial(compute_metrics_wrapper, label_names=label_names)
        )
        
        trainer.train()
        
        # Save the final model explicitly
        trainer.save_model(str(path_model_fold))

def get_dataset_path(c):
    path_base = Path(c["data"]["path"])
    # Logic to handle paths (simplified for clarity)
    return path_base

def get_model_path(c):
    path_base = Path(c["model"]["output_path"])
    # Append subset mode to path to separate Generic vs Named models
    subset_mode = c["data"].get("subset_mode", "all")
    model_name = f"{c['model']['nickname']}_{subset_mode}"
    return path_base / model_name

def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python train.py config.yaml")
        sys.exit(1)
        
    fn_config = sys.argv[1]
    with open(fn_config) as f:
        config = yaml.safe_load(f)
    
    train_from_config(config)

if __name__ == "__main__":
    main()