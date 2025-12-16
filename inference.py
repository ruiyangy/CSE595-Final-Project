import time
import sys
import logging
import json
from pathlib import Path
import yaml
import numpy as np
from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer

# Assuming loader.py is available
import loader 

def run(cfg):
    run_standard(cfg)

def run_standard(cfg):
    batch_size = cfg["models"].get("batch_size", 8)
    
    # Load Documents
    print("Loading documents...")
    # Assuming loader.load works as provided
    doc_iter = loader.load(cfg["input"]["path"], limit=cfg["input"].get("limit"))
    
    # Load Pipelines (This is where the Two-Model Magic happens)
    print("Loading models (Generic + Named)...")
    pipelines = get_pipes(cfg)
    
    infer_and_save(doc_iter, pipelines, cfg, batch_size)

def infer_and_save(doc_iter, pipelines, cfg, batch_size):
    all_predictions = []
    
    while True:
        docs = get_batch(doc_iter, batch_size)
        if not docs:
            break
            
        print(f"Processing batch of {len(docs)} documents")
        texts = [d["text"] for d in docs]
        
        # Predict
        # This will run Model 1, then Model 2, and merging happens inside predict_ents
        predictions = predict_ents(docs, pipelines)
        
        # Save
        if cfg["output"].get("onefile", True):
            all_predictions.extend(predictions)
        else:
            loader.dump(predictions, cfg["output"]["path"])

    if cfg["output"].get("onefile", True):
        out_path = Path(cfg["output"]["path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_predictions, f)

def get_pipes(cfg):
    device = cfg.get("device", -1) # -1 for CPU, 0 for GPU
    pipes = []
    
    # Iterate over the defined steps (e.g., [GenericModel, NamedModel])
    for model_cfg in cfg["models"]["steps"]:
        path = model_cfg["path"]
        print(f"Loading model from: {path}")
        
        # Load model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=512)
        
        # Create HF pipeline
        pipe = pipeline("ner", model=model, tokenizer=tokenizer, device=device, aggregation_strategy="none")
        pipes.append(pipe)
        
    return pipes

def predict_ents(docs, pipelines):
    texts = [d["text"] for d in docs]
    
    # 1. Run all pipelines on the texts
    all_pipe_results = []
    for pipe in pipelines:
        # returns list of list of dicts
        res = pipe(texts)
        # normalize to always be list of lists
        if isinstance(res[0], dict): 
             res = [res] 
        all_pipe_results.append(res)
        
    # 2. Merge predictions into the doc structure
    # We iterate over documents
    for i, doc in enumerate(docs):
        combined_spans = []
        
        # Iterate over each model's result for this document
        for pipe_res in all_pipe_results:
            doc_ specific_res = pipe_res[i] # annotations for doc i from model j
            spans = prediction_to_spans(doc_specific_res)
            combined_spans.extend(spans)
            
        # Add to document dictionary
        doc["pred_spans"] = combined_spans
        
    return docs

def prediction_to_spans(predictions):
    # Convert HF pipeline output to simple span dicts
    # HF Output: [{'entity': 'B-MLModel', 'score': 0.99, 'index': 1, 'word': 'Bert', 'start': 0, 'end': 4}, ...]
    # We need to reconstruct full spans from B/I tags or use aggregation_strategy="simple" in pipeline
    
    # Simplified approach: Using aggregation_strategy="none" (raw) requires logic.
    # Ideally, use aggregation_strategy="simple" in get_pipes to get full chunks.
    # Assuming "simple" strategy for cleaner code:
    # Output: [{'entity_group': 'MLModel', 'score': 0.99, 'word': 'Bert', 'start': 0, 'end': 4}]
    
    # If we stick to the provided code's manual reconstruction:
    spans = []
    current_span = None
    
    for tok in predictions:
        tag = tok['entity']
        if tag.startswith('B-'):
            if current_span:
                spans.append(current_span)
            current_span = {
                "label": tag[2:],
                "start": tok['start'],
                "end": tok['end'],
                "score": float(tok['score'])
            }
        elif tag.startswith('I-') and current_span:
            if tag[2:] == current_span["label"]:
                current_span["end"] = tok['end']
                # Average score?
                current_span["score"] = (current_span["score"] + float(tok['score'])) / 2
            else:
                # Broken sequence
                spans.append(current_span)
                current_span = None
        else:
            if current_span:
                spans.append(current_span)
                current_span = None
                
    if current_span:
        spans.append(current_span)
        
    return spans

def get_batch(doc_iter, batch_size):
    docs = []
    try:
        for _ in range(batch_size):
            docs.append(next(doc_iter))
    except StopIteration:
        pass
    return docs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py config.yaml")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    run(config)