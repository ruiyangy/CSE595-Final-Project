from functools import partial
import random
from transformers import AutoTokenizer

# Defined entity groups based on your report
GENERIC_TAGS = {"MLModelGeneric", "DatasetGeneric"}
# All other tags (Named)
NAMED_TAGS = {
    "MLModel", "Dataset", "Task", "ModelArchitecture", 
    "Method", "DataSource", "URL", "ReferenceLink"
}

def prepare_dataset(dataset_raw, model_name, tagset, subset_mode="all"):
    """
    Args:
        subset_mode: "all", "generic", or "named". 
                     Filters labels to train specific models.
    """
    # Get the list of label names from the dataset features
    # Assuming features are like ClassLabel
    feature_name = f"{tagset}_label"
    if hasattr(dataset_raw["train"].features[feature_name], "feature"):
        ner_label_names = dataset_raw["train"].features[feature_name].feature.names
    else:
        ner_label_names = dataset_raw["train"].features[feature_name].names

    print(f"Original Labels: {ner_label_names}")

    # Determine which labels to keep based on the mode
    labels_to_keep = set()
    if subset_mode == "generic":
        labels_to_keep = GENERIC_TAGS
    elif subset_mode == "named":
        labels_to_keep = NAMED_TAGS
    else:
        labels_to_keep = set(ner_label_names)

    # Filter the label list for the model's configuration
    # We keep the original ID mapping but will mask out ignored labels during tokenization
    active_labels = [l for l in ner_label_names if l in labels_to_keep]
    print(f"Training on subset '{subset_mode}': {active_labels}")

    # Create BIO tags only for active labels
    # Note: We keep the full list logic to ensure ID consistency, 
    # but the tokenizer function will output 'O' for ignored classes.
    label_names = ["O"] + [f"B-{l}" for l in ner_label_names] + [f"I-{l}" for l in ner_label_names]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenized_datasets = dataset_raw.map(
        partial(
            tokenize_annotated,
            tokenizer=tokenizer,
            tagset=tagset,
            original_label_names=ner_label_names,
            labels_to_keep=labels_to_keep,
            num_tags=len(ner_label_names),
        ),
        batched=True,
        remove_columns=dataset_raw["train"].column_names,
    )

    return tokenized_datasets, label_names


def tokenize_annotated(examples, tokenizer, tagset, original_label_names, labels_to_keep, num_tags=0):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        is_split_into_words=False,
    )

    ner_start = examples[f"{tagset}_start"]
    ner_end = examples[f"{tagset}_end"]
    ner_labels = examples[f"{tagset}_label"]
    
    new_labels = []
    subword_token = []
    
    for idx, (starts, ends, labels) in enumerate(zip(ner_start, ner_end, ner_labels)):
        token = tokenized_inputs[idx]
        
        # Filter spans here!
        filtered_annos = []
        for s, e, l_id in zip(starts, ends, labels):
            label_str = original_label_names[l_id]
            if label_str in labels_to_keep:
                filtered_annos.append((s, e, l_id))
            # Else: ignore this span (treat as O)
            
        new_label, new_token = align_labels_with_tokens(token, filtered_annos, num_tags)
        new_labels.append(new_label)
        subword_token.append(new_token)

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["token"] = subword_token
    return tokenized_inputs


def align_labels_with_tokens(token, annos, num_tags):
    word_ids = token.word_ids
    subword_token = token.tokens
    
    # Initialize with -100 (ignore index for loss)
    all_label = [-100 for _ in word_ids]
    
    # Set valid tokens to 0 ("O") initially
    for i, w_id in enumerate(word_ids):
        if w_id is not None:
            all_label[i] = 0

    for start, end, label in annos:
        start_token = token.char_to_token(start)
        end_token = token.char_to_token(end - 1)
        
        if start_token is None or end_token is None:
            continue
            
        # Apply BIO tagging
        # B-Tag is label + 1
        # I-Tag is label + 1 + num_tags
        
        # Note: We blindly overwrite. Since nested entities are separated by model,
        # we don't need complex collision logic here (assumes spans in one subset don't overlap much).
        
        all_label[start_token] = label + 1
        for i in range(start_token + 1, end_token + 1):
            if all_label[i] != -100: # Don't overwrite special tokens
                all_label[i] = label + 1 + num_tags

    return all_label, subword_token
