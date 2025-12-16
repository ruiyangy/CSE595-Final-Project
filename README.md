# Scientific Entity Recognition for ML Models and Datasets

A fine-grained Named Entity Recognition (NER) system for extracting machine learning models and datasets from computer science papers.

**EECS 595 Final Project**

## Overview

This project addresses the challenge of automatically extracting ML-related entities (models, datasets, methods) from scientific literature. We fine-tune domain-specific transformers and implement a two-model strategy to handle nested entities.

## Approach

We use a **two-model strategy** to handle nested entity structures:
- **Model 1 (Generic)**: Identifies informal mentions (`MLModelGeneric`, `DatasetGeneric`)
- **Model 2 (Named)**: Identifies specific entities (`MLModel`, `Dataset`, `Task`, etc.)

At inference, predictions from both models are merged.

## Installation

# Install dependencies
pip install transformers datasets numpy pandas
```

## Usage

### Training

```bash
# Train Generic model
python train.py

# Train Named model  
python train.py
```

### Inference

```bash
python inference.py config_inference.yaml
```

## Dataset

This project uses the [GSAP-NER corpus]:
- 100 annotated CS papers
- 54,598 entity mentions across 10 entity types
- 10-fold cross-validation splits
