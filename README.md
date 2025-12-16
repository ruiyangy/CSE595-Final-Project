# Scientific Entity Recognition for ML Models and Datasets

A fine-grained Named Entity Recognition (NER) system for extracting machine learning models and datasets from computer science papers.

**EECS 595 Final Project** | University of Michigan

## Overview

This project addresses the challenge of automatically extracting ML-related entities (models, datasets, methods) from scientific literature. We fine-tune domain-specific transformers and implement a two-model strategy to handle nested entities.

### Key Results

| Model | Exact F1 | Partial F1 |
|-------|----------|------------|
| SciDeBERTa-CS | **69.1%** | **77.8%** |
| SciBERT | 66.4% | 74.2% |
| Dictionary Baseline | 24.3% | 28.7% |

## Approach

We use a **two-model strategy** to handle nested entity structures:
- **Model 1 (Generic)**: Identifies informal mentions (`MLModelGeneric`, `DatasetGeneric`)
- **Model 2 (Named)**: Identifies specific entities (`MLModel`, `Dataset`, `Task`, etc.)

At inference, predictions from both models are merged.

## Installation

```bash
# Clone repository
git clone https://github.com/ruiyangy/CSE595-Final-Project.git
cd CSE595-Final-Project

# Install dependencies
pip install transformers datasets pyyaml numpy pandas
```

## Usage

### Training

```bash
# Train Generic model
python train.py config_generic.yaml

# Train Named model  
python train.py config_named.yaml
```

### Inference

```bash
python inference.py config_inference.yaml
```

### Configuration

Edit `config_run.yaml` to customize:
- `subset_mode`: `"generic"` or `"named"`
- `base_model`: `"allenai/scibert_scivocab_uncased"` or SciDeBERTa-CS
- Training hyperparameters (learning rate, epochs, batch size)

## Project Structure

```
├── train.py              # Model training script
├── inference.py          # Inference pipeline
├── evaluation_methods.py # F1 score calculation
├── loader.py             # Data loading utilities
├── config_run.yaml       # Configuration template
├── train.sh              # SLURM job script (Great Lakes)
└── inference.sh          # SLURM inference script
```

## Dataset

This project uses the [GSAP-NER corpus](https://aclanthology.org/2023.findings-emnlp.545/) (Otto et al., 2023):
- 100 annotated CS papers
- 54,598 entity mentions across 10 entity types
- 10-fold cross-validation splits

## References

```bibtex
@inproceedings{otto-etal-2023-gsap,
    title = "{GSAP}-{NER}: A Novel Task, Corpus, and Baseline for Scholarly Entity Extraction",
    author = "Otto, Wolfgang and Zloch, Matth{\"a}us and Gan, Lu and Karmakar, Saurav and Dietze, Stefan",
    booktitle = "Findings of EMNLP 2023",
    year = "2023",
    pages = "8166--8176"
}
```

## Author

Ruiyang Yu — ruiyangy@umich.edu
