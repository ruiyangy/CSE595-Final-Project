#!/bin/bash

#SBATCH --job-name=ner-train
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00        # Adjusted time (fine-tuning is faster than pre-training)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=ner-train-%j.out

# Load mamba and activate environment
module load mamba
# Make sure to use the environment where you installed transformers, datasets, etc.
conda activate hw3_new 

# Verify environment
echo "==================================="
echo "Environment Verification:"
echo "==================================="
which python
python --version
python -c 'import torch; print("PyTorch:", torch.__version__)'
python -c 'import transformers; print("Transformers:", transformers.__version__)'
echo "==================================="

# Define Paths
# Adjust these to where you actually put your project files on Great Lakes
export PROJECT_ROOT="/home/ruiyangy/FinalProject" 
export DATA_PATH="$PROJECT_ROOT/data/gsap_ner_processed"
export CONFIG_DIR="$PROJECT_ROOT/configs"

# Move to project directory
cd $PROJECT_ROOT

# --- STEP 1: Train Generic Model ---
echo "-----------------------------------"
echo "Starting GENERIC Model Training..."
echo "-----------------------------------"

# Ensure the config exists (you need to create this file based on my previous template)
if [ ! -f "$CONFIG_DIR/train_generic.yaml" ]; then
    echo "Error: $CONFIG_DIR/train_generic.yaml not found!"
    exit 1
fi

python train.py "$CONFIG_DIR/train_generic.yaml"

# --- STEP 2: Train Named Model ---
echo "-----------------------------------"
echo "Starting NAMED Model Training..."
echo "-----------------------------------"

if [ ! -f "$CONFIG_DIR/train_named.yaml" ]; then
    echo "Error: $CONFIG_DIR/train_named.yaml not found!"
    exit 1
fi

python train.py "$CONFIG_DIR/train_named.yaml"

echo "==================================="
echo "Training completed for both models!"
echo "==================================="