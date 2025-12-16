#!/bin/bash

#SBATCH --job-name=gpt-pretrain
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=pretrain-%j.out

# Load mamba and activate environment
module load mamba
conda activate hw3_new

# Verify environment (using single quotes to avoid bash issues)
echo "==================================="
echo "Environment Verification:"
echo "==================================="
which python
python --version
python -c 'import torch; print("PyTorch:", torch.__version__)'
python -c 'from torch.nn import RMSNorm; print("RMSNorm: Available")'
echo "==================================="

# Start training
echo "Starting full GPT training..."
echo "=================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="eecs595-gpt-pretraining"
export DATA_PATH="/home/ruiyangy/Homework3/Data"
export OUTPUT_DIR="/home/ruiyangy/Homework3/models/pretrained-models/"
export TOKENIZERS_PARALLELISM=false

python pretrain_gpt.py \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --max_epochs 1 \
    --emb_dim 512 \
    --n_layers 12 \
    --n_heads 8 \
    --context_length 1024 \
    --save_every 1000 \
    --eval_every 500 \
    --device cuda \
    --data_path $DATA_PATH/fineweb-edu-sample-1B-hf/ \
    --data_format arrow \
    --output_dir $OUTPUT_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "gpt-pretraining-$(date +%Y%m%d-%H%M%S)"

echo "Training completed!"
echo "Check the output directory for saved models and logs."
