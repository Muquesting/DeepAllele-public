#!/bin/bash
#SBATCH --job-name=cnn_model
#SBATCH --partition=gpu-a40
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wz86@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
source /gscratch/mostafavilab/wz86/miniconda3/etc/profile.d/conda.sh

conda activate DeepAllele

python train_model.py --out_folder='../output/separate_tf/' --model_type=Separate_Transformer --attention_layers=2 --conv_layers=6 --conv_repeat=1 --kernel_number=1024 --kernel_length=7 --filter_number=512 --kernel_size=5 --pooling_size=2