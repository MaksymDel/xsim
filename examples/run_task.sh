#!/bin/bash

#The name of the job is train
#SBATCH -J interpert

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is 8 days
#SBATCH -t 192:00:00

#SBATCH --mem=120GB

#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

# --gres=gpu:a100-80g

module load any/python/3.8.3-conda
module load cuda/11.3.1
conda activate paper3

# python -u encode_dataset_with_models.py mT5
# python -u encode_dataset_with_models.py xlmr
# python -u encode_dataset_with_models.py xglm

# python -u run_analysis.py mT5 acc
# python -u run_analysis.py mT5 acc-cent
# python -u run_analysis.py mT5 acc-procrustes
# python -u run_analysis.py mT5 cka

# python -u run_analysis.py xlmr acc
# python -u run_analysis.py xlmr acc-cent
# python -u run_analysis.py xlmr acc-procrustes
# python -u run_analysis.py xlmr cka

# python -u run_analysis.py xglm acc
# python -u run_analysis.py xglm acc-cent
# python -u run_analysis.py xglm acc-procrustes
# python -u run_analysis.py xglm cka


