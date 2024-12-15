#!/bin/bash

# to launch this training job on SuperCloud:
# LLsub diffunc/train/satellite/satellite_128px.sh -g volta:1

# Slurm sbatch options for SuperCloud
#SBATCH  --gres=gpu:volta:1
#SBATCH  --mem=128G

# Run the script
./diffunc/train/satellite/satellite_128px.py
