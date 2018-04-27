#!/bin/bash
# 
#SBATCH --job-name=rcnn
#SBATCH --nodes=1 
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=50GB 
#SBATCH --mem-per-cpu=50GB 
#SBATCH --gres=gpu:4 

cd /scratch/ss8464/setup
source /scratch/ss8464/setup/bin/activate
module purge
module load  tensorflow/python3.6/1.5.0 
python train.py