#!/bin/bash                                                                     
#                                                                               
#SBATCH --job-name=trainVGG                                                     
#SBATCH --nodes=2                                                               
#SBATCH --time=24:00:00                                                         
#SBATCH --cpus-per-task=4                                                       
#SBATCH --mem=150GB                                                             
#SBATCH --mem-per-cpu=150GB                                                     
#SBATCH --gres=gpu:4                                                            

module purge
module load pillow/python3.5/intel/4.2.1
module load  tensorflow/python3.5/1.4.0
cd /home/ss8464/DL_Project
python3 vgg.py
