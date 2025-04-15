#!/bin/bash
#SBATCH --job-name=Q3A
#SBATCH --output=Q3A.out
#SBATCH --error=Q3A.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=macs30123
#SBATCH --mail-type=END  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=zw2685@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

module load python
module load cuda/11.7
module load gcc

python q3a.py