#!/bin/bash
#SBATCH --job-name=q1a
#SBATCH --output=q1a.out
#SBATCH --error=q1a.err

#SBATCH --account=macs30123
#SBATCH --partition=caslake

#SBATCH --time=00:10:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mail-type=END  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=zw2685@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"
# Load required modules
module load python

python q1a_original.py

python q1a_aot.py build_ext --inplace

# Run the simulation script
python q1a_precompiled.py
