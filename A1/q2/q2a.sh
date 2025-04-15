#!/bin/bash
#SBATCH --job-name=q2a
#SBATCH --output=q2a.out
#SBATCH --error=q2a.err

#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --time=01:00:00       # Adjust time as needed
#SBATCH --ntasks=10           # Reserve up to 20 tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

#SBATCH --mail-type=END  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=zw2685@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

# Load Python and MPI modules
module load python
module load mpich

python q2a_aot.py build_ext --inplace

# Run the MPI grid search
mpirun -n 10 python q2a.py