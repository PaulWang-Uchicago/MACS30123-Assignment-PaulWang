#!/bin/bash
#SBATCH --job-name=q1b
#SBATCH --output=q1b.out
#SBATCH --error=q1b.err

#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --time=02:00:00       # Adjust time as needed
#SBATCH --ntasks=20           # Reserve up to 20 tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

#SBATCH --mail-type=END  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=zw2685@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

# Load necessary modules
module load python
module load mpich

python q1a_aot.py build_ext --inplace

# Output file for storing times
RESULT_FILE="q1b_times.out"
echo "Scaling study results" > $RESULT_FILE

# Loop over core counts 1..20
for i in {1..20}
do
    echo "------------------------------" >> $RESULT_FILE
    echo "Running with $i MPI processes:" >> $RESULT_FILE

    mpirun -n "$i" python3 q1b_rank0.py >> $RESULT_FILE
done

python q1b_plot.py