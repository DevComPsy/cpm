#!/bin/bash
#SBATCH --job-name=cpm-fit
#SBATCH --array=0-9               # 10 tasks, each fitting a tenth of the participants
#SBATCH --cpus-per-task=16        # cores per task, used to fit participants in parallel
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00           # hours:minutes:seconds, per task
#SBATCH --output=logs/%x_%A_%a.out

# load Python and activate the environment in which cpm is installed;
# the module name and the path depend on your cluster
module load python
source ~/venvs/cpm/bin/activate

python fit_chunk.py
