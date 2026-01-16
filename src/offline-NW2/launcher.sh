#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --begin=now
#SBATCH --time=00:60:00
#SBATCH --job-name=NW2flt

# Usage of the script sbatch --array=0-1900 launcher.sh. In total --array=0-2399 is needed but prohibited by scheduling system

echo " "
scontrol show jobid -dd $SLURM_JOB_ID
echo " "
echo "The number of alphafold processes:"
ps -e | grep -i alphafold | wc -l
echo " "
module purge

# 2D index
i=$SLURM_ARRAY_TASK_ID
time_idx=$(( i / 15 ))
zl_idx=$(( i % 15 ))

echo "SLURM task: $i  →  zl_idx=$zl_idx  time_idx=$time_idx"

singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u filter-NW2-data.py --zl_idx=${zl_idx} --time_idx=${time_idx} "

#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u filter-interfaces-GM-filter.py --zl_idx=${zl_idx} --time_idx=${time_idx} "
