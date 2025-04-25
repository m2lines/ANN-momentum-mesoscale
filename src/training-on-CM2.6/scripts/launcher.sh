#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --begin=now
#SBATCH --time=48:00:00
#SBATCH --job-name=ANN_training

### For training: ntasks=4, mem=64GB, time=48:00:00
### For filtering: --cpus-per-task=14 --mem=64GB time=48:00:00
### For training fluxes:  --cpus-per-task=8 --mem=30GB time=06:00:00 (estimated training time is 3.5 hours + 30 mins on testing) -- this is preferable because on one node 6 ANNs can be trained simultaneously


#singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u generate_3d_datasets.py --factor=9 "

#singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u train_script_fluxes.py --hidden_layers=\"[16,8]\" --path_save=flux-models/16-8-seed1 "

singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u train_script.py --dimensional_scaling=False --hidden_layers=\"[32,32]\" --path_save=dimensional-scaling/no-scaling-32-32 "
