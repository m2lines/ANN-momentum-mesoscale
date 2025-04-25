#!/bin/bash
#SBATCH --nodes=45
#SBATCH --ntasks-per-node=25
#SBATCH --cpus-per-task=1
#SBATCH --mem=46GB # i.e., 1.66 GB per core
#SBATCH --time=48:00:00
#SBATCH --job-name=OM4AIF

scontrol show jobid -dd $SLURM_JOB_ID
# These are the same environment variables
# as used for compilation of MOM6
module purge
module load intel/19.1.2
module load openmpi/intel/4.0.5
module unload netcdf
module load netcdf-fortran/intel/4.5.3
module load hdf5/intel/1.12.0

EXECUTABLE="MOM6_SIS2_nonsymmetric_2024_Sep_04"

for year in {1..10}; do
    # Only move files from RESTART to INPUT if there are files in RESTART
    if [ "$(ls -A RESTART)" ]; then
        echo "Checking year in RESTART file for possible saving..."
        for restart_year in 1978 2003; do
            if grep -q "$restart_year" RESTART/coupler.res; then
	            echo "Copying RESTART folder to RESTART-$restart_year-01-01"
                cp -r RESTART "RESTART-$restart_year-01-01"
            fi
        done
        echo "Moving files from RESTART to INPUT and deleting old INPUT files..."
        rm INPUT/*.res*
        mv RESTART/* INPUT/
    else
        echo "No files in RESTART to move. Continuing to run..."
    fi

    # Run the executable
    srun ./$EXECUTABLE

    # Check if srun was successful
    if [ $? -ne 0 ]; then
        echo "Error detected in srun. Terminating script"
        exit 1
    fi
done