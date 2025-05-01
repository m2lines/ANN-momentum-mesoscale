#!/bin/bash

module load cdo/intel/1.9.10

# Define base and destination prefixes
BASE_PREFIX="/vast/pp2681/CM26_datasets/ocean3d/subfilter-large/FGR3"
SAVE_PREFIX="/scratch/pp2681/GRL-dataset/FGR3"

# List of factors
factors=("factor-4" "factor-9" "factor-12" "factor-15")

# Loop through each factor
for factor in "${factors[@]}"; do
    base_path="${BASE_PREFIX}/${factor}"
    save_path="${SAVE_PREFIX}/${factor}"

    # Create save_path if it doesn't exist
    mkdir -p "$save_path"

    echo "Processing $factor..."

    # Combine train/test/validate NetCDF files
    cdo cat "${base_path}/test-"*.nc "${save_path}/test-combined.nc"
    cdo cat "${base_path}/train-"*.nc "${save_path}/train-combined.nc"
    cdo cat "${base_path}/validate-"*.nc "${save_path}/validate-combined.nc"

    # Copy metadata and configuration files
    cp "${base_path}/filter.txt" "$save_path/"
    cp "${base_path}/param.nc" "$save_path/"
    cp "${base_path}/permanent_features.nc" "$save_path/"

    echo "Finished processing $factor."
done

