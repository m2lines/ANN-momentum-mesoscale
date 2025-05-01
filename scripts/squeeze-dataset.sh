#!/bin/bash

module load cdo/intel/1.9.10

# Define base and destination prefixes
BASE_PREFIX="/scratch/pp2681/GRL-dataset/FGR3"

cd /scratch/pp2681/GRL-dataset/FGR3-squeeze

# List of factors
factors=("factor-4")
datasets=("validate" "test" "train")

# Loop through each factor
for factor in "${factors[@]}"; do
    base_path="${BASE_PREFIX}/${factor}"
    save_path="${factor}"

    # Create save_path if it doesn't exist
    mkdir -p "$save_path"

    echo "Processing $factor..."
    
    for key in "${datasets[@]}"; do
        # Select variables
        cdo selname,u,v,SGSx,SGSy,Txx,Txy,Tyy,rel_vort_h,sh_xy_h,sh_xx,div,deformation_radius,deltaU,NH "${base_path}/${key}-combined.nc" "${save_path}/${key}-tmp.nc"
        # Sort time
        cdo sorttimestamp "${save_path}/${key}-tmp.nc" "${save_path}/${key}.nc"
        # Remove temporary data
        rm "${save_path}/${key}-tmp.nc"
        # pack one dataset to one 
        tar -czf "${factor}-${key}.tar.gz" "${save_path}/${key}.nc"
        rm "${save_path}/${key}.nc"
    done
    # Copy metadata and configuration files
    cp "${base_path}/filter.txt" "$save_path/"
    cp "${base_path}/param.nc" "$save_path/"
    cp "${base_path}/permanent_features.nc" "$save_path/"

    tar -czf "${factor}-metadata.tar.gz" "${save_path}"
    rm -r "${save_path}"
    
    echo "Finished processing $factor."
done

