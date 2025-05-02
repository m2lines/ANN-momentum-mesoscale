# ANN parameterization of mesoscale momentum fluxes in ocean model MOM6
This repository contains training algorithm and MOM6 ocean model with implemented parameterization. Additionally, we include figure plotting notebooks for the paper "Generalizable neural-network parameterization of mesoscale eddies in idealized and global ocean models" by Pavel Perezhogin, Laure Zanna and Alistair Adcroft, to be submitted soon.

Example of online simulation with the proposed ANN parameterization in the idealized configuration NeverWorld2 of the MOM6 ocean model ([full movie](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/NW2-mp4.mp4), [code](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/NW2-movie.ipynb)):

<div align="center">
  <img src="https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/NW2-gif.gif" alt="Example GIF" />
</div>

Online simulation in global ocean model OM4 ([full movie](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/OM4-Atlantic.mp4), [code](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/OM4-movie.ipynb)):
<div align="center">
  <img src="https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/OM4-gif.gif" alt="Example GIF" />
</div>

## Paper figures
In folder [notebooks](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/notebooks) we show Jupyter notebooks for each Figure plotted in the paper.

## Data
Training data, offline and online skill can be found on [Zenodo](https://doi.org/10.5281/zenodo.15325227).
## Installation
```
git clone --recursive git@github.com:m2lines/ANN-momentum-mesoscale.git
```

## MOM6 online experiments
* The submodule [src/MOM6](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/src) contains the ocean model source code with [implemented ANN parameterization](https://github.com/m2lines/MOM6/blob/89f1fb391d05d3f52549e4f74c74a4b4d6c01960/src/parameterizations/lateral/MOM_Zanna_Bolton.F90#L661). No additional software is required. See [src/README.md](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/README.md) to visualize modifications to MOM6 source code.
* Weights of trained ANN parameterization used in the paper are in [CM26_ML_models/ocean3d/subfilter/FGR3/hidden-layer-20/seed-default/model/Tall.nc](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/CM26_ML_models/ocean3d/subfilter/FGR3/hidden-layer-20/seed-default/model). Additional seeds and ANN with more neurons are provided in [CM26_ML_models/ocean3d/subfilter/FGR3/](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/CM26_ML_models/ocean3d/subfilter/FGR3)
* The files required to run online experiments in idealizes and global ocean configurations are provided in folders [configurations/Neverworld2](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/configurations/NeverWorld2) and [configurations/OM4](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/configurations/OM4), respectively

## Training ANN on global ocean data CM2.6
The ANN parameterization with local dimensional scaling is defined in Python code in [training-on-CM2.6/helpers/state_functions.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/state_functions.py#L1328). Below we describe how to prepare training dataset and execute training algorithm.
### Downloading raw CM2.6 data
Raw data at resolution $1/10^\circ$ subsampled and splitted in time dimension is downloaded from the cloud using [script](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/download_raw_data.py):
```
cd src/training-on-CM2.6/scripts/
python download_raw_data.py
```
Make sure to set [PATH](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/download_raw_data.py#L11) variable.
### Filtering and coarsegraining
Dataset for each coarsegraining factor (`4,9,12,15`) is generated with the [script](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/generate_3d_datasets.py):
```
cd src/training-on-CM2.6/scripts/
python generate_3d_datasets --factor=4
```
Make sure to provide path to `rawdata` in [training-on-CM2.6/helpers/cm26.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/cm26.py#L120) and to coarsened data in 
 [cm26.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/cm26.py#L16) and [script](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/generate_3d_datasets.py#L25).
### Training algorithm and evaluation
[Training loop](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/train_ann.py#L110) is executed on CPUs via the following [script](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/train_script.py):
```
cd src/training-on-CM2.6/scripts/
python train_script.py
```
Make sure to provide [path](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/train_script.py#L38) where to save the trained ANN. This script [contains](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/train_script.py#L19-L32) the default hyperparameters used in the paper. The skill on the testing dataset will be available in `{path_save}/skill-test/factor-{factor}.nc` and log of training/valudation losses in `{path_save}/model/logger.nc`.

## Filtering and offline analysis in idealized configuration NW2
Filtered dataset with diagnosed subfilter fluxes in idealized configuration NW2 is constructed using scripts [filter-NW2-data.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/offline-NW2/filter-NW2-data.py) and [filter-interfaces-GM-filter.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/offline-NW2/filter-interfaces-GM-filter.py). Second script is optional and used only to more accurately estimate APE in outcropping regions. Offline prediction of subfilter fluxes and evaluation of offline skill is present in notebook [offline-analysis-NW2.ipynb](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/notebooks/offline-analysis-NW2.ipynb).
