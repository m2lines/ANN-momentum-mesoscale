# ANN parameterization of mesoscale momentum fluxes in ocean model MOM6
This repository contains training algorithm and MOM6 ocean model with implemented parameterization. Additionally, we include figure plotting notebooks for the paper "Generalizable neural-network parameterization of mesoscale eddies in idealized and global ocean models" by Pavel Perezhogin, Laure Zanna and Alistair Adcroft, to be submitted soon.

## MOM6 online experiments
* The MOM6 source code with implemented ANN parameterization can be found in [src/MOM6](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/src)
* Weights of trained ANN parameterization used in the paper are in [CM26_ML_models/ocean3d/subfilter/FGR3/hidden-layer-20/seed-default/model/Tall.nc](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/CM26_ML_models/ocean3d/subfilter/FGR3/hidden-layer-20/seed-default/model). Additional seeds and ANN with more neurons are provided in [CM26_ML_models/ocean3d/subfilter/FGR3/](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/CM26_ML_models/ocean3d/subfilter/FGR3)
* The files required to run online experiments in idealizes and global ocean configurations are provided in folders [configurations/Neverworld2](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/configurations/NeverWorld2) and [configurations/OM4](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/configurations/OM4), respectively

## Training ANN on global ocean data CM2.6
### Downloading raw CM2.6 data

### Filtering/coarsegraining and computing subfilter forcing

### Training algorithm
[Training loop](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/train_ann.py#L110) is executed on CPUs via the following [script](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/train_script.py):
```
cd src/training-on-CM2.6/scripts/
python train_script.py
```
This script [contains](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/train_script.py#L19-L32) the default hyperparameters used in the paper.
