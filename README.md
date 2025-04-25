# ANN parameterization of mesoscale momentum fluxes in ocean model MOM6
This repository contains training algorithm and MOM6 ocean model with implemented parameterization. Additionally, we include figure plotting notebooks for the paper "Generalizable neural-network parameterization of mesoscale eddies in idealized and global ocean models" by Pavel Perezhogin, Laure Zanna and Alistair Adcroft, to be submitted soon.

## MOM6 online experiments
* The MOM6 source code with implemented ANN parameterization can be found in [src/MOM6](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/src).
* The configuration files required to run online experiments are provided for [Neverworld2](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/configurations/NeverWorld2) and global [OM4](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/configurations/OM4) ocean configurations.
* Weights of trained ANN parameterization are in folder [CM26_ML_models](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/CM26_ML_models/ocean3d/subfilter/FGR3)
