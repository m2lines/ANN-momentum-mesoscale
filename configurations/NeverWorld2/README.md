Please follow the link for configuration files of the [NeverWorld2](https://github.com/ocean-eddy-cpt/NeverWorld2/tree/main/simulations/baselines/nw2_0.25deg_N15_baseline_hmix5) and update them with our MOM_override, input.nml and diag_table files.

Please note that:
* Unparameterized and ZB20 experiments can be run with default MOM6 branch: [NOAA-GFDL/MOM6/tree/dev/gfdl](https://github.com/NOAA-GFDL/MOM6/tree/dev/gfdl) 
* Yankovsky24 parameterization should be run with different MOM6 branch: [ElizabethYankovsky/MOM6/tree/EBT_testing](https://github.com/ElizabethYankovsky/MOM6/tree/EBT_testing)
* ANN experiment must be run with MOM6 branch [dev/m2lines](https://github.com/m2lines/MOM6/tree/89f1fb391d05d3f52549e4f74c74a4b4d6c01960). ANN weights are stored in this repository [see CM26_ML_models](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/CM26_ML_models/ocean3d/subfilter/FGR3)

Raw simulation data can be found on [Zenodo](https://zenodo.org/records/15328410)
