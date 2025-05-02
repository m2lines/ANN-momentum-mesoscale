We use standard [OM4_SIS2](https://github.com/NOAA-GFDL/MOM6-examples/tree/dev/gfdl/ice_ocean_SIS2/OM4_025) configuration with CORE-II forcing. For forcing data ask Alistair or see [wiki](https://github.com/NOAA-GFDL/MOM6-examples/wiki/Getting-started#downloading-input-data). Installing OM4 configuration on Greene is explained in [computing](https://github.com/Pperezhogin/computing/tree/OM4).

Please make OM4_SIS2 configuration work and then update it with our configuration files.

Please note that:
* Unparameterized and Chang23 experiments can be run with default MOM6 branch: [NOAA-GFDL/MOM6/tree/dev/gfdl](https://github.com/NOAA-GFDL/MOM6/tree/dev/gfdl) 
* ANN experiment must be run with MOM6 branch [dev/m2lines](https://github.com/m2lines/MOM6/tree/89f1fb391d05d3f52549e4f74c74a4b4d6c01960). ANN weights are stored in this repository [see CM26_ML_models](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/CM26_ML_models/ocean3d/subfilter/FGR3)
