We use standard [OM4_SIS2](https://github.com/NOAA-GFDL/MOM6-examples/tree/dev/gfdl/ice_ocean_SIS2/OM4_025) configuration with CORE-II forcing. For forcing data ask Alistair or see [wiki](https://github.com/NOAA-GFDL/MOM6-examples/wiki/Getting-started#downloading-input-data). Installing OM4 configuration on Greene is explained in [computing](https://github.com/Pperezhogin/computing/tree/OM4).

Please make OM4_SIS2 configuration work and then update it with our configuration files.

Please note that:
* Unparameterized and Chang23 experiments can be run with default MOM6 branch: [NOAA-GFDL/MOM6/tree/dev/gfdl](https://github.com/NOAA-GFDL/MOM6/tree/dev/gfdl) 
* ANN experiment must be run with MOM6 branch [Pperezhogin/MOM6/tree/m2lines-mesoscale-ann](https://github.com/Pperezhogin/MOM6/tree/m2lines-mesoscale-ann). ANN weights you will find in [ANN/INPUT](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/configurations/OM4/ANN/INPUT)
