Changes introduced to MOM6 source code in order to implement ANN parameterization are stored in [mom6_modifications.patch](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/mom6_modifications.patch) and can be visualised as follows (this feature exists to support long-term storage of source code on Zenodo):
```
git clone --recursive git@github.com:NOAA-GFDL/MOM6.git MOM6-test
cd MOM6-test
git checkout e63a8220e
git apply ../mom6_modifications.patch
git status
```

Submodule `analysis-of-global-ocean` is additionally stored on [Zenodo](https://zenodo.org/records/15325780).
