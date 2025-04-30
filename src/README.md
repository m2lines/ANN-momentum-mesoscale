Changes introduced to MOM6 source code in order to implement ANN parameterization are stored in [mom6_modifications.patch](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/mom6_modifications.patch)

To visualize changes, do:
```
cd MOM6
git checkout e63a8220e
git apply ../mom6_modifications.patch
git status
```
