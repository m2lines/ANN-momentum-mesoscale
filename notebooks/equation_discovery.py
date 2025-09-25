import xarray as xr
import numpy as np
from pysr import PySRRegressor
from sympy import simplify

ds = xr.open_dataset('dataset.nc')

# Here, we know the grid spacing scaling we already applied
#ds['target'] = - ds['Ttr'] / ds['delta_x']**2
y_units="m/s^2"
#ds['target'] = (ds['sh_xx']**2 + ds['sh_xy']**2 + ds['rel_vort']**2) / 3.0

# Constructing scaling functions
strain_scale = np.sqrt((
               (ds['mask'] * ds['sh_xx']**2).mean(['xh', 'yh']) + \
               (ds['mask'] * ds['sh_xy']**2).mean(['xh', 'yh']) + \
               (ds['mask'] * ds['rel_vort']**2).mean(['xh', 'yh'])
                        ) / 3.0)

length_scale = float(ds['delta_x'].mean())

centers=['sh_xx', 'sh_xy', 'rel_vort']
sides = []
for key in centers:
    for side in ['_e', '_w', '_n', '_s']:
        sides.append(f'{key}{side}')
corners = []
for key in centers:
    for corner in ['_sw', '_se', '_nw', '_ne']:
        corners.append(f'{key}{corner}')

# We removed cross derivatives as they correlate too much
high_order_derivatives = ['d2udx2', 'd2udy2', 'd2vdx2', 'd2vdy2']
# Basis funcion given by gradient models of different orders
gradient_basis = ['VGM2tr', 'VGM4tr', 'VGM6tr']

def sqscale(x):
    return np.sqrt((x**2 * ds['mask']).mean(['xh', 'yh'])).values

def linscale(x):
    return (x * ds['mask']).mean(['xh', 'yh']).values

# Rescaling the input features
for key in centers + sides + corners + high_order_derivatives:
    ds[key] = ds[key] / strain_scale
    print(key, sqscale(ds[key]).mean())

# Rescaling the output features
for key in gradient_basis:
    ds[key] = ds[key] / strain_scale**2
    print(key, linscale(ds[key]).mean())

ds['SGSx'] = ds['SGSx'] / strain_scale**2 / length_scale
ds['delta_x'] = ds['delta_x'] / length_scale
ds['u'] = ds['u'] / strain_scale / length_scale
ds['v'] = ds['v'] / strain_scale / length_scale

for key in ['SGSx', 'delta_x', 'u', 'v']:
    print(key, sqscale(ds[key]).mean())

s = ds['VGM2tr'] + ds['VGM4tr'] + ds['VGM6tr']
ds['scaling2'] = ds['VGM2tr'] / s
ds['scaling4'] = ds['VGM4tr'] / s
ds['scaling6'] = ds['VGM6tr'] / s

base_features = ['u', 'v', 'dudx', 'dvdx', 'dudy', 'dvdy']
X_units = ["m/s"]*2 +  ["1/s"]*4

mask1d = ds['mask'].values.reshape(-1)
valid  = ~np.isnan(mask1d)  # True where mask is not NaN
def flatten(array):
    array1d = array.values.reshape(-1)
    return array1d[valid]

x = np.array([flatten(ds[feature])
                for feature in base_features]).T
y = flatten(ds['SGSx'])

print('Output squared scale', np.mean(y**2))
print('Output linear scale', np.mean(y))

model = PySRRegressor(
    maxsize=30,
    niterations=300,  # < Increase me for better results
    binary_operators=["*", "+"],
    dimensionless_constants_only = True,
    dimensional_constraint_penalty = 10**5,
)

import numpy as np
idx = np.random.choice(x.shape[0], 10000)

model.fit(x[idx],y[idx], X_units=X_units, y_units=y_units)

hof = model.get_hof()
for i, row in hof.iterrows():
    expr = row["sympy_format"]
    simp = simplify(expr)
    print(f"Complexity {row['complexity']}: {simp}")