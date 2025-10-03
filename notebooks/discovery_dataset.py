import sys
sys.path.append('../src/training-on-CM2.6')
sys.path.append('../src/tensor_calculus')
import xarray as xr
import xgcm
from itertools import combinations_with_replacement

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Dataset.dims.*"
)

from helpers.plot_helpers import *
from helpers.selectors import *
from tensor_calculus import Tensor

def read_dataset(key='train'):
    base_path = '/scratch/zanna/data/cm2.6-Perezhogin-etal-2025/factor-4'
    
    # Read file with grid information
    depth_selector = lambda x: x.isel(zl=np.arange(0,50,5)) if len(x.zl)==50 else x
    static = depth_selector(xr.open_mfdataset(f'{base_path}/param.nc', chunks={'zl':1}))
    
    # Read permanent features
    permanent_features = xr.open_mfdataset(f'{base_path}/permanent_features.nc', chunks={'zl':1})
    
    # Read time-dependent data
    data = xr.open_mfdataset(f'{base_path}/{key}*.nc', chunks={'time':1, 'zl':1}, concat_dim='time', combine='nested').sortby('time')

    # Merge permanent and time-depending datasets
    data = xr.merge([data, permanent_features])

    # xgcm grid
    grid = xgcm.Grid(static, coords={
                'X': {'center': 'xh', 'right': 'xq'},
                'Y': {'center': 'yh', 'right': 'yq'}
            },
            boundary={"X": 'periodic', 'Y': 'fill'},
            fill_value = {'Y':0})

    return data.astype('float64'), static.astype('float64'), grid

data, param, grid = read_dataset()

derivatives = {}
derivatives['D0'] = Tensor.init_vector(data.u_h, data.v_h, label="u_i")
derivatives['D1'] = derivatives['D0'].diff(param, grid)
derivatives['D2'] = derivatives['D1'].diff(param, grid)
derivatives['D3'] = derivatives['D2'].diff(param, grid)
derivatives['D4'] = derivatives['D3'].diff(param, grid)

derivatives['D2'].set_symmetric_indices(['j','k'])
derivatives['D3'].set_symmetric_indices(['j','k', 'm'])
derivatives['D4'].set_symmetric_indices(['j','k', 'm', 'n'])

def construct_basis_of_tensors(derivatives, max_derivative=2, max_nonlinearity = 1,
                               max_total_derivative=6):
    results = []

    if max_nonlinearity >= 1:
        for d1 in range(0, max_derivative+1):
            if (d1 > max_total_derivative):
                continue
            high_rank_tensor = derivatives[f'D{d1}']
            results.extend(high_rank_tensor.contract_to_rank_one())

    if max_nonlinearity >= 2:
        for d1, d2 in combinations_with_replacement(range(0, max_derivative+1), 2):
            if (d1 + d2 > max_total_derivative):
                continue
            high_rank_tensor = derivatives[f'D{d1}']*derivatives[f'D{d2}']
            results.extend(high_rank_tensor.contract_to_rank_one())

    if max_nonlinearity >= 3:
        for d1, d2, d3 in combinations_with_replacement(range(0, max_derivative+1), 3):
            if (d1 + d2 + d3> max_total_derivative):
                continue
            high_rank_tensor = derivatives[f'D{d1}']*derivatives[f'D{d2}']*derivatives[f'D{d3}']
            results.extend(high_rank_tensor.contract_to_rank_one())

    if max_nonlinearity >= 4:
        for d1, d2, d3, d4 in combinations_with_replacement(range(0, max_derivative+1), 4):
            if (d1 + d2 + d3 + d4> max_total_derivative):
                continue
            high_rank_tensor = derivatives[f'D{d1}']*derivatives[f'D{d2}']*derivatives[f'D{d3}']*derivatives[f'D{d4}']
            results.extend(high_rank_tensor.contract_to_rank_one())
    return results

basis = construct_basis_of_tensors(derivatives, max_derivative=2, max_nonlinearity=3, max_total_derivative=6)

print('Length of basis', len(basis))

from tensor_calculus import transposition_data
SGS = transposition_data(xr.concat([data.SGSx_h, data.SGSy_h], dim='i'))
selector = lambda x: x.isel(zl=0).sel(yh=slice(10, 50), xh=slice(-250,-130)).isel(time=[i for i in range(20)] + [i for i in range(-20,0)])

wet_nan = data.wet_nan
def filter_n(array, n=1):
    for i in range(n):
        array = grid.interp(array, ['X', 'Y'])
    return array

wet_nan10 = filter_n(wet_nan, 10)

dataset = xr.Dataset()
dataset['wet_nan'] = (wet_nan).compute()
dataset['wet_nan10'] = (wet_nan10).compute()
dataset['SGS'] = selector(SGS).compute()
dataset['SGS'].attrs["long_name"] = r"$\mathcal{S}$"

V = transposition_data(xr.concat([data.u_h, data.v_h], dim='i'))
dataset['V'] = selector(V).compute()
dataset['V'].attrs["long_name"] = '$u_i$'
dataset['dx'] = data.delta_x

for j, tensor in enumerate(basis):
    dataset[f'{j}'] = selector(tensor.array).compute()
    dataset[f'{j}'].attrs["long_name"] = tensor._repr_latex_()
    print(j)

dataset.astype('float64').to_netcdf('/scratch/pp2681/mom6/equation-discovery/Pacific_D2_DD6_NN3.nc')