import sys
sys.path.append('../src/training-on-CM2.6')
sys.path.append('../src/tensor_calculus')
import xarray as xr
import xgcm
from itertools import combinations_with_replacement
import argparse
import json
import copy

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Dataset.dims.*"
)

from helpers.plot_helpers import *
from helpers.selectors import *
from tensor_calculus import Tensor

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=9999)
parser.add_argument('--initial_tensors', type=str, default='[D0, D1, D2]')
parser.add_argument('--their_derivatives', type=str, default='[0, 1, 2]')
parser.add_argument('--max_nonlinearity', type=int, default=2)
parser.add_argument('--max_derivative', type=int, default=4)
parser.add_argument('--advection_nonlinearity', type=int, default=3)
parser.add_argument('--add_perp', type=str, default='False')
parser.add_argument('--add_advection', type=str, default='True')
args = parser.parse_args()
print(args)

# --- Save to JSON ---
args_dict = vars(args)  # convert Namespace to dictionary
with open(f'/scratch/pp2681/mom6/equation-discovery/Pacific_mean_medium/json/args_{args.index}.json', 'w') as f:
    json.dump(args_dict, f, indent=4)

import xrft
def cross_spectrum(SGS, V, dx_mean=1):    
    x = dx_mean*np.arange(len(SGS.xh))
    SGS['xh'] = x
    V['xh'] = x

    sp = xrft.cross_spectrum(SGS, V, dim='xh', window='hann', detrend='linear', true_phase=False)
    
    # Normalize to angular frequencies
    sp['freq_xh'] = sp['freq_xh'] * 2 * np.pi
    sp = sp / (2 * np.pi)
    sp = sp.sel(freq_xh=slice(1e-9,None))
    return np.real(sp).compute()

def metrics(SGS, V):
    SGS_mean = SGS.mean('time').compute()
    V_mean = V.mean('time').compute()

    dEdt = (SGS * V).mean('time').compute()
    dEdt_mean = SGS_mean * V_mean
    dEdt_transient = dEdt - dEdt_mean

    selector = lambda x: x.sel(xh=slice(-215,-150), yh=slice(30,40))

    dx = float(selector(param.dxT).mean())
    
    transfer = cross_spectrum(selector(SGS), selector(V), dx).mean(['time', 'yh']).compute()
    transfer_mean = cross_spectrum(selector(SGS_mean), selector(V_mean), dx).mean(['yh']).compute()
    transfer_transient = transfer - transfer_mean
    
    return SGS_mean, dEdt, dEdt_mean, dEdt_transient, \
           transfer, transfer_mean, transfer_transient

def read_dataset(key='train'):
    base_path = '/scratch/zanna/data/cm2.6-Perezhogin-etal-2025/factor-4'

    selector = lambda x: x.sel(yh=slice(10, 50), xh=slice(-249.8,-130)).sel(yq=slice(10, 50), xq=slice(-249.8,-130)).isel(zl=0)
    
    # Read file with grid information
    depth_selector = lambda x: x.isel(zl=np.arange(0,50,5)) if len(x.zl)==50 else x
    static = selector(depth_selector(xr.open_mfdataset(f'{base_path}/param.nc'))).isel(zi=0)
    
    # Read time-dependent data
    data = selector(xr.open_mfdataset(f'{base_path}/{key}*.nc', chunks={'time':1, 'zl':1}, concat_dim='time', combine='nested').sortby('time'))

    # xgcm grid
    grid = xgcm.Grid(static, coords={
                'X': {'center': 'xh', 'right': 'xq'},
                'Y': {'center': 'yh', 'right': 'yq'}
            },
            boundary={"X": 'fill', 'Y': 'fill'},
            fill_value = {'Y':0, 'X':0})

    return data.astype('float64'), static.astype('float64'), grid

data, param, grid = read_dataset()

eps = Tensor.levi_civita()

D0 = Tensor.init_vector(data.u_h, data.v_h, label="u_i")
D1 = D0.diff(param, grid)
D2 = D1.diff(param, grid)
D3 = D2.diff(param, grid)
#D4 = D3.diff(param, grid)

D2.set_symmetric_indices(['j','k'])
D3.set_symmetric_indices(['j','k', 'm'])
#D4.set_symmetric_indices(['j','k', 'm', 'n'])

S1 = 0.5*(D1 + D1.transpose(['i', 'j']))
S1.label = '%_{ij}'
O1 = D1 - S1
O1.label = '&_{ij}'

S2 = 0.5*(D2 + D2.transpose(['i', 'j']))
S2.label = '@_k%_{ij}'
O2 = D2 - S2
O2.label = '@_k&_{ij}'

def construct_basis_of_tensors(initial_tensors, their_derivatives=None, max_nonlinearity = 1, max_derivative=4, add_perp=False, add_advection=False, advection_nonlinearity=3):
    results = []

    for nonlinearity in range(1, max_nonlinearity+1):
        for idx in combinations_with_replacement(range(0, len(initial_tensors)), nonlinearity):
            tensor = initial_tensors[idx[0]]
            for i in range(1, len(idx)):
                tensor = tensor*initial_tensors[idx[i]]

            sum_derivative = sum([their_derivatives[idx] for idx in idx])
            if sum_derivative > max_derivative:
                continue
            #display(Math(tensor._repr_latex_()+ f'${sum_derivative}$'))

            results.extend(tensor.contract_to_rank_one(add_perp=add_perp))

    final_results = copy.deepcopy(results)

    if add_advection:
        for t in results:
            for tt in results:
                if t.label.count('@') + tt.label.count('@') < max_derivative:
                    if t.label.count('u') + tt.label.count('u') <= advection_nonlinearity:
                        final_results.append((t * tt.diff(param, grid)).contract(['i', 'j']).rename())


    return final_results

basis = construct_basis_of_tensors(eval(args.initial_tensors), 
                                    eval(args.their_derivatives),
                                    max_nonlinearity=args.max_nonlinearity, 
                                    max_derivative=args.max_derivative,
                                    advection_nonlinearity=args.advection_nonlinearity,
                                    add_perp = eval(args.add_perp),
                                    add_advection = eval(args.add_advection))

print('Length of basis', len(basis))

from tensor_calculus import transposition_data
SGS = transposition_data(xr.concat([data.SGSx_h, data.SGSy_h], dim='i'))
V = transposition_data(xr.concat([data.u_h, data.v_h], dim='i'))

# wet_nan = data.wet_nan
# def filter_n(array, n=1):
#     for i in range(n):
#         array = grid.interp(array, ['X', 'Y'])
#     return array

# wet_nan10 = filter_n(wet_nan, 10)

dataset = xr.Dataset()

if args.index == 9999:
    # dataset['wet_nan'] = wet_nan
    # dataset['wet_nan10'] = wet_nan10

    SGS_mean, dEdt, dEdt_mean, dEdt_transient, transfer, transfer_mean, transfer_transient = metrics(SGS,V)

    dataset['SGS_mean'] = SGS_mean
    dataset['dEdt'] = dEdt
    dataset['dEdt_mean'] = dEdt_mean
    dataset['dEdt_transient'] = dEdt_transient
    dataset['transfer'] = transfer
    dataset['transfer_mean'] = transfer_mean
    dataset['transfer_transient'] = transfer_transient
else:
    idx = args.index
    tensor = basis[idx]

    SGS_mean, dEdt, dEdt_mean, dEdt_transient, transfer, transfer_mean, transfer_transient = metrics(tensor.array,V)

    dataset[f'{idx}_SGS_mean'] = SGS_mean
    dataset[f'{idx}_dEdt'] = dEdt
    dataset[f'{idx}_dEdt_mean'] = dEdt_mean
    dataset[f'{idx}_dEdt_transient'] = dEdt_transient
    dataset[f'{idx}_transfer'] = transfer
    dataset[f'{idx}_transfer_mean'] = transfer_mean
    dataset[f'{idx}_transfer_transient'] = transfer_transient


    print(tensor._repr_latex_())
    dataset[f'{idx}_SGS_mean'].attrs["long_name"] = tensor._repr_latex_()
    print('Dataset created')

dataset.astype('float64').to_netcdf(f'/scratch/pp2681/mom6/equation-discovery/Pacific_mean_small/{args.index}.nc')
print('Dataset saved')