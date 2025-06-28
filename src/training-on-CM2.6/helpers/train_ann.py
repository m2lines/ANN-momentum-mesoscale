import sys
import numpy as np
import xarray as xr
from helpers.cm26 import read_datasets
from helpers.ann_tools import ANN, tensor_from_xarray
import torch
import torch.optim as optim
import itertools

import os
from time import time

def get_subfilter_forcing(batch):
    SGSx = batch.data.SGSx
    SGSy = batch.data.SGSy
    
    SGSx = tensor_from_xarray(SGSx)
    SGSy = tensor_from_xarray(SGSy)

    SGS_norm = 1. / torch.sqrt((SGSx**2 + SGSy**2).mean())
    SGSx = SGSx * SGS_norm
    SGSy = SGSy * SGS_norm

    return SGSx, SGSy, SGS_norm

def get_subfilter_fluxes(batch):
    Txx = tensor_from_xarray(batch.data.Txx)
    Tyy = tensor_from_xarray(batch.data.Tyy)
    Txy = tensor_from_xarray(batch.data.Txy)
    
    T_norm = 1. / torch.sqrt((Txx**2 + Tyy**2 + Txy**2).mean())
    Txx = Txx * T_norm
    Tyy = Tyy * T_norm
    Txy = Txy * T_norm

    return Txx, Tyy, Txy, T_norm

def train_ANN(factors=[9],
              stencil_size = 3,
              hidden_layers=[32,32],
              dimensional_scaling=True, 
              symmetries='All',
              time_iters=50,
              learning_rate = 1e-3,
              depth_idx=np.arange(1),
              print_iters=1,
              feature_functions=[],
              gradient_features=['sh_xy', 'sh_xx', 'rel_vort'],
              permute_factors_and_depth=True,
              subfilter='subfilter',
              FGR=3, loss_function='forcing'):
    '''
    time_iters is the number of time snaphots
    randomly sampled for each factor and depth

    depth_idx is the indices of the vertical layers which
    participate in training process
    '''
    ########### Read dataset ############
    dataset = read_datasets(['train', 'validate'], factors, subfilter=subfilter, FGR=FGR)

    ########## Init logger ###########
    logger = xr.Dataset()
    for key in ['MSE_train', 'MSE_validate']:
        logger[key] = xr.DataArray(np.zeros([time_iters, len(factors), len(depth_idx)]), 
                                   dims=['iter', 'factor', 'depth'], 
                                   coords={'factor': factors, 'depth': depth_idx})

    ########## Init ANN ##############
    # As default we have 3 input features on a stencil: D, D_hat and vorticity
    num_input_features = stencil_size**2 * len(gradient_features) + len(feature_functions)
    ann_Tall = ANN([num_input_features, *hidden_layers, 3])
    
    ########## Symmetries as data augmentation ######
    def augment():
        if symmetries == 'False':
            # Values for rotation, reflect_x and reflect_y
            return zip([0],[False],[False])
        elif symmetries == 'True':
            rots  = [90, 0]
            refxs = [True, False]
            refys = [True, False]
            return zip([rots[np.random.binomial(1,0.5)]], 
                        [refxs[np.random.binomial(1,0.5)]], 
                        [refys[np.random.binomial(1,0.5)]])
        elif symmetries == 'All':
            rots  = [90, 0]
            refxs = [True, False]
            refys = [True, False]
            return itertools.product(rots, refxs, refys)
        elif symmetries == 'None':
            rots  = [0, 0]
            refxs = [False, False]
            refys = [False, False]
            return itertools.product(rots, refxs, refys)
        else:
            print('Wrong symmetries parameter:', symmetries)

    ########## Random sampling of depth and factors #######
    def iterator(x,y):
        # Product of two 1D iterators
        x_prod = np.repeat(x,len(y))
        y_prod = np.tile(y,len(x))
        xy_prod = np.vstack([x_prod,y_prod]).T
        if permute_factors_and_depth:
            # Randomly permuting iterator along common dimension
            return np.random.permutation(xy_prod)
        else:
            # This is equivalent to
            # for xx in x:
            #    for yy in y:
            #       ....
            return xy_prod
    
    ############ Init optimizer ##############
    all_parameters = ann_Tall.parameters()
    optimizer = optim.Adam(all_parameters, lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=[int(time_iters/2), int(time_iters*3/4), int(time_iters*7/8)], gamma=0.1)

    t_s = time()
    for time_iter in range(time_iters):
        t_e = time()

        for factor, depth in iterator(factors, depth_idx):
            # Note here we randomly sample time moment 
            # for every combination of factor and depth
            # So, consequetive snapshots are not correlated (on average)
            # Batch is a dataset consisting of one 2D slice of data
            batch = dataset[f'train-{factor}'].select2d(zl=depth)

            ############## Training step ###############
            if loss_function == 'forcing':
                SGSx, SGSy, SGS_norm = get_subfilter_forcing(batch)
            elif loss_function == 'fluxes':
                Txx, Tyy, Txy, T_norm = get_subfilter_fluxes(batch)
            else:
                print("Error: wrong value of loss_function")

            ######## Optionally, apply symmetries by data augmentation #########
            for rotation, reflect_x, reflect_y in augment():
                optimizer.zero_grad()

                prediction = batch.state.Apply_ANN(None, None, ann_Tall,
                    stencil_size=stencil_size, dimensional_scaling=dimensional_scaling,
                    feature_functions=feature_functions, gradient_features=gradient_features,
                    rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y)

                if loss_function == 'forcing':
                    ANNx = prediction['ZB20u'] * SGS_norm
                    ANNy = prediction['ZB20v'] * SGS_norm

                    MSE_train = ((ANNx-SGSx)**2 + (ANNy-SGSy)**2).mean()
                elif loss_function == 'fluxes':
                    ANNxx = prediction['Txx'] * T_norm
                    ANNyy = prediction['Tyy'] * T_norm
                    ANNxy = prediction['Txy'] * T_norm

                    MSE_train = (
                        (ANNxx - Txx)**2 + 
                        (ANNyy - Tyy)**2 + 
                        (ANNxy - Txy)**2 ).mean()
                else:
                    print("Error: wrong value of loss_function")

                MSE_train.backward()
                optimizer.step()

            del batch

            ############ Validation step ##################
            batch = dataset[f'validate-{factor}'].select2d(zl=depth)
            
            if loss_function == 'forcing':
                SGSx, SGSy, SGS_norm = get_subfilter_forcing(batch)
            elif loss_function == 'fluxes':
                Txx, Tyy, Txy, T_norm = get_subfilter_fluxes(batch)
            else:
                print("Error: wrong value of loss_function")

            prediction = batch.state.Apply_ANN(None, None, ann_Tall,
                    stencil_size=stencil_size, dimensional_scaling=dimensional_scaling,
                    feature_functions=feature_functions, gradient_features=gradient_features,
                    rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y)

            if loss_function == 'forcing':
                ANNx = prediction['ZB20u'] * SGS_norm
                ANNy = prediction['ZB20v'] * SGS_norm

                MSE_validate = ((ANNx-SGSx)**2 + (ANNy-SGSy)**2).mean()
            elif loss_function == 'fluxes':
                ANNxx = prediction['Txx'] * T_norm
                ANNyy = prediction['Tyy'] * T_norm
                ANNxy = prediction['Txy'] * T_norm

                MSE_validate = (
                    (ANNxx - Txx)**2 + 
                    (ANNyy - Tyy)**2 + 
                    (ANNxy - Txy)**2 ).mean()
            else:
                    print("Error: wrong value of loss_function")
                
            del batch
        
            ########### Logging ############
            MSE_train = float(MSE_train.data)
            MSE_validate = float(MSE_validate.data)

            for key in ['MSE_train', 'MSE_validate']:
                logger[key].loc[{'iter': time_iter, 'factor': factor, 'depth': depth}] = eval(key)
            if (time_iter+1) % print_iters == 0:
                print(f'Factor: {factor}, depth: {depth}, '+'MSE train/validate: [%.6f, %.6f]' % (MSE_train, MSE_validate))
        t = time()
        if (time_iter+1) % print_iters == 0:
            print(f'Iter/num_iters [{time_iter+1}/{time_iters}]. Iter time/Remaining time in seconds: [%.2f/%.1f]' % (t-t_e, (t-t_s)*(time_iters/(time_iter+1)-1)))
        scheduler.step()

    for factor in factors:
        for train_str in ['train', 'validate']:
            del dataset[f'{train_str}-{factor}']
    
    return ann_Tall, logger