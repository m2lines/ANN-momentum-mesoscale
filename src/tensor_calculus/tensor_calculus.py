import xarray as xr
import numpy as np
import xgcm
import itertools
import copy

# Service functions

default_indices_names = ["i", "j", "k", "m", "n", "l", "o", "s", "t", "p", "q", "r", "a", "b"]

def return_dims(array):
    return [idx for idx in array.dims if idx in default_indices_names]

def transposition_data(array):
    '''
    Note transposing does not relabel data
    but simply changes its representation in xarray
    Thus, it is safe to be used everywhere
    without affecting math
    '''
    dims = []
    for dim in ['time', 'zl', 'yh', 'xh']:
        if dim in array.dims:
            dims.append(dim)
    dims.append(...)
    for dim in default_indices_names:
        if dim in array.dims:
            dims.append(dim)
    return array.transpose(*dims)

def transpose_tensor(array, pair):
    return array.copy().swap_dims({pair[0]:pair[1], pair[1]:pair[0]})

def symmetrize_tensor(_array, sym_axes):
    array = _array * 0
    permutations = list(itertools.permutations(sym_axes))
    for ax in permutations:
        d = {key:val for key,val in zip(sym_axes, ax)}
        array += _array.copy().swap_dims(d)
    return array / len(list(permutations))

def ddx(array):
    '''
    d/dx operator applied in center points
    '''
    out = param.wet * grid.interp(param.wet_u * grid.diff(param.wet * array,'X') / param.dxCu, 'X')
    return transposition_data(out)

def ddy(array):
    '''
    d/dy operator applied in center points
    '''
    out = param.wet * grid.interp(param.wet_v * grid.diff(param.wet * array,'Y') / param.dyCv, 'Y')
    return transposition_data(out)

class Tensor():
    def __init__(self, array, label='', hash_array=None):
        '''
        array is xarray
        label is the Latex code describing the 
        tensor. letters describing default indices
        should not be used in other sense
        all tensor indices present in array must
        be specified in label as well

        hash_array is used to perform cheap calculations
        and determine tensors which are similar
        '''
        self.array = transposition_data(array)
        self.label = label
        if hash_array is None:
            dims = return_dims(self.array)
            # We do not assume any symmetry here
            self.hash_array = xr.DataArray(np.random.randn(*[2]*len(dims)), dims=dims)
        else:
            self.hash_array = transposition_data(hash_array)

    # Service functions
    def _repr_latex_(self):
        '''
        Print tensor latex equation
        '''
        if self.label:
            return f"${self.label}$".replace("@", "\\partial")
        else:
            return f"$T({', '.join(self.dims())})$"

    def rename(self):
        '''
        Rename indices to the default set.
        Useful after contraction.
        Modifies both array and label
        '''
        label = self.label
        array = self.array.copy()
        hash_array = self.hash_array.copy()
        for idx_old, idx_new in zip(self.dims(), default_indices_names):
            array = array.rename({idx_old:idx_new})
            hash_array = hash_array.rename({idx_old:idx_new})
            label = label.replace(idx_old, idx_new)
        return Tensor(array, label, hash_array)

    def dims(self):
        '''
        Returns list of tensor indices
        '''
        idx_array = return_dims(self.array)
        idx_hash_array = return_dims(self.hash_array)
        if set(idx_array) != set(idx_hash_array):
            raise ValueError(f"Mismatch between array dims and hash_array dims: {idx_array} vs {idx_hash_array}")
        
        return idx_array

    def copy(self):
        '''
        Copy tensor object and associated data
        '''
        return Tensor(self.array.copy(), copy.deepcopy(self.label), self.hash_array.copy())
    
    def transpose(self, pair):
        '''
        Swap two indices (e.g. transpose over i and j).
        pair: tuple of two indices (e.g. ("i", "j"))
        '''
        # Swap dims in the underlying xarray
        array = transpose_tensor(self.array, pair)
        hash_array = transpose_tensor(self.hash_array, pair)

        # Update LaTeX label: swap pair[0] and pair[1]
        tmp_symbol = "__TMP__"
        label = copy.deepcopy(self.label)
        new_label = (
            label
            .replace(pair[0], tmp_symbol)
            .replace(pair[1], pair[0])
            .replace(tmp_symbol, pair[1])
        )

        return Tensor(array, label, hash_array)
    
    def contract_to_rank_one(self):
        """
        Non-recursive contraction to rank-1 tensors.
        Only keep unique results (by hash_array / 1D array comparison).
        Returns a list of Tensor objects of rank 1.
        """
        n = len(self.dims())
        
        if n == 1:
            return [self]

        if n%2 == 0:
            return []
        
        # Number of pairs to contract
        n_pairs = (n - 1) // 2
        
        # Free indices to contract
        free_indices = self.dims()
        
        # Generate all unordered sets of indices to contract
        # Each combination is a list of pairs of indices
        all_pairs = list(itertools.combinations(free_indices, 2))

        # Generate all possible sets of indices without repetition
        # Repetition is enforces by the fact that all indices are different
        sets_of_pairs = []
        for set_of_pairs in list(itertools.combinations(all_pairs, n_pairs)):
            '''
            Keep only pairs which have all different indices
            '''
            all_indices = set([idx for pair in set_of_pairs for idx in pair])
            if len(all_indices) == n_pairs * 2:
                sets_of_pairs.append(set_of_pairs)
        
        # Keep only those sets of pairs which return unique hash
        unique_hashes = []
        results = []
        for pair_set in sets_of_pairs:
            # Perform contractions
            hash_array = self.hash_array.copy()
            for pair in pair_set:
                hash_array = sum(hash_array.isel({pair[0]: i, pair[1]: i}) for i in range(2)).compute()

            # Convert to 1D numpy array
            hash_array = np.array(hash_array)
            
            # Check uniqueness
            if not any(np.allclose(hash_array, h) for h in unique_hashes):
                unique_hashes.append(hash_array)
                tensor = self
                for pair in pair_set:
                    tensor = tensor.contract(pair)
                results.append(tensor)
        
        return results

    def contract(self, pair):
        '''
        Contract tensor over a pair of indices
        By convention, contracted indices are big in the label
        and thus they are not anymore free dimensions of the tensor
        '''

        # Contract array
        array = sum(self.array.isel({pair[0]: i, pair[1]: i}) for i in range(2))
        # Contract hash_array
        hash_array = sum(self.hash_array.isel({pair[0]: i, pair[1]: i}) for i in range(2))
        
        # pick a single uppercase letter from the pair[0]
        contracted = pair[0].upper()
        # replace both indices in the label with the uppercase one
        label = self.label.replace(pair[0], contracted).replace(pair[1], contracted)

        return Tensor(array, label, hash_array)

    def __add__(self, tensor2):
        return Tensor(self.array+tensor2.array, self.label+'+'+tensor2.label, self.hash_array+tensor2.hash_array)

    def __sub__(self, tensor2):
        return Tensor(self.array-tensor2.array, self.label+'-'+tensor2.label, self.hash_array-tensor2.hash_array)

    def __mul__(self, tensor2):
        '''
        Multiplies two tensors as outer product,
        i.e. without repeating indices
        Main code handles that indices are indeed not reepeting
        '''
        # Indices in the left tensor
        idx_set1 = self.dims()
        # Indices in the right tensor
        idx_set2 = tensor2.dims()

        if len(idx_set1) + len(idx_set2) > len(default_indices_names):
            raise ValueError(f"So many dimensions is not supported")

        # Identify intices in the second tensor to be renamed
        # as they are identical to those in the left tensor
        idx_rename = [idx for idx in idx_set2 if idx in idx_set1]

        # Empty set of indices not yet used in these tensor
        idx_empty = [idx for idx in default_indices_names if idx not in idx_set1 and idx not in idx_set2]
        
        # Rename indices
        array2 = tensor2.array.copy()
        hash_array2 = tensor2.hash_array.copy()
        label2 = copy.deepcopy(tensor2.label)

        for i,j in zip(idx_rename, idx_empty):
            array2 = array2.rename({i:j})
            hash_array2 = hash_array2.rename({i:j})
            label2 = label2.replace(i, j)

        array = self.array * array2
        hash_array = self.hash_array * hash_array2
        label = f'({self.label})({label2})'
            
        return Tensor(array, label, hash_array)

    def diff(self):
        '''
        Differentiate current tensor by applying \partial operator
        and increasing tensor rank by one
        '''
        idx_empty = [idx for idx in default_indices_names if idx not in self.dims()]
        new_index = idx_empty[0]

        # @ is a special symbol representing \\partial
        label = f'@_{new_index}' + self.label
        return Tensor(xr.concat([ddx(self.array), ddy(self.array)], dim=new_index), label=label)

    @classmethod
    def init_vector(cls, u, v, label="u_i"):
        """
        Construct a rank-1 vector Tensor from two components u, v.
        Concatenates along a new index 'i'.
        """
        arr = xr.concat([u, v], dim='i')
        return cls(arr, label=label)

    def set_symmetric_indices(self, dims):
        '''
        This function modifies hash_array to be symmetric over the
        set of indices. This does not apply to the array itself
        as we assume that it is already symmetric in these indices

        As this is simple operation updating hash_array, we do it in-place
        '''
        self.hash_array = symmetrize_tensor(self.hash_array, dims)