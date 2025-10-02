import xarray as xr
import xgcm
import itertools
import copy

# Service functions

default_indices_names = ["i", "j", "k", "m", "n", "l", "o", "s", "t", "p", "q", "r", "a", "b"]

def transpose_data(array):
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

def ddx(array):
    '''
    d/dx operator applied in center points
    '''
    out = param.wet * grid.interp(param.wet_u * grid.diff(param.wet * array,'X') / param.dxCu, 'X')
    return transpose_data(out)

def ddy(array):
    '''
    d/dy operator applied in center points
    '''
    out = param.wet * grid.interp(param.wet_v * grid.diff(param.wet * array,'Y') / param.dyCv, 'Y')
    return transpose_data(out)

def pair_is_equal(pair1, pair2, symmetric_indices_sets):
    '''
    Service two function which decides if
    contraction of pair1 of indices and over pair2
    is identical given the symmetry of the tensor
    symmetric_indices_sets = list(list())
    '''
    if set(pair1) == set(pair2):
        return True

    # This operator returns a full list of entries which
    # are different in two sets. In a case of symmetric 
    # tensors these should be two elements and
    # both should lie in symmetric_indices_set
    different_indices = list(set(pair1) ^ set(pair2))
    
    # We check each set indipendently
    # if different indices correspond to any of sets
    # then the pairs are equal
    for symmetric_indices in symmetric_indices_sets:
        idx_in_symmetric_set = [False] * len(different_indices)
        for j, idx in enumerate(different_indices):
            if idx in symmetric_indices:
                idx_in_symmetric_set[j] = True
        if all(idx_in_symmetric_set):
            return True
    return False

class Tensor():
    def __init__(self, array, label='', symmetric_indices_sets=[]):
        '''
        array is xarray
        label is the Latex code describing the 
        tensor. letters describing default indices
        should not be used in other sense
        all tensor indices present in array must
        be specified in label as well
        '''
        self.array = transpose_data(array)
        self.verbose = False
        self.label = label
        # All tuples of indices which show symmetry of the tensor
        # for example d3 u / dx_i dx_j dx_k is symmetric in indices
        # i,j,k and thus should pass a tuple ('i', 'j', 'k')
        # Temporarily, this set is should be set directly
        self.symmetric_indices_sets = symmetric_indices_sets
    
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
        for idx_old, idx_new in zip(self.dims(), default_indices_names):
            self.array = self.array.rename({idx_old:idx_new})
            label = label.replace(idx_old, idx_new)
        return Tensor(self.array, label)

    def dims(self):
        '''
        Returns list of tensor indices
        '''
        return [idx for idx in self.array.dims if idx in default_indices_names]

    def copy(self):
        '''
        Copy tensor object and associated data
        '''
        return Tensor(self.array.copy(), self.label, copy.deepcopy(self.symmetric_indices_sets))
    
    def transpose(self, pair):
        '''
        Swap two indices (e.g. transpose over i and j).
        pair: tuple of two indices (e.g. ("i", "j"))
        '''
        # Swap dims in the underlying xarray
        arr_new = self.array.rename({pair[0]: "tmp"}).rename({pair[1]: pair[0]}).rename({"tmp": pair[1]})

        # Update LaTeX label: swap pair[0] and pair[1]
        tmp_symbol = "__TMP__"
        new_label = (
            self.label
            .replace(pair[0], tmp_symbol)
            .replace(pair[1], pair[0])
            .replace(tmp_symbol, pair[1])
        )

        return Tensor(arr_new, new_label)
    
    def contract_to_rank(self, rank=1):
        """
        Recursively contract tensor indices until it has the desired rank.
        Returns a list of Tensor objects of the target rank.
        """
        n = len(self.dims())

        # Base case: impossible
        if n < rank or (n - rank) % 2 != 0:
            return []

        if n == rank:
            return [self]

        # Base case: one contraction left
        if n - rank == 2:
            return self.list_of_all_contractions()

        # Recursive case
        results = []
        for contracted in self.list_of_all_contractions():
            results.extend(contracted.contract_to_rank(rank=rank))
        
        return results

    def list_of_all_contractions(self):
        '''
        Takes the current tensor and if its
        rank is higher or equal than 2,
        finds all possible contractions over
        two indices filtering out identical
        ones given by symmetry of the tensor
        '''

        if len(self.dims()) < 2:
            return []

        # All unordered combinations
        pairs = [pair for pair in itertools.combinations(self.dims(),2)]

        # Keep only pairs which remain after applying symmetry conditions
        pairs_filtered = []

        for pair1 in pairs:
            append = True
            for pair2 in pairs_filtered:
                if pair_is_equal(pair1, pair2, self.symmetric_indices_sets):
                    append=False
            
            if append:
                pairs_filtered.append(pair1)

        # Compute contractions
        output = []
        for pair in pairs_filtered:
            output.append(self.contract(pair))

        return output

    def contract(self, *pairs):
        out = self
        for pair in pairs:
            out = out._contract(pair)
        return out
        
    def _contract(self, pair):
        '''
        Contract tensor over a pair of indices
        By convention, contracted indices are big in the label
        and thus they are not anymore free dimensions of the tensor
        '''
        out = 0
        for i in range(2):
            out += self.array.isel({pair[0]:i, pair[1]:i})
        # pick a single uppercase letter from the pair[0]
        contracted = pair[0].upper()
        # replace both indices in the label with the uppercase one
        new_label = self.label.replace(pair[0], contracted).replace(pair[1], contracted)

        # Remove contracted indices from the list of symmetric indices
        new_symmetric_indices_sets = [[s for s in sub if s not in pair] for sub in self.symmetric_indices_sets]
        return Tensor(transpose_data(out), new_label, new_symmetric_indices_sets)

    def __add__(self, _tensor2):
        return Tensor(self.array+_tensor2.array, self.label+'+'+_tensor2.label)

    def __sub__(self, _tensor2):
        return Tensor(self.array-_tensor2.array, self.label+'-'+_tensor2.label)

    def __mul__(self, _tensor2):
        '''
        Multiplies two tensors as outer product,
        i.e. without repeating indices
        Main code handles that indices are indeed not reepeting
        '''
        tensor2 = _tensor2.copy()

        idx_set1 = self.dims()
        idx_set2 = tensor2.dims()

        # We rename indices of the second tensor
        idx_rename = [idx for idx in idx_set2 if idx in idx_set1]

        # Empty set of indices not yet used in these tensor
        idx_empty = [idx for idx in default_indices_names if idx not in idx_set1 and idx not in idx_set2]
        
        label2 = tensor2.label
        symmetric_indices_sets2 = tensor2.symmetric_indices_sets
        for i,j in zip(idx_rename, idx_empty):
            tensor2.array = tensor2.array.rename({i:j})
            label2 = label2.replace(i, j)
            symmetric_indices_sets2 = [[s.replace(i, j) for s in sub] for sub in symmetric_indices_sets2]
        
        label1 = f'({self.label})'
        label2 = f'({label2})'

        out = self.array * tensor2.array
        if self.verbose:
            print('Final set of indices', [idx for idx in out.dims if idx in default_indices_names])
            
        return Tensor(out, label1+label2,self.symmetric_indices_sets+symmetric_indices_sets2)

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