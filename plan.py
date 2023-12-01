
# Sparsification happens immediately

# Unique-ify to get indices and count

# Make hash map : 
    # HASH MAP FUNCTION: int64_keys_and_index_vals(keys=unique_dense_indices)
    # Need flatten and widen functions

# Copy for output

# Use broadcast to get kernel element offsets all at once
    # shape = kernel_ind X (unique_dense_indices)

# Iterate over kernel elements (except 'center'/one-to-one):
    # Make 'in-bound' mask (could skip this step???)
    # get output indices (nonzero()) and masked-dense-input-indices
    # query hash table int64_find_keys(masked-dense-input-indices), returns hash-table-indices or MAX_LONG
    # Filter again for valid indices then query
    # Make the rulebook segment

import torch

def sp_flatten(t, shape):
    '''
    Expects B x Z x Y x X shape
    '''
    assert t.dtype == torch.long
    t : torch.Tensor = t.detach().clone()
    t[:,:3] *= shape[3]
    t[:,:2] *= shape[2]
    t[:,:1] *= shape[1]
    return t.sum(dim=-1)

def sp_expand(t, shape):
    '''
    Expects B x Z x Y x X shape
    '''
    assert t.dtype == torch.long
    ot = torch.empty(len(t), 4)
    for i in range(3, -1, -1):
        ot[:,i] = t % shape[i]
        t /= shape[i]
    return ot

def uniquify(t : torch.Tensor):
    '''
    Returns (unique_elements, counts)
    '''
    return t.unique(sorted=True, return_counts=True, dim=0)

from copy import deepcopy

class SparseHashMap:
    def __init__(self, keys, vals=None):
        self.map = dict()
        for i, k in enumerate(keys):
            self.map[k.item()] = i

    def query_for_keys(self, keys):
        '''
        Returns value or MAX_LONG
        '''
        # print(f"Querying:\n{self.map}\nFOR:\n{keys}")
        query = torch.empty_like(keys, dtype=torch.long)
        for i, k in enumerate(keys):
            query[i] = self.map.get(k.item(), torch.iinfo(torch.long).max)
        return query


class SparseRep:
    def __init__(self, cells, feature_count, dense_shape):
        self.flat_indices = sp_flatten(cells, dense_shape)
        self.dense_indices = cells
        self.dense_shape = dense_shape
        
        self.hash_map = SparseHashMap(self.flat_indices)
        
        self.vals = torch.empty(len(self.flat_indices), feature_count)
        
    def __str__(self):
        return f"<Sparse Rep of {self.flat_indices.shape} for {self.vals.shape[1]} features>"


def make_rulebook(out_rep : SparseRep, kernel_size, shape, in_rep : SparseRep):
    '''
    out_indices : N x 4
    '''
    t_shape = torch.tensor([shape[0], shape[1], shape[2], shape[3]])

    # Make offsets per kernel
    hk = kernel_size // 2
    ks = kernel_size
    single_indices = torch.arange(kernel_size, dtype=torch.long) - hk
    vz, vy, vx = torch.meshgrid(single_indices, single_indices, single_indices, indexing='ij')
    kernel_indices = torch.column_stack((torch.zeros(pow(kernel_size, 3), dtype=torch.long), vz.ravel(), vy.ravel(), vx.ravel()))
    offset_indices = kernel_indices.detach().clone()    
    kernel_indices[:, 1:] += hk

    # Apply Offsets
    # offset_indices : K^3 x 4
    per_kernel_inputs = offset_indices.unsqueeze(1) + out_rep.dense_indices.unsqueeze(0)
    # per_kernel_inputs : K^3 x N x 4

    # Get elements in bounds
    in_bound_mask = torch.logical_and(torch.all(per_kernel_inputs >= 0, dim=-1), 
                                      torch.all(per_kernel_inputs < t_shape[None, None, :], dim=-1))

    # Make rules per kernel
    rules = []
    rules_counts = []
    for i, (_, z, y, x) in enumerate(kernel_indices):
        k_inputs = per_kernel_inputs[i, in_bound_mask[i]]
        flat_k_inputs = sp_flatten(k_inputs, shape)
        sp_flat_k_inputs = in_rep.hash_map.query_for_keys(flat_k_inputs)
        valid_inputs = sp_flat_k_inputs != torch.iinfo(torch.long).max

        sp_flat_k_inputs = sp_flat_k_inputs[valid_inputs].squeeze()
         # TODO : BELOW: this is tricky so ensure this, assume output entering with same indexing as leaving
        sp_flat_k_outputs = torch.nonzero(in_bound_mask[i])[valid_inputs].squeeze()
        kernel_ele_index = ((((z * ks) + y) * ks) + x).item()

        k_rules = torch.stack([sp_flat_k_inputs, sp_flat_k_outputs, 
                               torch.full_like(sp_flat_k_outputs, kernel_ele_index, dtype=torch.long)], dim=-1)
        
        if len(k_rules.shape) == 1: k_rules = k_rules.unsqueeze(0)
        rules.append(k_rules)
        rules_counts.append(len(k_rules))

    return torch.cat(rules, dim=0), rules_counts

dims = 8
points = torch.rand(100, 3)
points = torch.floor(points * dims).to(torch.long)
points = torch.cat([torch.zeros(len(points), 1, dtype=torch.long), points], dim=1)

cells, count = points.unique(return_counts=True, dim=0)

dense_shape = [1, dims, dims, dims]
sp_in_rep = SparseRep(cells, 1, dense_shape)
sp_in_rep.vals[:] = count[:].unsqueeze(-1)
sp_out_rep = SparseRep(cells, 8, dense_shape)

rulebook, rule_counts = make_rulebook(sp_out_rep, 3, dense_shape, sp_in_rep)
print(sp_in_rep.hash_map.map)
print(rulebook)
print(rule_counts)
