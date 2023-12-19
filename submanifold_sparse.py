import torch
import pytorch_custom_apis


def unravel_index(ind, shape):
    '''
    ind = tensor of shape N
    shape = tuple of size V
    '''
    assert ind.dtype == torch.long

    out = torch.empty((len(ind), len(shape)), dtype=torch.long, device=ind.device)
    for i, dim in enumerate(reversed(shape)):
        out[:, -i -1] = ind % dim
        ind = ind // dim
    return out

def ravel_index(inds, shape):
    '''
    inds = tensor of shape N, V
    shape = tuple of size V
    '''
    assert inds.dtype == torch.long
    
    strides = [1]
    for s in reversed(shape[1:]):
        strides.append(strides[-1] * s)
    strides = torch.tensor([list(reversed(strides))], dtype=torch.long, device=inds.device)
    return torch.sum(inds * strides, -1)


from itertools import product
def is_in_bounds(ind : torch.Tensor, shape):
    max_bounds = torch.tensor([[*shape]], dtype=torch.long, device=ind.device)
    in_min =  torch.all(0 <= ind, dim=-1)
    in_max =  torch.all(max_bounds > ind, dim=-1)
    return torch.logical_and(in_min, in_max)

'''
*********************************************************************************************************************************************
*********************************************************************************************************************************************
*********************************************************************************************************************************************
'''

class SparseRep():
    def __init__(self, hash : torch.Tensor, 
                 d_indices : torch.Tensor, d_shape : torch.Size):
        self.hash = hash.contiguous()
        self.d_indices = d_indices.contiguous()
        self.dense_shape = d_shape

    def query_hash_map(self, queries):
        return pytorch_custom_apis.query_hash_map(queries, self.hash)

class Rulebook:
    def __init__(self, rules : torch.Tensor, rules_count : torch.Tensor, kernel_size : int, 
                 in_dense_shape : torch.Size, out_dense_shape : torch.Size, out_rep : SparseRep):
        self.rules = rules
        self.rules_count = rules_count
        self.kernel_size = kernel_size
        self.in_dense_shape = in_dense_shape
        self.out_dense_shape = out_dense_shape
        self.out_rep = out_rep

# DEP
def verify_sparse_hash(s_rep : SparseRep, dense_mat : torch.Tensor):
    dense_mat = dense_mat.cuda()
    d_indices = torch.nonzero(torch.any(dense_mat != 0, dim=-1)).squeeze(1)
    for hi, i in enumerate(s_rep.hash[:, 0]):
        if i == torch.iinfo(torch.long).max: continue
        if not torch.any(d_indices == i).item() or \
            not torch.all(dense_mat[i] == s_rep.data[s_rep.hash[hi, 1]]):
            #print("Hash fail for:", i)
            #print(dense_mat[i], s_rep.data[s_rep.hash[hi, 1]])
            return False
    return True

def convert_dense_to_sparse(mat : torch.Tensor):
    '''
    ASSUMES B, H, W, D, C
    Turn the dense matrix into the sparsified format
    '''
    dense_shape = mat.shape
    mat = mat.reshape(-1, mat.shape[-1]).cuda() # flatten
    d_indices = torch.nonzero(torch.any(mat != 0, dim=-1)).squeeze(1) # sparse determine function
    s_data = mat[d_indices]
    #print("Going in with", d_indices.shape)
    hash_map = pytorch_custom_apis.make_hash_map(d_indices)
    s_rep = SparseRep(hash_map, d_indices, dense_shape)
    #print("Coming out")
    #v = verify_sparse_hash(s_rep, mat)
    #print("verified", v)
    return s_rep, s_data

def convert_sparse_to_dense(data : torch.Tensor, s_rep : SparseRep):
    ''' Turn the sparse representation into a dense matrix '''
    dense_mat = torch.zeros((*s_rep.dense_shape[:-1], data.shape[-1]), dtype=data.dtype, device=data.device)
    dense_mat.view(-1, data.shape[-1])[s_rep.d_indices] += data
    return dense_mat.permute(0, 4, 1, 2, 3).contiguous()

def copy_sparse(s_rep : SparseRep):
    return SparseRep(s_rep.hash.clone(), s_rep.d_indices.clone(), s_rep.dense_shape)

'''
*********************************************************************************************************************************************
*********************************************************************************************************************************************
*********************************************************************************************************************************************
'''

import time
def make_rulebook_3d(in_rep : SparseRep, out_rep : SparseRep, kernel_size):
    with torch.no_grad():
        st_time = time.time()
        # assert torch.all(out_rep.d_indices == ravel_index(unravel_index(out_rep.d_indices, out_rep.dense_shape[:-1]), out_rep.dense_shape[:-1]))
        
        #print("Looking at making rulebook for", out_rep.d_indices.shape)
        #print(out_rep.d_indices.tolist())
        
        # g_out_inds = unravel_index(out_rep.d_indices, out_rep.dense_shape[:-1])
        spec_g_out_inds = unravel_index(out_rep.d_indices, out_rep.dense_shape[1:-1])
        #print("OUT IND:", g_out_inds)

        sub_kernal_bar = (kernel_size - 1) // 2
        bk = [-sub_kernal_bar, (kernel_size - 1) - sub_kernal_bar]
        # assert kernel_size == (abs(bk[1] - bk[0]) + 1)
        
        rule_counts = torch.empty(1 + (kernel_size*kernel_size*kernel_size), dtype=torch.long, device=spec_g_out_inds.device)
        rule_counts[0] = 0 # TODO : REMEMBER THIS!
        
        rules = []

        for i, (kz, ky, kx) in enumerate(product(range(bk[0], bk[1]+1), range(bk[0], bk[1]+1), range(bk[0], bk[1]+1))):
            # Get ravelled kernel mat index
            unravel_offset = torch.tensor([[kz, ky, kx]], dtype=torch.long, device=spec_g_out_inds.device)
            
            k = ((((kz - bk[0]) * kernel_size) + (ky - bk[0])) * kernel_size) + (kx - bk[0])
            # get stride step (not accounting for channel)
            s = in_rep.dense_shape
            stride_step = (((((kz) * s[-3]) + (ky)) * s[-2]) + (kx))

            # get in bounds, remember: for bottom-right (max) input should be higher
            # g_in_inds = torch.tensor([[0, kz, ky, kx]], dtype=torch.long, device=g_out_inds.device) + g_out_inds
            spec_g_in_inds = unravel_offset + spec_g_out_inds
            # g_in_bound = is_in_bounds(g_in_inds, out_rep.dense_shape[:-1])
            spec_g_in_bound = is_in_bounds(spec_g_in_inds, out_rep.dense_shape[1:-1])
            # assert torch.all(spec_g_in_bound == g_in_bound)

            # g_in_inds_bound = torch.nonzero(g_in_bound).squeeze(1)
            g_in_inds_bound = torch.nonzero(spec_g_in_bound).squeeze(1)
            
            # get has input, and the sparse index of the input
            # g_vin_inds = ravel_index(g_in_inds[g_in_inds_bound], out_rep.dense_shape[:-1])
            spec_g_vin_inds = out_rep.d_indices[g_in_inds_bound] + stride_step
            # assert torch.all(g_vin_inds == spec_g_vin_inds)

            spec_s_vin_inds = in_rep.query_hash_map(spec_g_vin_inds)
            # s_vin_inds = in_rep.query_hash_map(g_vin_inds)
            #print(g_vin_inds, "queried to get", s_vin_inds, torch.iinfo(torch.long).max)
            # s_has_input = s_vin_inds != torch.iinfo(torch.long).max
            spec_s_has_input = spec_s_vin_inds != torch.iinfo(torch.long).max
            #print(s_has_input)

            # get the sparse output index, which is in order thanks to the way we order d_ind and data
            spec_s_vin_inds = spec_s_vin_inds[spec_s_has_input]
            spec_s_vout_inds = g_in_inds_bound[spec_s_has_input]
            
            g_ksize = torch.tensor([k], dtype=torch.long, device=spec_g_in_inds.device)
            
            rules.append(torch.stack([g_ksize.expand(len(spec_s_vin_inds)), spec_s_vin_inds, spec_s_vout_inds], -1))
            #print("I:", i, "K:", (kz, ky, kx), ", Stride:", expected_stride_step, "R", len(rules))

            rule_counts[i+1] = len(spec_s_vin_inds)
            # SANITY CHECKS
            # assert torch.all(s_vin_inds < len(in_rep.d_indices))
            # assert torch.all(s_vout_inds < len(out_rep.d_indices))
            # assert s_vout_inds.unique().numel() == s_vout_inds.numel()
            # assert s_vin_inds.unique().numel() == s_vin_inds.numel()
        
        rulebook_prime_data = torch.cat(rules, dim=0)
        del rules

        #print(in_rep.hash)
        return Rulebook(rulebook_prime_data, rule_counts, kernel_size, in_rep.dense_shape, out_rep.dense_shape, out_rep)


def make_rulebook_3d_down(in_rep : SparseRep, kernel_size : int):
    with torch.no_grad():
        '''
        Must return both rulebook and rep. Data not needed
        '''
        # assert in_rep.dense_shape[2] % kernel_size == 0
        ks = kernel_size
        ids = in_rep.dense_shape
        ods = ids[0], ids[1] // ks, ids[2] // ks, ids[3] // ks, ids[4]

        d_in_inds = unravel_index(in_rep.d_indices, in_rep.dense_shape[:-1])
        # d_out_inds per d_in_inds (CAN BE REPEATS)
        #print(d_in_inds.shape)

        d_out_inds = d_in_inds // torch.tensor([[1, ks, ks, ks]], dtype=torch.long, device=in_rep.hash.device)
        
        # Get the kernel element for each active site (there will only be one)
        k_element = torch.remainder(d_in_inds[:, 1:], torch.tensor([[ks, ks, ks]], dtype=torch.long, device=in_rep.hash.device))
        k_element *= torch.tensor([[ks*ks, ks, 1]], dtype=torch.long, device=in_rep.hash.device)
        k_element = k_element.sum(-1)

        # Make flat d_indices for output and hash it
        d_in_targets_flat : torch.Tensor = (d_out_inds * torch.tensor([[ods[1]*ods[2]*ods[3], ods[2]*ods[3], ods[3], 1]], dtype=torch.long, device=in_rep.hash.device)).sum(-1)
        d_out_inds_flat = d_in_targets_flat.unique()
        #print(d_out_inds_flat.shape,d_out_inds_flat.device, d_out_inds_flat.dtype)

        out_inds_hash = pytorch_custom_apis.make_hash_map(d_out_inds_flat)

        # Make sort into rulebook
        rule_sort = torch.argsort(k_element)

        rb_elements = k_element[rule_sort]
        rb_in = rule_sort # This is possible because d_indices correspond to one-to-one with sparse indices at the index level
        rb_out = pytorch_custom_apis.query_hash_map(d_in_targets_flat, out_inds_hash)[rule_sort]

        # RULES COUNTS
        rb_uni_eles, ele_counts = torch.unique(rb_elements, return_counts=True)
        rule_counts = torch.zeros(ks*ks*ks, dtype=torch.long, device=in_rep.hash.device)
        rule_counts[rb_uni_eles] = ele_counts
            
        rules = torch.stack([rb_elements, rb_in, rb_out], -1)
        
        out_rep = SparseRep(out_inds_hash, d_out_inds_flat, ods)
        #print("DOWNSAMPLING FROM", d_in_inds.tolist(), "TO", d_out_inds.tolist())
        rulebook = Rulebook(rules, rule_counts, 2, in_rep.dense_shape, ods, out_rep)

        # SANITY CHECKS
        # assert torch.all(rb_in < len(in_rep.d_indices))
        # assert torch.all(rb_out < len(out_rep.d_indices))
        # for i in range(ks*ks*ks):
        #    rb_in = rules[rules[:,0] == i, 1]
        #    rb_out = rules[rules[:,0] == i, 2]
        #    assert rb_in.unique().numel() == rb_in.numel()
        #    assert rb_out.unique().numel() == rb_out.numel()
        #print(rules)

        return rulebook

'''

Forward Y = K * X + B

Back:
B : automatic
K : dy * dx
X : dx * dy

'''