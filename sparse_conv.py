import torch
import pytorch_custom_apis

class SparseRep():
    def __init__(self, hash, data, d_indices, d_shape : torch.Size):
        self.hash = hash.contiguous()
        self.data = data.contiguous()
        self.d_indices = d_indices.contiguous()
        self.dense_shape = d_shape

def verify_sparse_hash(s_rep : SparseRep, dense_mat : torch.Tensor):
    dense_mat = dense_mat.cuda()
    d_indices = torch.nonzero(torch.any(dense_mat != 0, dim=-1)).squeeze()
    for hi, i in enumerate(s_rep.hash[:, 0]):
        if i == torch.iinfo(torch.long).max: continue
        if not torch.any(d_indices == i).item() or \
            not torch.all(dense_mat[i] == s_rep.data[s_rep.hash[hi, 1]]):
            print("Hash fail for:", i)
            print(dense_mat[i], s_rep.data[s_rep.hash[hi, 1]])
            return False
    return True

def full_sparsify_mat(mat : torch.Tensor):
    '''
    Turn the dense matrix into the sparsified format
    '''
    dense_shape = mat.shape
    mat = mat.reshape(-1, mat.shape[-1]).cuda() # flatten
    d_indices = torch.nonzero(torch.any(mat != 0, dim=-1)).squeeze() # sparse determine function
    print(d_indices.shape)
    s_data = mat[d_indices]
    print(s_data.shape)
    print("Going in with", d_indices.shape)
    hash_map = pytorch_custom_apis.make_hash_map(d_indices)
    print("Coming out")
    s_rep = SparseRep(hash_map, s_data, d_indices, dense_shape)
    verify_sparse_hash(s_rep, mat)
    print("Verified")
    return s_rep

def make_sparse_sub_manifold_output(in_rep : SparseRep, out_size : int):
    '''
    Make a sparse representation of the output
    '''
    return SparseRep(in_rep.hash.clone(), torch.zeros(in_rep.data.shape[0], out_size).cuda(), in_rep.d_indices.clone(), in_rep.dense_shape)


def unravel_index(ind, shape):
    '''
    ind = tensor of shape N
    shape = tuple of size V
    '''
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



def verify_3d_rulebook(in_rep : SparseRep, out_rep : SparseRep, rulebook, kernel_size):
    assert torch.all(out_rep.d_indices == ravel_index(unravel_index(out_rep.d_indices, out_rep.dense_shape[:-1]), out_rep.dense_shape[:-1])).item()

    # remember that d_indices ignores the channel dim
    rulebook_prime_data = torch.empty((0, 3), dtype=torch.long, device=out_rep.d_indices.device)
    # print(rulebook_prime_data)
    print("Looking at verifying rulebook for", out_rep.d_indices.shape)
    print(out_rep.d_indices.tolist())
    
    g_out_inds = unravel_index(out_rep.d_indices, out_rep.dense_shape[:-1])
    bk = [-(kernel_size // 2), kernel_size // 2] # TODO : assumes odd

    for kz, ky, kx in product(range(bk[0], bk[1]+1), range(bk[0], bk[1]+1), range(bk[0], bk[1]+1)):
        
        k = ((((kz - bk[0]) * kernel_size) + (ky - bk[0])) * kernel_size) + (kx - bk[0])
        s = in_rep.dense_shape
        expected_stride_step = (((((kz) * s[-3]) + (ky)) * s[-2]) + (kx))

        # get in bounds, remember: for bottom-right (max) input should be higher
        g_off_inds = torch.tensor([[0, kz, ky, kx]], dtype=torch.long, device=g_out_inds.device) + g_out_inds
        g_in_bound = is_in_bounds(g_off_inds, out_rep.dense_shape[:-1])

        # get has input
        g_voff_inds = ravel_index(g_off_inds[g_in_bound], out_rep.dense_shape[:-1])
        g_has_input = torch.any(g_voff_inds.unsqueeze(1) == in_rep.d_indices.unsqueeze(0), dim=-1)

        # get sparse index (ASSUMES HASH IGNORES Channel)
        g_vin_inds = g_voff_inds[g_has_input]
        g_vout_inds = out_rep.d_indices[g_in_bound][g_has_input]

        g_vin_hash = torch.any(g_vin_inds.unsqueeze(0) == in_rep.hash[:,0].unsqueeze(1), dim=-1)
        g_vout_hash = torch.any(g_vout_inds.unsqueeze(0) == out_rep.hash[:,0].unsqueeze(1), dim=-1)
        g_vout = out_rep.hash[g_vout_hash,1]
        g_vin = in_rep.hash[g_vin_hash,1]
        g_ksize = torch.full((len(g_vin),), k, dtype=torch.long, device=g_vin_hash.device)

        rules = torch.stack([g_ksize, g_vin, g_vout], -1)

        print(rules.tolist(), "K:", k, ", Stride:", expected_stride_step)
        rulebook_prime_data = torch.cat([rulebook_prime_data, rules], dim=0)

    return rulebook_prime_data
        #rulebook.data_ptr[rbInd + 0] = kernelInd;
        #rulebook.data_ptr[rbInd + 1] = in_s_ind;
        #rulebook.data_ptr[rbInd + 2] = out_s_ind;

def make_3d_rulebook(in_rep : SparseRep, out_rep : SparseRep, kernel_size, format='None'):
    '''
    '''
    # Returns (Kernel_Size**3) tensor of count
    rulebook = None
    ''' TODO TODO TODO
    num_rules = pytorch_custom_apis.get_rules_count(in_rep, out_rep, kernel_size, in_rep.dense_shape)
    print(num_rules)
    cumsum_rules = torch.cumsum( torch.cat([torch.tensor([0], device='cuda:0'), num_rules.flatten()]), dim=0)
    print(cumsum_rules)
    rulebook = pytorch_custom_apis.make_rulebook_std(in_rep, out_rep, kernel_size, cumsum_rules, in_rep.dense_shape)
    # print(rulebook)
    '''
    verified_rulebook = verify_3d_rulebook(in_rep, out_rep, rulebook, kernel_size)
    return verified_rulebook

def has_unique_elements(tensor : torch.Tensor):
    unique_elements = tensor.unique(sorted=False)
    return unique_elements.numel() == tensor.numel()

def verify_use_of_3d_rulebook(in_rep : SparseRep, kernels : torch.Tensor, rulebook : torch.Tensor, result):
    out_rep = make_sparse_sub_manifold_output(in_rep, kernels.shape[2])

    shifts = torch.empty(kernels.shape[0] + 1, dtype=torch.long, device=rulebook.device)
    shifts[0] = 0
    shifts = (torch.nonzero(rulebook[1:,0] - rulebook[:-1,0]) + 1).squeeze()
    print("SHIFTS:", shifts.tolist())
    shifts = torch.cat([torch.tensor([0],dtype=torch.long,device=rulebook.device), shifts, torch.tensor([len(rulebook)],dtype=torch.long,device=rulebook.device)], dim=0)
    for i in range(len(shifts)-1):
        sh_start = shifts[i]
        sh_end = shifts[i+1]
        kernel = kernels[rulebook[sh_start, 0]]
        print("Trying kernel", rulebook[sh_start, 0].item(), "with ", sh_start.item(), "to", sh_end.item())
        in_inds = rulebook[sh_start:sh_end, 1]
        out_inds = rulebook[sh_start:sh_end, 2]
        assert has_unique_elements(in_inds) and has_unique_elements(out_inds)
        out_rep.data[out_inds] += torch.matmul(in_rep.data[in_inds], kernel)
    return out_rep

def use_rulebook_std_inplace(in_rep, out_rep, rulebook, kernel_mats):
    print(in_rep.data.device, out_rep.data.device, rulebook.device, kernel_mats.device)
    print(in_rep.data)
    pytorch_custom_apis.use_rulebook_std_inplace(in_rep, out_rep, rulebook, kernel_mats)
    verified_result = verify_use_of_3d_rulebook(in_rep, kernel_mats, rulebook, out_rep)
    print(verified_result.data)
    print(out_rep.data)
    print(out_rep.data - verified_result.data)

def make_mask_for_dense(mat):
    return torch.any(mat != 0, dim=-1)

def make_indices_tensor(mask):
    return torch.nonzero(mask)


if __name__ == "__main__":
    mat = torch.zeros(2, 4, 4, 4, 4)
    mat[torch.rand(*mat.shape) > 0.97] = 1.0
    kernels = torch.ones(3*3*3, 4, 8, device='cuda:0')
    print(mat.shape)
    in_s_rep = full_sparsify_mat(mat)
    out_s_rep = make_sparse_sub_manifold_output(in_s_rep, 8)
    rulebook = make_3d_rulebook(in_s_rep, out_s_rep, 3)

    use_rulebook_std_inplace(in_s_rep, out_s_rep, rulebook, kernels)