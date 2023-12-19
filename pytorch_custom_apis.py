import torch as th
import gp_apis


'''

****************    HASH KEYS    ****************

'''


def make_hash_map(keys : th.Tensor, scale=4):
    assert keys.dim() == 1 and keys.dtype == th.long and keys.device != "cpu"

    hash_map = th.full((keys.shape[0] * scale, 2), th.iinfo(th.long).max, device=keys.device, dtype=th.long)
    gp_apis.gp_make_hash_map(keys, hash_map, keys.device)
    return hash_map

def query_hash_map(keys : th.Tensor, hash_map : th.Tensor):
    assert keys.dim() == 1 and keys.dtype == th.long and keys.device != "cpu"
    assert hash_map.dim() == 2 and hash_map.dtype == th.long and hash_map.device != "cpu"

    out_vals = th.zeros_like(keys, dtype=th.long, device=keys.device)
    gp_apis.gp_query_hash_map(keys, hash_map, out_vals, keys.device)
    return out_vals

'''
********************************************************************************
********************************************************************************
********************************************************************************
'''




class apply_rulebook_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, rulebook, kernel, in_vals, rulebook_cumsum, dim0, dim1):
        dev = rulebook.device
        out_vals = th.zeros((dim0, dim1), device=dev)

        assert kernel.is_contiguous()
        assert rulebook.is_contiguous()
        assert in_vals.is_contiguous()
        assert out_vals.is_contiguous()

        gp_apis.gp_apply_rulebook(rulebook, kernel, in_vals, out_vals, dev)
        ctx.save_for_backward(rulebook, kernel, in_vals, rulebook_cumsum)    
        return out_vals

    @staticmethod
    def backward(ctx, dZ):

        dev = dZ.device
        rulebook, kernel, in_vals, rulebook_cumsum = ctx.saved_tensors
        t_kernel = kernel.permute(0, 2, 1).contiguous()
        dx, dw = None, None
        # COMPUTE DX
        if ctx.needs_input_grad[1]:
            dx = th.zeros_like(in_vals, device=dev)
            assert(t_kernel.is_contiguous())
            assert(t_kernel.shape[1] == dZ.shape[1])
            assert(t_kernel.shape[2] == dx.shape[1])
            apply_rulebook_back_dx(rulebook, t_kernel, dx, dZ.contiguous())
            
        
        # COMPUTE DW
        if ctx.needs_input_grad[2]:
            dw = th.zeros_like(kernel, device=dev)
            
            for i in range(len(rulebook_cumsum) - 1):
                # KERNEL KERNEL ELEMENT
                sh_start = rulebook_cumsum[i]
                sh_end = rulebook_cumsum[i+1]
                if (sh_end - sh_start) == 0: continue

                in_mat = th.index_select(in_vals, 0, rulebook[sh_start:sh_end,1]).T # INPUTS
                dy_mat = th.index_select(dZ, 0, rulebook[sh_start:sh_end,2]) # DY
                dw[i] = th.matmul(in_mat, dy_mat)
            # apply_rulebook_back_dw(rulebook, dw, in_vals, dZ, dev)

        return None, dw, dx, None, None, None

def apply_rulebook(rulebook, kernel, in_vals, rulebook_cumsum, dim0, dim1):
    return apply_rulebook_impl.apply(rulebook, kernel, in_vals, rulebook_cumsum, dim0, dim1)







def apply_rulebook_back_dx(rulebook, t_kernel, in_dx_vals, out_dy_vals):
    '''
    Rulebook :   the rulebook (maybe adjusted)
    t_kernel :   the kernels with element matrices possibly transposed
    in_dx_vals:  the output dx
    out_dy_vals: the input dy for the backprop
    '''
    gp_apis.gp_apply_rulebook_back_dx(rulebook, t_kernel, in_dx_vals, out_dy_vals, rulebook.device)

def apply_rulebook_back_dw(rulebook, out_dw_kernel, in_vals, out_dy_vals):
    '''
    Rulebook :      the rulebook (maybe adjusted)
    out_dw_kernel : the out kernel dw
    in_vals     :   in_vals (not transposed)
    out_dy_vals :   the input dy for the backprop
    '''
    gp_apis.gp_apply_rulebook_back_dw(rulebook, out_dw_kernel, in_vals, out_dy_vals, rulebook.device)

