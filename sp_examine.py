
# WE WANT TO TIME:
    # SPARSE CONV, AT GRID_SIZE, FEATURE SIZE
    # DENSE CONV , AT GRID_SIZE, FEATURE SIZE
    # SPARSE GEN , AT GRID_SIZE
    # RULEBOOK GEN, AT GRID_SIZE, KERNEL_SIZE
    #   USING : RANDOM SAMPLE, SPHERE, RANDOM_MODEL
# WE ALSO WANT: GO FOR ATOMIC AND INPLACE

import submanifold_sparse as sms
import torch
import time
from sparse_conv3d_atomic import SparseConv3d_Atomic as AtomicConv
from sparse_conv3d_inplace import SparseConv3d_InPlace as InPlaceConv
import torch.nn as nn

def examine_sparsify(x):
    dx = x.permute(0, 2, 3, 4, 1)
    torch.cuda.synchronize()
    st_time = time.time()
    s_rep, x = sms.convert_dense_to_sparse(dx)
    print("For", dx.shape, len(x) / (dx.shape[0]*dx.shape[1]*dx.shape[2]*dx.shape[3]))
    torch.cuda.synchronize()
    return time.time() - st_time, s_rep, x

def examine_densify(s_rep, x):
    d_inds = s_rep.d_indices
    d_shape = s_rep.dense_shape
    torch.cuda.synchronize()
    st_time = time.time()
    d_out = torch.zeros(d_shape[0]*d_shape[1]*d_shape[2]*d_shape[3],x.shape[1], device=x.device)
    d_out[d_inds] = x
    d_out = d_out.reshape(d_shape[0],d_shape[1],d_shape[2],d_shape[3],x.shape[1]).permute(0, 4, 1, 2, 3)
    torch.cuda.synchronize()
    return time.time() - st_time, d_out

def examine_make_down_rulebook(s_rep : sms.SparseRep, kernel_size : int):
    torch.cuda.synchronize()
    st_time = time.time()
    rb = sms.make_rulebook_3d_down(s_rep, kernel_size)
    torch.cuda.synchronize()
    return time.time() - st_time, rb

def examine_make_uni_rulebook(s_rep : sms.SparseRep, kernel_size : int):
    torch.cuda.synchronize()
    st_time = time.time()
    rb = sms.make_rulebook_3d(s_rep, s_rep, kernel_size)
    torch.cuda.synchronize()
    return time.time() - st_time, rb

def examine_use_rulebook(rulebook : sms.Rulebook, data : torch.Tensor, conv_type, out_features, bias=False):
    conv = conv_type(rulebook.kernel_size, data.shape[1], out_features, bias=bias).cuda()
    conv.load_data(None, rulebook)
    torch.cuda.synchronize()
    st_time = time.time()
    res = conv(data)
    torch.cuda.synchronize()
    return time.time() - st_time, res

# output.backward(grad_output)

def examine_back_rulebook(rulebook : sms.Rulebook, data : torch.Tensor, conv_type, out_features, bias=False):
    conv = conv_type(rulebook.kernel_size, data.shape[1], out_features, bias=bias).cuda()
    conv.load_data(None, rulebook)
    fake_data = torch.rand_like(data, device=data.device)
    res : torch.Tensor = conv(fake_data)
    fake_grad = torch.rand_like(res, device=data.device)
    mem_use = torch.cuda.memory_allocated(device="cuda")
    torch.cuda.synchronize()
    st_time = time.time()
    b = res.backward(fake_grad)
    torch.cuda.synchronize()
    return time.time() - st_time, b, mem_use

def examine_use_conv(data : torch.Tensor, kernel_size, out_features, stride=1, bias=False):
    conv = nn.Conv3d(data.shape[1], out_features, kernel_size, stride, padding=kernel_size//2).cuda()
    torch.cuda.synchronize()
    st_time = time.time()
    res = conv(data)
    torch.cuda.synchronize()
    return time.time() - st_time, res

def examine_back_conv(data : torch.Tensor, kernel_size, out_features, stride=1, bias=False):
    conv = nn.Conv3d(data.shape[1], out_features, kernel_size, stride, padding=kernel_size//2).cuda()
    fake_data = torch.rand_like(data, device=data.device)
    res = conv(fake_data)
    fake_grad = torch.rand_like(res, device=data.device)
    mem_use = torch.cuda.memory_allocated(device="cuda")

    torch.cuda.synchronize()
    st_time = time.time()
    b = res.backward(fake_grad)
    torch.cuda.synchronize()
    return time.time() - st_time, b, mem_use


# WE WANT TO TIME:
    # SPARSE CONV, AT GRID_SIZE, FEATURE SIZE
    # DENSE CONV , AT GRID_SIZE, FEATURE SIZE
    # SPARSE GEN , AT GRID_SIZE
    # RULEBOOK GEN, AT GRID_SIZE, KERNEL_SIZE
    #   USING : RANDOM SAMPLE, SPHERE, RANDOM_MODEL
# WE ALSO WANT: GO FOR ATOMIC AND INPLACE
import binvox
import numpy as np

def get_test_model(id=0):
    global_data = ["ModelNet10/chair/train/chair_0043.off_128.binvox"][id]
    v = binvox.Binvox(np.zeros((128, 128, 128), dtype=bool), axis_order="xyz")
    v = v.read(global_data, mode='dense')
    return torch.from_numpy(v.data).cuda()

def adjust_input(input : torch.Tensor, grid_size):
    bs = input.shape[2] // grid_size
    gs = grid_size
    input = input.reshape(-1, gs, bs, gs, bs, gs, bs).permute(0, 1, 3, 5, 2, 4, 6).reshape(-1,gs,gs,gs,bs*bs*bs).contiguous()
    input = torch.any(input, dim=-1)
    input = input.unsqueeze(0)
    return input.to(torch.float)



import json

def generate_sparsify_data(sizes, feature_count):
    model = get_test_model(id=0)
    results = {"sizes": sizes, "on_count": [], "times": []}
    for s in sizes:
        m = adjust_input(model, s)
        results["on_count"].append((m.count_nonzero().item(), m.numel()))
        m = m.repeat(1, feature_count, 1, 1, 1)
        t,_,_ = examine_sparsify(m)
        results["times"].append(t)
    print("\n&&&\n\n\n")
    return results

def generate_densify_data(sizes, feature_count=16):
    model = get_test_model(id=0)
    results = {"sizes": sizes, "on_count": [], "times": []}
    for s in sizes:
        m = adjust_input(model, s)
        results["on_count"].append((m.count_nonzero().item(), m.numel()))
        print(m.shape)
        _, s_rep, x = examine_sparsify(m)
        x = x.repeat(1, feature_count)
        t, _ = examine_densify(s_rep, x)
        results["times"].append(t)
    
    return results


def generate_uni_rulebook_data(sizes, kernel_size):
    model = get_test_model(id=0)
    results = {"sizes": sizes, "on_count": [], "times": []}
    for s in sizes:
        m = adjust_input(model, s)
        assert m.is_cuda
        results["on_count"].append((m.count_nonzero().item(), m.numel()))
        _, s_rep, x = examine_sparsify(m)
        assert s_rep.d_indices.is_cuda
        t, _ = examine_make_uni_rulebook(s_rep, kernel_size)
        results["times"].append((s, t))
    
    return results


def generate_use_rulebook_data(sizes, kernel_size, ConvType=InPlaceConv, feature_count=16):
    model = get_test_model(id=0)
    results = {"sizes": sizes, "on_count": [], "conv_times": []}
    for s in sizes:
        m = adjust_input(model, s)
        results["on_count"].append((m.count_nonzero().item(), m.numel()))
        _, s_rep, x = examine_sparsify(m)
        x = x.repeat(1, feature_count)
        print(x.shape)
        _, rb = examine_make_uni_rulebook(s_rep, kernel_size)
        tf, _ = examine_use_rulebook(rb, x, ConvType, feature_count)
        tb, _, mem_use = examine_back_rulebook(rb, x, ConvType, feature_count)
        results["conv_times"].append({"time_fore": tf, "time_back": tb, "mem": mem_use})
    return results

def generate_conv_data(sizes, kernel_size, feature_count=16):
    model = get_test_model(id=0)
    results = {"sizes": sizes, "on_count": [], "conv_times": []}
    for s in sizes:
        m = adjust_input(model, s)
        results["on_count"].append((m.count_nonzero().item(), m.numel()))
        tf, _ = examine_use_conv(m, kernel_size, feature_count)
        tb, _, mem_use = examine_back_conv(m, kernel_size, feature_count)
        results["conv_times"].append({"size": s, "time_fore": tf, "time_back": tb, "mem": mem_use})
    return results


if __name__ == "__main__":
    sizes = [16, 32, 64, 128]
    feature_count = 32
    kernel_size = [3, 5, 7]
    sparse_results = generate_sparsify_data(sizes, feature_count)
    dense_results = generate_densify_data(sizes, feature_count)

    rb_results = [(k, generate_uni_rulebook_data(sizes, k)) for k in kernel_size]
    rb_results = [(k, generate_uni_rulebook_data(sizes, k)) for k in kernel_size]
    print("Genned UNI")
    conv_results = [(k, generate_conv_data(sizes, k, feature_count)) for k in kernel_size]
    conv_results = [(k, generate_conv_data(sizes, k, feature_count)) for k in kernel_size]
    print("Genned DENSE CONV")

    ip_conv_results = [(k, generate_use_rulebook_data(sizes, k, InPlaceConv, feature_count)) for k in kernel_size]
    ip_conv_results = [(k, generate_use_rulebook_data(sizes, k, InPlaceConv, feature_count)) for k in kernel_size]
    print("Genned IP")

    at_conv_results = [(k, generate_use_rulebook_data(sizes, k, AtomicConv, feature_count)) for k in kernel_size]
    at_conv_results = [(k, generate_use_rulebook_data(sizes, k, AtomicConv, feature_count)) for k in kernel_size]
    print("Genned ATOMIC")
    
    with open('RES_sparse_dense.json', 'w') as file:
        json.dump({"sparse": sparse_results, "dense": dense_results}, file, indent=4)
    
    # SPARSE/DENSE PLOT: BAR GRAPH FOR EACH SIZE 

    with open('RES_make_rulebook.json', 'w') as file:
        json.dump({"uni_rulebook": rb_results}, file, indent=4)

    # RB PLOT: LINE PLOT WITH LEGEND FOR EACH KERNEL SIZE

    with open('RES_use_rulebook.json', 'w') as file:
        json.dump({"inplace": ip_conv_results, "atomic": at_conv_results, "conv": conv_results}, file, indent=4)
    
    # CONV PLOT: LINE PLOT WITH LEGEND FOR EACH KERNEL SIZE
    # SIDE BY SIDE? IN SAME GRAPH???    

    # !!! SAME FOR BACK !!!