import torch.nn as nn
import submanifold_sparse as sms
import torch
import torch.nn.init as init
import submanifold_sparse as sms
import pytorch_custom_apis as pca

def initialize_weights_xavier(layer):
    if isinstance(layer, nn.Conv2d):
        init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            init.constant_(layer.bias, 0)

def initialize_weights_kaiming(layer):
    if isinstance(layer, nn.Conv2d):
        init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            init.constant_(layer.bias, 0)

class SparseConv3d_Atomic(nn.Module):
    """Some Information about SparseConv3d_InPlace"""
    def __init__(self, kernel_size, in_features, out_features, bias=True):
        super(SparseConv3d_Atomic, self).__init__()
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.empty(kernel_size * kernel_size * kernel_size, in_features, out_features), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=True) if bias else None
        self.in_features = in_features
        self.out_features = out_features

        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.xavier_uniform_(self.weights)
        if self.bias != None: init.constant_(self.bias, 0)

        self.rulebook = None

    def load_data(self, x : sms.SparseRep, rulebook : sms.Rulebook):
        # assert isinstance(x, sms.SparseRep)
        assert rulebook.kernel_size == self.kernel_size
        self.rulebook = rulebook

    def forward(self, x_data):
        
        shifts = self.rulebook.rules_count.cumsum(dim=0)
        # CONV
        out_active_sites = len(self.rulebook.out_rep.d_indices)
        out = pca.apply_rulebook(self.rulebook.rules, self.weights, x_data, shifts, out_active_sites, self.out_features)
        # BIAS
        if self.bias is not None: out += self.bias[None,:]
        return out
    







class SparseConv3d_InPlace_Autograd(nn.Module):
    """Some Information about SparseConv3d_InPlace"""
    def __init__(self, kernel_size, in_features, out_features, bias=True):
        super(SparseConv3d_InPlace_Autograd, self).__init__()
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.empty(kernel_size * kernel_size * kernel_size, in_features, out_features), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=True) if bias else None
        self.in_features = in_features
        self.out_features = out_features

        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.xavier_uniform_(self.weights)
        if self.bias != None: init.constant_(self.bias, 0)

        self.rulebook = None

    def load_data(self, x : sms.SparseRep, rulebook : sms.Rulebook):
        assert rulebook.kernel_size == self.kernel_size
        self.rulebook = rulebook

    def forward(self, x_data):
        
        # Make output sparse mat
        x_out = torch.zeros((len(self.rulebook.out_rep.d_indices), self.out_features), device=x_data.device)

        # Get start and end indices of kernel elements, TODO : could pop out, should TODO
        shifts = self.rulebook.rules_count.cumsum(dim=0)
        #print("RULES COUNT:", self.rulebook.rules_count.tolist())
        #print("SHIFTS:", shifts.tolist())

        # For each kernel matrix
        rules_processed = 0
        for i in range(len(shifts)-1):
            sh_start = shifts[i]
            sh_end = shifts[i+1]
            rules_processed += (sh_end - sh_start).item()
            if (sh_end - sh_start) == 0: continue

            # print("Trying kernel", self.rulebook.rules[sh_start, 0].item(), "with ", sh_start.item(), "to", sh_end.item())

            # Get kernel matrix
            kernel = self.weights[self.rulebook.rules[sh_start, 0].item(), :, :]
            assert torch.all(self.rulebook.rules[sh_start:sh_end, 0] == self.rulebook.rules[sh_start, 0].item())

            print(self.rulebook.rules[sh_start, 0].item(), kernel.data_ptr())
            # Get input and output indices in the sparse matrix
            in_inds = self.rulebook.rules[sh_start:sh_end, 1]
            out_inds = self.rulebook.rules[sh_start:sh_end, 2]

            # Get input elements
            in_vals = torch.index_select(x_data, 0, in_inds)
            # Apply matrix multiplication
            x_mat = torch.matmul(in_vals, kernel)
            # Scatter values to output elements
            x_out.index_add_(0, out_inds, x_mat)
            # x_out = x_out.index_add(0, out_inds, x_mat).requires_grad_()
        
        print("INPLACE RULES PROCESSED:", rules_processed)
        if self.bias is not None:
            x_out.data = x_out.data + self.bias[None,:]

        return x_out





if __name__ == "__main__":
    x = torch.randint(0, 2, (2, 64, 64, 64, 1), dtype=torch.float).cuda()
    x = x.repeat(1, 1, 1, 1, 64)

    s_rep, s_data = sms.convert_dense_to_sparse(x)
    rb = sms.make_rulebook_3d(s_rep, s_rep, 3)
    assert rb.rules[:,1].min() >= 0
    
    # Create instances of the networks
    torch.cuda.manual_seed(101)
    torch.manual_seed(101)
    net1 = SparseConv3d_Atomic(3, 64, 64, True).cuda()
    torch.cuda.manual_seed(101)
    torch.manual_seed(101)
    net2 = SparseConv3d_InPlace_Autograd(3, 64, 64, True).cuda()
    assert torch.allclose(net1.weights, net2.weights)
    # Set the networks to training mode and initialize gradients
    net1.train()
    net2.train()

    # Forward pass
    s_data1 = s_data.clone().requires_grad_()
    s_data2 = s_data.clone().requires_grad_()
    net1.load_data(s_rep, rb)
    net2.load_data(s_rep, rb)
    output1 = net1(s_data1)
    output2 = net2(s_data2)
    print(rb.rules_count)
    print(len(torch.nonzero(~torch.isclose(output1, output2, atol=0.00001))) // 4, len(output2), len(output1))
    print("Mean:", (output1 - output2).abs().mean())
    print(output1[-10:] - output2[-10:])
    assert torch.allclose(output1, output2, atol=0.00001)

    # Compute loss for both networks
    loss1 = output1.sum()
    loss2 = output2.sum()

    # Backpropagation for net1
    net1.zero_grad()
    loss1.backward()

    # Backpropagation for net2
    net2.zero_grad()
    loss2.backward()

    # Compare gradients of corresponding parameters in both networks
    parameters_net1 = list(net1.parameters())
    parameters_net2 = list(net2.parameters())
    
    print(rb.rules_count)
    print(torch.nonzero(~torch.isclose(net1.weights.grad, net2.weights.grad, atol=0.00001)))
    assert torch.allclose(net1.weights.grad, net2.weights.grad)

    print((s_data1.grad - s_data2.grad).mean())
    print(s_data1.grad.mean(), s_data2.grad.mean())
    #print(s_data2.grad[closes[0], closes[1]])
    assert torch.allclose(s_data1.grad, s_data2.grad, atol=0.00001)
