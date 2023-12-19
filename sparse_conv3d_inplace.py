import torch.nn as nn
import submanifold_sparse as sms
import torch
import torch.nn.init as init
from torch.autograd import Function as AutoFunction

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


def has_unique_elements(tensor : torch.Tensor):
    return tensor.unique().numel() == tensor.numel()


class Conv3D_Inplace_Func(AutoFunction):
    @staticmethod
    def forward(ctx, input, k_weights, rules_counts, rules, out_len):
        
        # Make output sparse mat
        x_out = torch.zeros((out_len, k_weights.shape[-1]), device=input.device)

        # Get start and end indices of kernel elements, TODO : could pop out, should TODO
        shifts = rules_counts.cumsum(dim=0)
        #print("RULES COUNT:", self.rulebook.rules_count.tolist())
        #print("SHIFTS:", shifts.tolist())

        ctx.save_for_backward(input, k_weights, shifts, rules)

        # For each kernel matrix
        for i in range(len(shifts)-1):
            sh_start = shifts[i]
            sh_end = shifts[i+1]
            if (sh_end - sh_start) == 0: continue

            k_ind = rules[sh_start, 0].item()
            # print("Trying kernel", self.rulebook.rules[sh_start, 0].item(), "with ", sh_start.item(), "to", sh_end.item())

            # Get kernel matrix
            kernel = k_weights[k_ind, :, :]
            # Get input and output indices in the sparse matrix
            in_inds = rules[sh_start:sh_end, 1]
            out_inds = rules[sh_start:sh_end, 2]

            # Get input elements
            # Apply matrix multiplication
            # Scatter values to output elements
            x_out = x_out.index_add(0, out_inds, torch.matmul(torch.index_select(input, 0, in_inds), kernel))

        return x_out


    @staticmethod
    def backward(ctx, dY):
        # Retrieve the saved tensor from the context
        input, k_weights, rule_shifts, rules = ctx.saved_tensors

        dx = torch.zeros_like(input, device=input.device)
        dw = torch.zeros_like(k_weights, device=input.device)

        for i in range(len(rule_shifts)-1):
            sh_start = rule_shifts[i]
            sh_end = rule_shifts[i+1]
            if (sh_end - sh_start) == 0: 
                continue

            k_ind = rules[sh_start, 0].item()
            # print("Trying kernel", self.rulebook.rules[sh_start, 0].item(), "with ", sh_start.item(), "to", sh_end.item())

            # Get kernel matrix
            kernel = k_weights[k_ind, :, :]
            # Get input and output indices in the sparse matrix
            in_inds = rules[sh_start:sh_end, 1]
            out_inds = rules[sh_start:sh_end, 2]

            # Get input elements
            # Apply matrix multiplication
            # Scatter values to output elements
            
            # DX
            if ctx.needs_input_grad[0]:
                dx.index_add_(0, in_inds, torch.matmul(torch.index_select(dY, 0, out_inds), kernel.T))

            # DW : IN x OUT
            if ctx.needs_input_grad[1]:
                dw[k_ind] = torch.matmul(torch.index_select(input, 0, in_inds).T, torch.index_select(dY, 0, out_inds))

        return dx, dw, None, None, None

class SparseConv3d_InPlace(nn.Module):
    """Some Information about SparseConv3d_InPlace"""
    def __init__(self, kernel_size, in_features, out_features, bias=True, weights=None):
        super(SparseConv3d_InPlace, self).__init__()
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.empty(kernel_size * kernel_size * kernel_size, in_features, out_features), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(1, out_features), requires_grad=True) if bias else None
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
        rules_count = self.rulebook.rules_count
        rules = self.rulebook.rules
        out_size = len(self.rulebook.out_rep.d_indices)

        # CONV
        x_out = Conv3D_Inplace_Func.apply(x_data, self.weights, rules_count, rules, out_size)
        # BIAS
        if self.bias is not None: x_out += self.bias

        return x_out
        





'''
************************************************************************************************************
************************************************************************************************************
************************************************************************************************************
'''







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
        assert isinstance(x, sms.SparseRep)
        assert rulebook.kernel_size == self.kernel_size
        self.rulebook = rulebook

    def forward(self, x_data):
        
        # Make output sparse mat
        x_out = torch.zeros((len(self.rulebook.out_rep.d_indices), self.out_features), device=x_data.device)

        # Get start and end indices of kernel elements
        shifts = self.rulebook.rules_count.cumsum(dim=0)
        #print("RULES COUNT:", self.rulebook.rules_count.tolist())
        #print("SHIFTS:", shifts.tolist())

        # For each kernel matrix
        for i in range(len(shifts)-1):
            sh_start = shifts[i]
            sh_end = shifts[i+1]
            if (sh_end - sh_start) == 0: continue

            # print("Trying kernel", self.rulebook.rules[sh_start, 0].item(), "with ", sh_start.item(), "to", sh_end.item())

            # Get kernel matrix
            kernel = self.weights[self.rulebook.rules[sh_start, 0].item(), :, :]
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

        if self.bias is not None:
            x_out.data = x_out.data + self.bias[None,:]

        return x_out





if __name__ == "__main__":
    x = torch.randint(0, 2, (2, 64, 64, 64, 4), dtype=torch.float).cuda()
    s_rep, s_data = sms.convert_dense_to_sparse(x)
    rb = sms.make_rulebook_3d(s_rep, s_rep, 3)

    # Create instances of the networks
    torch.cuda.manual_seed(101)
    torch.manual_seed(101)
    net1 = SparseConv3d_InPlace(3, 4, 4, False).cuda()
    torch.cuda.manual_seed(101)
    torch.manual_seed(101)
    net2 = SparseConv3d_InPlace_Autograd(3, 4, 4, False).cuda()
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
    closes = torch.nonzero(~torch.isclose(s_data1.grad, s_data2.grad, atol=0.00001))
    print(closes, s_data1.grad.shape)
    print(s_data1.grad[closes[:,0], closes[:,1]] - s_data2.grad[closes[:,0], closes[:,1]])
    #print(s_data2.grad[closes[0], closes[1]])
    assert torch.allclose(s_data1.grad, s_data2.grad, atol=0.00001)
