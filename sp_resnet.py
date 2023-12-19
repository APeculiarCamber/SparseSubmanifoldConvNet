import torch
import torch.nn as nn

# Define the basic building blocks of ResNet
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # HERE
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # HERE
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicLayer(nn.Module):
    def __init__(self, resnet, block, planes, num_blocks, stride):
        super(BasicLayer, self).__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(resnet.in_planes, planes, stride))
            resnet.in_planes = planes
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, num_blocks, dim_size, m, in_features=1, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.start_m = m

        self.conv1 = nn.Conv3d(in_features, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = BasicLayer(self, BasicBlock, m, num_blocks[0], stride=1)
        self.layer2 = BasicLayer(self, BasicBlock, m*2, num_blocks[1], stride=2)
        self.layer3 = BasicLayer(self, BasicBlock, m*4, num_blocks[2], stride=2)
        self.layer4 = BasicLayer(self, BasicBlock, m*8, num_blocks[3], stride=2)
        fs = (((dim_size // 2) // 2) // 2)
        self.linear = nn.Linear(m*8 * fs * fs * fs, num_classes)


    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # TODO : maybe pool
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Define ResNet-18
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], 64, 32, num_classes=10)




'''
*******************************************************************************************************
*******************************************************************************************************
*******************************************************************************************************
'''

import torch
import torch.nn as nn
from sparse_conv3d_inplace import SparseConv3d_InPlace
from sparse_conv3d_atomic import SparseConv3d_Atomic
import submanifold_sparse as sms

class SpBatchNorm(nn.Module):
    def __init__(self, num_features):
        '''
        TODO : not verified to be appropriate
        '''
        super(SpBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if len(x) <= 1: return x
        return self.bn(x)

# Define the basic building blocks of ResNet
class SpBasicBlock(nn.Module):

    def __init__(self, in_planes, planes, ConvOp, stride=1):
        super(SpBasicBlock, self).__init__()
        self.is_down_block = stride != 1

        # HERE
        self.conv1 = ConvOp(stride if self.is_down_block else 3, in_planes, planes)
        self.bn1 = SpBatchNorm(planes)
        self.conv2 = ConvOp(3, planes, planes)
        self.bn2 = SpBatchNorm(planes)

        # DOWNSAMPLING BY CONV OP, this lessens residual skip
        self.shortcut = nn.Sequential()
        if self.is_down_block:
            self.shortcut = nn.Sequential(
                ConvOp(stride if self.is_down_block else 3, in_planes, planes, bias=False),
                SpBatchNorm(planes),
            )
        elif in_planes != planes:
            # ASSUMES INPUT IS TENSOR NOT SPARSE REP
            self.shortcut = nn.Sequential(
                nn.Linear(in_planes, planes, bias=False),
                nn.BatchNorm1d(planes, )
            )
        self.relu = nn.ReLU()

    def load_data(self, x_rep : sms.SparseRep, str_rulebook : sms.Rulebook, down_rulebook : sms.Rulebook):
        self.x_rep = x_rep
        self.str_rulebook = str_rulebook
        self.down_rulebook = down_rulebook

    def forward(self, x):
        if self.is_down_block:
            self.conv1.load_data(self.x_rep, self.down_rulebook)
            out = self.relu(self.bn1(self.conv1(x)))
            self.conv2.load_data(self.down_rulebook.out_rep, self.str_rulebook)
            out = self.bn2(self.conv2(out))
            # TODO : this here should not be a conv, it should be a maxpool
            self.shortcut[0].load_data(self.x_rep, self.down_rulebook)
            out_data = self.shortcut[0](x)
            out_data = self.shortcut[1](out_data)
            out += out_data
            out = self.relu(out)
        else:
            self.conv1.load_data(self.x_rep, self.str_rulebook)
            out = self.relu(self.bn1(self.conv1(x)))
            self.conv2.load_data(self.x_rep, self.str_rulebook)
            out = self.bn2(self.conv2(out))
            out_data = self.shortcut(x)
            out.data += out_data
            out = self.relu(out)
        return out

class SpBasicLayer(nn.Module):
    def __init__(self, resnet, ConvOp, block, planes, num_blocks, stride):
        super(SpBasicLayer, self).__init__()

        self.strides = [stride] + [1] * (num_blocks - 1)
        self.is_down = stride != 1
        layers = []
        for stride in self.strides:
            layers.append(block(resnet.in_planes, planes, ConvOp, stride))
            resnet.in_planes = planes
        self.blocks = nn.Sequential(*layers)
        self.out_rep = None

    def load_data(self, x_rep, str_rulebook=None, down_rulebook=None):
        str_x = x_rep
        if self.is_down: 
            down_rulebook = sms.make_rulebook_3d_down(x_rep, 2)
            str_x = down_rulebook.out_rep

        # Handle straight shot, new_x == x iff no downsampling else new_x == down(x)
        if str_rulebook == None: str_rulebook = sms.make_rulebook_3d(str_x, str_x, 3)
        
        for i, block in enumerate(self.blocks): 
            block.load_data(x_rep, str_rulebook, down_rulebook)
            if i == 0: x_rep = str_x

        self.out_rep = str_x
    
    def forward(self, x):
        # Handle downsampling
        return self.blocks(x)


# Define the ResNet model
class SpResNet(nn.Module):
    def __init__(self, num_blocks, dim_size, m, ConvOp=SparseConv3d_InPlace, in_features=1, num_classes=1000):
        super(SpResNet, self).__init__()
        self.in_planes = in_features
        self.start_m = m

        self.layer0 = SpBasicLayer(self, ConvOp, SpBasicBlock, m, 1, stride=1)
        self.layer1 = SpBasicLayer(self, ConvOp, SpBasicBlock, m, num_blocks[0], stride=1)
        self.layer2 = SpBasicLayer(self, ConvOp, SpBasicBlock, m*2, num_blocks[1], stride=2)
        self.layer3 = SpBasicLayer(self, ConvOp, SpBasicBlock, m*4, num_blocks[2], stride=2)
        self.layer4 = SpBasicLayer(self, ConvOp, SpBasicBlock, m*8, num_blocks[3], stride=2)
        pool_size = 2
        self.pool = None
        if pool_size > 1:
            self.pool = nn.MaxPool3d(2)
        fs = max((((dim_size // 2) // 2) // 2) // pool_size, 1)
        self.linear = nn.Linear(m*8 * fs * fs * fs, num_classes)
        self.test_linear = nn.Linear(1, num_classes)

    def forward(self, x):
        # CONVERT
        x = x.permute(0, 2, 3, 4, 1)
        assert x.shape[-2] == x.shape[-3] and x.shape[-3] == x.shape[-4]

        x_rep, x_data = sms.convert_dense_to_sparse(x)

        self.layer0.load_data(x_rep)
        out = self.layer0(x_data) # Returns srb for kernel 3 at FULL
        self.layer1.load_data(self.layer0.out_rep)
        out = self.layer1(out)  # Uses srb for kernel 3 at FULL
        self.layer2.load_data(self.layer1.out_rep)
        out = self.layer2(out)
        self.layer3.load_data(self.layer2.out_rep)
        out = self.layer3(out)
        self.layer4.load_data(self.layer3.out_rep)
        out = self.layer4(out)
        
        d_inds = self.layer4.out_rep.d_indices
        d_shape = self.layer4.out_rep.dense_shape
        d_out = torch.zeros(d_shape[0]*d_shape[1]*d_shape[2]*d_shape[3], out.shape[1], device=out.device)
        d_out[d_inds] += out
        d_out = d_out.reshape(d_shape[0],d_shape[1],d_shape[2],d_shape[3], out.shape[1]).permute(0, 4, 1, 2, 3)

        if self.pool is not None: d_out = self.pool(d_out)

        perc = self.linear(d_out.reshape(d_shape[0], -1))

        return perc

# Define ResNet-18
def SpResNet18():
    return SpResNet([2, 2, 2, 2], dim_size=64, m=32, ConvOp=SparseConv3d_InPlace, num_classes=10).cuda()

import torch
import torch.nn as nn
import torch.optim as optim

'''
# Access gradients of the network parameters
for name, param in resnet18.named_parameters():
    # print(f"Parameter: {name}, Gradient: {param.grad}")
    print(f"Parameter: {name}, Gradient: {param.grad}")
'''