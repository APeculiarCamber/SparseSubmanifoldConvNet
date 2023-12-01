import torch.nn as nn
import torch 

def toLongTensor(dimension, x):
    if hasattr(x, 'type') and x.type() == 'torch.LongTensor':
        return x
    elif isinstance(x, (list, tuple)):
        assert len(x) == dimension
        return torch.LongTensor(x)
    else:
        return torch.LongTensor(dimension).fill_(x)


class InputLayer(nn.Module):
    """
    Takes a tuple (coords, features, batch_size [optional])
    * coords is 2d torch.LongTensor with size
       N x dimension   (batch size == 1)
    or
       N x (dimension+1)  (first d columns are coordinates, last column is batch index)

    * features is a CPU or CUDA float tensor with size

      N x n_feature_planes

    * batch_size if given, set a lower bound on the the number of samples in the output tensor.
    Batch size can normally be inferred from the last column of coords, but this may fail if
    some of the batch items are totally empty.

    In case of repetition in coords:
    mode == 0 if the input is guaranteed to have no duplicates
    mode == 1 to use the last item at each spatial location
    mode == 2 to keep the first item at each spatial location
    mode == 3 to sum feature vectors sharing one spatial location
    mode == 4 to average feature vectors at each spatial location

    Output is a SparseConvNetTensor
    """
    def __init__(self, dimension, spatial_size, mode=3):
        nn.Module.__init__(self)
        self.dimension = dimension
        self.spatial_size = toLongTensor(dimension, spatial_size)
        self.mode = mode
        self.device = None

    def to(self, device):
        self.device=device
        return self

    def forward(self, input):
        output = SparseConvNetTensor(
            metadata=Metadata(
                self.dimension),
            spatial_size=self.spatial_size)
        output.features = InputLayerFunction.apply(
            self.dimension,
            output.metadata,
            self.spatial_size,
            input[0].cpu().long(),
            input[1].to(self.device) if self.device else input[1],
            0 if len(input) == 2 else input[2],
            self.mode
        )
        return output


class OutputLayer(Module):
    """
    Used in conjunction with an InputLayer for 'autoencoder' style networks
    Takes a SparseConvNetTensor and results a float Tensor of size

    N x n_feature_planes

    N is defined by the InputLayer

    Behavior during forward-/back-propagation depends on the InputLayer's mode
    """
    def __init__(self, dimension):
        Module.__init__(self)
        self.dimension = dimension

    def forward(self, input):
        output = OutputLayerFunction.apply(
            self.dimension,
            input.metadata,
            input.features
        )
        return output
