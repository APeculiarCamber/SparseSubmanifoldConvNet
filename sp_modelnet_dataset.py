import torch
import os
import binvox
import numpy as np

def list_directories(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def list_models(path):
    directories = [os.path.join(path, d) for d in os.listdir(path) if ".binvox" in d]
    return directories

class ModelNetDataset(torch.utils.data.Dataset):
    """Some Information about ModelNetDataset"""
    def __init__(self, dir_path, train=True):
        super(ModelNetDataset, self).__init__()
        self.dir_path = dir_path
        self.sub_path = "train" if train else "test"
        self.is_train = train

        model_list = list_directories(dir_path)
        type_counts = [0] * len(model_list)
        off_counts = [0] * len(model_list)
        self.model_list = model_list

        for i, model_dir in enumerate(model_list):
            data_models = list_models(os.path.join(dir_path, model_dir, self.sub_path))
            off_models = list_models(os.path.join(dir_path, model_dir, "train"))
            type_counts[i] = len(data_models)
            off_counts[i] = len(off_models) if not train else 0

        print(type_counts)
        print(model_list)

        self.model_types = []
        self.model_index = []

        self.type_counts = torch.tensor(type_counts, dtype=torch.int)
        self.type_offset = torch.tensor(off_counts, dtype=torch.int)
        self.type_cumsum = torch.cumsum(self.type_counts, dim=0)
        print(self.type_counts)
        print(self.type_cumsum)



    def __getitem__(self, index):
        model_id = torch.count_nonzero(index >= self.type_cumsum)
        if model_id != 0: index -= self.type_cumsum[model_id-1]
        index += self.type_offset[model_id]

        model_name = self.model_list[model_id]
        fn = os.path.join(self.dir_path, model_name, self.sub_path, f'{model_name}_{index+1:04d}.off_128.binvox')
        v = binvox.Binvox(np.zeros((128, 128, 128), dtype=bool), axis_order="xyz")
        v = v.read(fn, mode='dense')
        model = torch.from_numpy(v.data)
        return model, model_id

    def __len__(self):
        return self.type_cumsum[-1]

if __name__ == "__main__":
    print()
    v = ModelNetDataset("ModelNetZip/ModelNet10", False)

    for i in range(len(v)):
        m, l = v[i]
        print(m.shape, l, i)
    v = ModelNetDataset("ModelNetZip/ModelNet10")
    for i in range(len(v)):
        m, l = v[i]
        print(m.shape, l, i)
