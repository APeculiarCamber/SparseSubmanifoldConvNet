import os
import torch

def list_directories(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def list_models(path):
    directories = [os.path.join(path, d) for d in os.listdir(path) if ".off" in d]
    return directories


def read_off(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Check if the file is in OFF format
        if lines[0].strip() != 'OFF':
            raise ValueError("Not a valid OFF file.")

        # Read the number of vertices, faces, and edges
        num_vertices, num_faces, _ = map(int, lines[1].split())

        vertices = []
        faces = []

        # Read vertices
        for line in lines[2:2+num_vertices]:
            vertex = list(map(float, line.split()))
            vertices.append(vertex)

        # Read faces
        for line in lines[2+num_vertices:]:
            face = list(map(int, line.split()[1:]))
            faces.append(face)

        vertices_tensor = torch.tensor(vertices)
        faces_tensor = torch.tensor(faces)

        return vertices_tensor, faces_tensor

import subprocess
def process_mesh(model_path, out_dir, id):
    
    if os.path.exists(model_path[:-4] + ".binvox"): 
        print(model_path[:-4] + ".binvox", "exists")
        return None
    subprocess.run(['~/cuda_voxelizer/build/cuda_voxelizer', '-f', '-s', model_path, '256', "-o", "binvox"])
    print(model_path[:-4] + ".binvox", "now created", os.path.exists(model_path[:-4] + ".binvox"))


directory_path = 'ModelNet10'

directories_list = list_directories(directory_path)
type_counts = dict([(d, [0, 0]) for d in directories_list])

for model_dir in directories_list:
    test_models = list_models(os.path.join(directory_path, model_dir, "test"))
    train_models = list_models(os.path.join(directory_path, model_dir, "train"))

    type_counts[model_dir] = [len(train_models), len(test_models)]
    
    train_dir = f"bin_out/{model_dir}/train/"
    print(train_models)
    os.makedirs(train_dir, exist_ok=True)
    for i, tm in enumerate(train_models): process_mesh(tm, train_dir, i)

    test_dir = f"bin_out/{model_dir}/test/"
    print(test_models)
    os.makedirs(test_dir, exist_ok=True)
    for i, tm in enumerate(test_models): process_mesh(tm, train_dir, i)

print(type_counts)