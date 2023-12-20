import binvox
import numpy as np
import matplotlib.pyplot as plt

v = binvox.Binvox(np.zeros((128, 128, 128), dtype=bool), axis_order="xyz")
v = v.read("ModelNet10/bed/train/bed_0039.binvox", mode='dense')
print(v.data.nonzero())
print("SDF")


def visualize_boolean_tensor(tensor, view_angle=(30, 30)):
    # Create meshgrid for 3D plot
    x, y, z = np.indices(tensor.shape)

    # Get coordinates where tensor values are True
    x_true, y_true, z_true = x[tensor], y[tensor], z[tensor]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot True values as filled cubes
    ax.scatter(x_true, y_true, z_true, c='blue', marker='o', s=100)

    # Set limits and labels
    ax.set_xlim([0, tensor.shape[0]])
    ax.set_ylim([0, tensor.shape[1]])
    ax.set_zlim([0, tensor.shape[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the view angle
    ax.view_init(*view_angle)

    plt.title('3D Boolean Tensor Visualization')
    plt.show()
