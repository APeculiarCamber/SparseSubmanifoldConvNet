

ModelNet10 Dataset:
    Download ModelNet10
    Use a tool like cuda_voxelizer to convert all meshes into binvox files of size 128 x 128 x 128.
        - (Jeroen Baert, Cuda Voxelizer: A GPU-accelerated Mesh Voxelizer, https://github.com/Forceflow/cuda_voxelizer)
        - Using model_net_voxelizer.py (overwrite ~/cuda_voxelizer/build/cuda_voxelizer with your path)
    Or download from Google Drive and extract (TODO: LINK).
    Ensure that Model10 is in the deepgen-code directory.
    pip install binvox

Compiling CUDA (Standard deepgen-code use)
    Make a copy of kernel.cu
    Run python main.py
    Copy the kernel.cu back
    Run ./compile_kernel_from_scratch.sh

Training and Testing:
    Run python sp_training.py
        Note the various command line options in the if __name__ == "__main__" block
    !!! WARNING !!!: This code uses WandB for logging, remove instances if needed.


Files:
    kernel.cu: GPU Hash map and Atomic Rulebook SMS Convolution implementations
    pytorch_custom_apis.cu: Initialization code for linking to C++/CUDA kernel calls
    sp_resnet.py: Contains the ResNet implementations for both dense and SMS ResNets
    sparse_conv3d_atomic.py: Implementation of the atomic SMS rulebook use scheme. Kernel implementations in kernel.cu
    sparse_conv3d_inplace.py: Implementation of the Pytorch only rulebook use scheme.
        - Running python sparse_conv3d_inplace.py or sparse_conv3d_atomic.py will result in evaluation code being run.
    sp_examine.py: Evaluation code for testing runtimes
    sp_modelnet_dataset.py: Custom dataset class for ModelNet10.
        Please ensure that the directory structure assumed matches the actual.