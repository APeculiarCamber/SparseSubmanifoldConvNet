#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#define LONG_TYPE unsigned long long
#define MAX_LONG 0x7FFFFFFFFFFFFFFF
#define MIN(x, y) (x > y) ? y : x
#define MAX(x, y) (x < y) ? y : x


__device__ LONG_TYPE d_Hash(LONG_TYPE key, LONG_TYPE hashSize) {
    /*key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key % hashSize;*/

    /*
    unsigned long long int A = 2654435769;
    int hash = ((unsigned long long int)key * A) % hashSize;
    return hash;*/

    const LONG_TYPE m = 0x5bd1e995;
    const LONG_TYPE r = 24;

    LONG_TYPE h = 1232 ^ sizeof(LONG_TYPE);

    key *= m;
    key ^= key >> r;
    key *= m;

    h *= m;
    h ^= key;

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h % hashSize;
}

__device__ int d_InsertHash(LONG_TYPE key, LONG_TYPE val, LONG_TYPE* hashMap, LONG_TYPE hashSize) {
    //  atomicCAS(int* address, int compare, int val)

    int hash = d_Hash(key, hashSize) * 2;
    while (atomicCAS(&hashMap[hash], MAX_LONG, key) != MAX_LONG) {
        hash = (hash + 2) % (hashSize*2);
    }
    hashMap[hash + 1] = val;
    return hash;
}

__global__ void d_MakeMap(array1d_t<LONG_TYPE> keys, array2d_t<LONG_TYPE> hash_map) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int globalWarp = tid / 32;
    int warpID = tid % 32;
    int numThreads = blockDim.x * gridDim.x;
    int totalWork = keys.col_count;
    int numPerWarp = ((totalWork + numThreads - 1) / numThreads) * 32;

    int lb = globalWarp * numPerWarp;
    int rb = MIN(totalWork, lb + numPerWarp);

    // WARP TOGETHER
    int v;
    int i = lb;
    for (; i+32 < rb; i += 32) {
        v = i + warpID;
        d_InsertHash(keys.data_ptr[v], v, hash_map.data_ptr, hash_map.row_count);
        __syncwarp();
    }
    
    // WARP DIVERGE
    v = i + warpID;
    if (v < rb) {
        d_InsertHash(keys.data_ptr[v], v, hash_map.data_ptr, hash_map.row_count);
    }
}




/*
* Keys : Assumes N size array of long longs for hashing {key -> key_index}
* hash_map: Assumes N*S x 2 tensor of hashmap (starting all MAX_LONG_VALUE)
*/
void make_hash_map(array1d_t<float>& keys, array2d_t<float>& hash_map) {
    array1d_t<LONG_TYPE> ll_keys((LONG_TYPE*)keys.data_ptr, keys.col_count);
    array2d_t<LONG_TYPE> ll_hash_map((LONG_TYPE*)hash_map.data_ptr, hash_map.row_count, hash_map.col_count);

    const int numThreads = 256;
    const int numBlocks = MIN(256, (ll_keys.col_count+numThreads-1) / numThreads);

    d_MakeMap<<<numBlocks, numThreads>>>(ll_keys, ll_hash_map);
}



__device__ LONG_TYPE d_FindHashedVal(LONG_TYPE key, LONG_TYPE* hashMap, LONG_TYPE hashSize) {
    
    int hash = d_Hash(key, hashSize) * 2;
    LONG_TYPE hashKey = hashMap[hash];
    while ((hashKey != key) & (hashKey != MAX_LONG)) {
        hash = (hash + 2) % (hashSize * 2);
        hashKey = hashMap[hash];
    }
    return (hashKey != MAX_LONG) ? hashMap[hash+1] : MAX_LONG;
}

__global__ void d_QueryMap(array1d_t<LONG_TYPE> keys, array2d_t<LONG_TYPE> hash_map, array1d_t<LONG_TYPE> out_vals) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int globalWarp = tid / 32;
    int warpID = tid % 32;
    int numThreads = blockDim.x * gridDim.x;
    int totalWork = keys.col_count;
    int numPerWarp = ((totalWork + numThreads - 1) / numThreads) * 32;

    int lb = globalWarp * numPerWarp;
    int rb = MIN(totalWork, lb + numPerWarp);

    // WARP TOGETHER
    int v;
    int i = lb;
    for (; i+32 < rb; i += 32) {
        v = i + warpID;
        out_vals.data_ptr[v] = d_FindHashedVal(keys.data_ptr[v], hash_map.data_ptr, hash_map.row_count);
        __syncwarp();
    }

    // WARP DIVERGE
    v = i + warpID;
    if (v < rb) {
        out_vals.data_ptr[v] = d_FindHashedVal(keys.data_ptr[v], hash_map.data_ptr, hash_map.row_count);
    }
}

/*
* Keys : Assumes N size array of long longs for hashing {key -> key_index}
* hash_map: Assumes N*S x 2 tensor of hashmap (starting all MAX_LONG_VALUE)
* out_vals: Assumes N size array of long longs for storing VALUES (or MAX_LONG_VALUE)
*/
void query_hash_map(array1d_t<float>& keys, array2d_t<float>& hash_map, array1d_t<float>& out_vals){
    array1d_t<LONG_TYPE> ll_keys((LONG_TYPE*)keys.data_ptr, keys.col_count);
    array2d_t<LONG_TYPE> ll_hash_map((LONG_TYPE*)hash_map.data_ptr, hash_map.row_count, hash_map.col_count);
    array1d_t<LONG_TYPE> ll_out_vals((LONG_TYPE*)out_vals.data_ptr, out_vals.col_count);
    
    const int numThreads = 256;
    const int numBlocks = MIN(256, (ll_keys.col_count+numThreads-1) / numThreads);
    d_QueryMap<<<numBlocks, numThreads>>>(ll_keys, ll_hash_map, ll_out_vals);
}





/**
 * ***********************************************************************************
 * ***********************************************************************************
*/












/**
 * inLine: 1 x N
 * denseMat : N x M
 * outLine: 1 x M
*/
#define LOC_TILE_SIZE 32
#define SYNC_LINE_THREADS __syncwarp
// TODO : might make this per-block and use __syncthreads
__device__ void d_oneLineMatMul(float* inLine, float* kMat, float* outLine, int inSize, int outSize, float* sharedIn, int warpID) {
    int numChunks = (inSize + LOC_TILE_SIZE - 1) / LOC_TILE_SIZE;

    // ITERATE OVER OUTPUTS
    for (int floorOut = 0; floorOut < outSize; floorOut += 32) {
        int o = floorOut + warpID;

        float val = 0.0;
        // ITERATE OVER INPUTS
        for (int iChunk = 0; iChunk < numChunks; iChunk++) {

            int iStep = (iChunk*LOC_TILE_SIZE);

            // LOAD INPUTS
            for (int i = warpID; (i < LOC_TILE_SIZE) && ((iStep + i) < inSize); i += 32) {
                sharedIn[i] = inLine[(iChunk*LOC_TILE_SIZE) + i];
            }
            SYNC_LINE_THREADS();

            // USE INPUTS
            if (o < outSize) {
                for (int i = 0; (i < LOC_TILE_SIZE) && ((iStep + i) < inSize); i++) {
                    int realInIndex = iStep + i;
                    float inVal = sharedIn[i];
                    float kVal = kMat[(realInIndex * outSize) + o];
                    val += inVal * kVal;
                }
            }
            SYNC_LINE_THREADS();
        }
        // APPLY THE ADDITION
        if (o < outSize) {
            atomicAdd(&outLine[o], val);
        }
        
        SYNC_LINE_THREADS();
    }
}

__global__ void d_ApplyRulebook(array2d_t<LONG_TYPE> rulebook, array3d_t<float> kernel, array2d_t<float> inVals, array2d_t<float> outVals) {
    extern __shared__ float sharedInputs[];

    int numThreads = blockDim.x * gridDim.x;
    int numWarps = numThreads / 32;
    int work = rulebook.row_count;

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int localWarp = threadIdx.x / 32;
    int globalWarp = tid / 32;
    int warpID = tid % 32;

    int UWarp = (work+numWarps-1) / numWarps;
    int lb = (globalWarp * UWarp);
    int ub = MIN(work, lb + UWarp);
    float* localSharedInput = &sharedInputs[localWarp * LOC_TILE_SIZE];

    for (int r = lb; r < ub; r++) {
        int rInd = r * rulebook.col_count;
        //assert(r < rulebook.row_count);
        //assert(rulebook.data_ptr[rInd + 0] < kernel.matrix_count);
        //assert(rulebook.data_ptr[rInd + 1] < inVals.row_count);
        //assert(rulebook.data_ptr[rInd + 2] < outVals.row_count);
        
        LONG_TYPE kernelInd = rulebook.data_ptr[rInd + 0] * (LONG_TYPE)(kernel.col_count * kernel.row_count);
        LONG_TYPE inInd = rulebook.data_ptr[rInd + 1] * (LONG_TYPE)inVals.col_count;
        LONG_TYPE outInd = rulebook.data_ptr[rInd + 2] * (LONG_TYPE)outVals.col_count;

        float* inFeatures = &inVals.data_ptr[inInd];
        float* outFeatures = &outVals.data_ptr[outInd];
        float* kMat = &kernel.data_ptr[kernelInd];
        d_oneLineMatMul(inFeatures, kMat, outFeatures, inVals.col_count, outVals.col_count, localSharedInput, warpID);
    }
}


__global__ void d_ApplyRulebook_reversed_book(array2d_t<LONG_TYPE> rulebook, array3d_t<float> kernel, array2d_t<float> inVals, array2d_t<float> outVals) {
    extern __shared__ float sharedInputs[];

    int numThreads = blockDim.x * gridDim.x;
    int numWarps = numThreads / 32;
    int work = rulebook.row_count;

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int localWarp = threadIdx.x / 32;
    int globalWarp = tid / 32;
    int warpID = tid % 32;

    int UWarp = (work+numWarps-1) / numWarps;
    int lb = (globalWarp * UWarp);
    int ub = MIN(work, lb + UWarp);
    float* localSharedInput = &sharedInputs[localWarp * LOC_TILE_SIZE];

    for (int r = lb; r < ub; r++) {
        int rInd = r * rulebook.col_count;
        //assert(r < rulebook.row_count);
        //assert(rulebook.data_ptr[rInd + 0] < kernel.matrix_count);
        //assert(rulebook.data_ptr[rInd + 1] < inVals.row_count);
        //assert(rulebook.data_ptr[rInd + 2] < outVals.row_count);
        
        LONG_TYPE kernelInd = rulebook.data_ptr[rInd + 0] * (LONG_TYPE)(kernel.col_count * kernel.row_count);
        LONG_TYPE outInd = rulebook.data_ptr[rInd + 1] * (LONG_TYPE)outVals.col_count;
        LONG_TYPE inInd = rulebook.data_ptr[rInd + 2] * (LONG_TYPE)inVals.col_count;

        float* inFeatures = &inVals.data_ptr[inInd];
        float* outFeatures = &outVals.data_ptr[outInd];
        float* kMat = &kernel.data_ptr[kernelInd];
        d_oneLineMatMul(inFeatures, kMat, outFeatures, inVals.col_count, outVals.col_count, localSharedInput, warpID);
    }
}

void apply_rulebook(array2d_t<float>& rulebook, array3d_t<float>& kernel, array2d_t<float>& in_vals, array2d_t<float>& out_vals) {
    if (rulebook.row_count == 0) return;

    array2d_t<LONG_TYPE> ll_rulebook((LONG_TYPE*)rulebook.data_ptr, rulebook.row_count, rulebook.col_count);

    size_t numThreads = 256;
    size_t numWarps = numThreads / 32;
    size_t numShared = numWarps * LOC_TILE_SIZE;
    size_t numBlocks = MIN(256, (rulebook.row_count + numWarps - 1) / numWarps);

    //assert(in_vals.col_count == kernel.row_count);
    //assert(out_vals.col_count == kernel.col_count);

    d_ApplyRulebook<<<numBlocks, numThreads, numShared*4>>>(ll_rulebook, kernel, in_vals, out_vals);
}

void apply_rulebook_back_dx(array2d_t<float>& rulebook, array3d_t<float>& t_kernel, array2d_t<float>& in_dx_vals, array2d_t<float>& out_dy_vals){
    if (rulebook.row_count == 0) return;

    array2d_t<LONG_TYPE> ll_rulebook((LONG_TYPE*)rulebook.data_ptr, rulebook.row_count, rulebook.col_count);

    size_t numThreads = 256;
    size_t numWarps = numThreads / 32;
    size_t numShared = numWarps * LOC_TILE_SIZE;
    size_t numBlocks = MIN(256, (rulebook.row_count + numWarps - 1) / numWarps);
    
    assert(out_dy_vals.col_count == t_kernel.row_count);
    assert(in_dx_vals.col_count == t_kernel.col_count);
    
    d_ApplyRulebook_reversed_book<<<numBlocks, numThreads, numShared*4>>>(ll_rulebook, t_kernel, out_dy_vals, in_dx_vals);
}

void apply_rulebook_back_dw(array2d_t<float>& rulebook, array3d_t<float>& out_dw_kernel, array2d_t<float>& in_vals, array2d_t<float>& out_dy_vals){
    // /UNUSED
}
