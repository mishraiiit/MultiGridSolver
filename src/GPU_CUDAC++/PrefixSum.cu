#ifndef PREFIX_SUM_GPU
#define PREFIX_SUM_GPU
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

void prefixSumGPUCUB(int * in, int n) {
    void *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, in, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, in, n);
    cudaFree(d_temp_storage);
}

void prefixSumGPU(int * in, int n) {
    #ifdef BLELLOCH
        prefixSumGPUCUB(in, n);
    #else
        #error "BLELLOCH is not defined, and no fallback prefix sum implementation is available after refactoring."
    #endif
}

#endif
