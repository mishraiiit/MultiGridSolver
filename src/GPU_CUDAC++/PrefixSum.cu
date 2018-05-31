#ifndef PREFIX_SUM_GPU
#define PREFIX_SUM_GPU
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

__global__ void prefixSumGPUSingleThreadKernel(int * in, int n) {
    for(int i = 1; i < n; i++) {
        in[i] += in[i - 1];
    }
}

void prefixSumGPUSingleThread(int * in, int n) {
    prefixSumGPUSingleThreadKernel <<<1,1>>> (in, n);
}

void prefixSumGPUCUB(int * in, int n) {
    void *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, in, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, in, n);
    cudaFree(d_temp_storage);
}

void prefixSumGPUCPUTransfer(int * in, int n) {
    int * cpu_in = (int *) malloc(sizeof(int) * n);
    cudaMemcpy(cpu_in, in, sizeof(int) * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < n; i++) {
        cpu_in[i] += cpu_in[i - 1];
    }
    cudaMemcpy(in, cpu_in, sizeof(int) * n, cudaMemcpyHostToDevice);
    free(cpu_in);
}

void prefixSumGPU(int * in, int n) {
    #ifdef BLELLOCH
        prefixSumGPUCUB(in, n);
    #else
        prefixSumGPUSingleThread(in, n);
    #endif
}

#endif
