#ifndef PREFIX_SUM_GPU
#define PREFIX_SUM_GPU
#include "scan.cu"

__global__ void copyToInput(int * out, int * in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    if(i == n - 1) {
        in[i] += out[i];
    } else {
        in[i] = out[i + 1];
    }
}

__global__ void prefixSumGPUSingleThreadKernel(int * in, int n) {
    for(int i = 1; i < n; i++) {
        in[i] += in[i - 1];
    }
}

void prefixSumGPUSingleThread(int * in, int n) {
    prefixSumGPUSingleThreadKernel <<<1,1>>> (in, n);
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
        int * out;
        cudaMalloc(&out, sizeof(int) * n);
        sum_scan_blelloch(out, in, n);
        copyToInput <<< (n + 1024 - 1) / 1024, 1024 >>> (out, in, n);
        cudaFree(out);
    #else
        prefixSumGPUSingleThread(in, n);
    #endif
}

#endif
