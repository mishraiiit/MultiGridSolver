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

void prefixSumGPU(int * in, int n) {
	int * out;
	cudaMalloc(&out, sizeof(int) * n);
	sum_scan_blelloch(out, in, n);
	copyToInput <<< (n + 1024 - 1) / 1024, 1024 >>> (out, in, n);
	cudaFree(out);
}


// void prefixSumGPUExclusive(int * in, int n) {
// 	int * out;
// 	cudaMalloc(&out, sizeof(int) * n);
// 	sum_scan_blelloch(out, in, n);
// 	cudaMemcpy(in, out, sizeof(int) * n, cudaMemcpyDeviceToDevice);
// 	cudaFree(out);
// }


void prefixSumCPU(int * in, int n) {
	int * cpu_in = (int *) malloc(n * sizeof(int));
	cudaMemcpy(cpu_in, in, sizeof(int) * n, cudaMemcpyDeviceToHost);
	int sum = 0;
	for(int i = 0; i < n; i++) {
		cpu_in[i] += sum;
		sum = cpu_in[i];
	}
	cudaMemcpy(in, cpu_in, sizeof(int) * n, cudaMemcpyHostToDevice);
}

#endif