#include "bits/stdc++.h"
#include "Matrix.cu"

using namespace std;

__global__ void debugCOO(MatrixCOO * matrix) {
	for(int i = 0; i < matrix->nnz; i++) {
		printf("%d %d %lf\n", matrix->i[i], matrix->j[i], matrix->val[i]);
	}
}

__global__ void debugCSR(MatrixCSR * matrix) {
	for(int i = 0; i < matrix->rows; i++) {
		for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
			printf("%d %d %lf\n", i, matrix->j[j], matrix->val[j]);
		}
	}
}

__global__ void linebyline(MatrixCSR * matrix) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i == 5) {
		for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
			printf("%d %d %lf----", i, matrix->j[j], matrix->val[j]);
		}
		printf("\n");
	}
}

int main() {

	auto tempCOO = readMatrixGPUMemoryCOO("../../matrices/SmallTestMatrix.mtx");
	debugCOO <<<1,1>>> (tempCOO);
	cudaDeviceSynchronize();

	auto tempCSC = readMatrixUnifiedMemoryCSC("../../matrices/SmallTestMatrix.mtx");
	for(int i = 0; i < tempCSC->cols; i++) {
		for(int j = tempCSC->j[i]; j < tempCSC->j[i + 1]; j++) {
			printf("%d %d %lf\n", tempCSC->i[j], i, tempCSC->val[j]);
		}
	}
	

	// auto tempCSRCPU = readMatrixCPUMemoryCSR("../../matrices/SmallTestMatrix.mtx");
	// auto tempCSR = readMatrixGPUMemoryCSR("../../matrices/SmallTestMatrix.mtx");


	// // debugCSR <<<1,1>>> (tempCSR);
	// cudaDeviceSynchronize();

	// linebyline <<< (tempCSRCPU->rows + 1 - 1) / 1, 1 >>> (tempCSR);
	// cudaDeviceSynchronize();

	return 0;
}
