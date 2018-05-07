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
		for(int j = 0; j < matrix->cols; j++) {
			printf("%lf ", getElementMatrixCSR(matrix, i, j));
		}
		printf("\n");
	}
}

__global__ void debugCSC(MatrixCSC * matrix) {
	for(int i = 0; i < matrix->rows; i++) {
		for(int j = 0; j < matrix->cols; j++) {
			printf("%lf ", getElementMatrixCSC(matrix, i, j));
		}
		printf("\n");
	}
}

int main() {

	auto tempCSR = readMatrixGPUMemoryCSR("../../matrices/SmallTestMatrix.mtx");
	debugCSR <<<1,1>>> (tempCSR);
	cudaDeviceSynchronize();

	auto tempCSC = readMatrixGPUMemoryCSC("../../matrices/SmallTestMatrix.mtx");
	debugCSC <<<1,1>>> (tempCSC);
	cudaDeviceSynchronize();

	int number_of_blocks = 1;
	int number_of_threads = 1024;
	comptueRowColumnAbsSum <<<number_of_blocks, number_of_threads>>> (tempCSR, tempCSC, NULL);
	cudaDeviceSynchronize();
	
	return 0;
}
