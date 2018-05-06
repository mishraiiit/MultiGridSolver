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

int main() {

	// auto tempCOO = readMatrixGPUMemoryCOO("../../matrices/poisson10000.mtx");
	// debugCOO <<<1,1>>> (tempCOO);
	// cudaDeviceSynchronize();
	

	auto tempCSR = readMatrixGPUMemoryCSR("../../matrices/poisson10000.mtx");
	debugCSR <<<1,1>>> (tempCSR);
	cudaDeviceSynchronize();

	return 0;


	

	/*
	for(int i = 0; i < temp->nnz; i++) {
		cout << temp->i[i] << " " << temp->j[i] << " " << temp->val[i] << endl;
	}
	*/
	return 0;
}
