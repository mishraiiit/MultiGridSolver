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

__global__ void debugmuij(MatrixCSR * matrix, double * Si) {
	for(int i = 0; i < matrix->rows; i++) {
		for(int j = 0; j < matrix->cols; j++) {
			printf("%d %d %lf\n", i, j, muij(i, j, matrix, Si));
		}
	}
}

int main() {



	TicToc readtime("Read time total");
	readtime.tic();

	string filename = "../matrices/poisson10000.mtx";

	auto tempCSRCPU = readMatrixCPUMemoryCSR(filename);
	auto tempCSCCPU = readMatrixCPUMemoryCSC(filename);
	auto tempCSR = readMatrixGPUMemoryCSR(filename);
	auto neighbour_list = readMatrixGPUMemoryCSR(filename);
	auto tempCSC = readMatrixGPUMemoryCSC(filename);
	// debugCSR <<<1,1>>> (tempCSR);
	// cudaDeviceSynchronize();

	
	// debugCSC <<<1,1>>> (tempCSC);
	// cudaDeviceSynchronize();

	readtime.toc();


	TicToc cudaalloctime("cudaalloctime");
	cudaalloctime.tic();

	double * Si;
	cudaMallocManaged(&Si, sizeof(double) * tempCSRCPU->rows);

	bool * ising0;
	cudaMallocManaged(&ising0, sizeof(bool) * tempCSRCPU->rows);

	bool * allowed;
	cudaMallocManaged(&allowed, sizeof(bool) * tempCSRCPU->nnz);

	double * Si_host = (double *) malloc(sizeof(double) * tempCSRCPU->rows);

	int * paired_with;
	cudaMallocManaged(&paired_with, sizeof(int) * tempCSRCPU->rows);

	int * inmis;
	cudaMallocManaged(&inmis, sizeof(int) * tempCSRCPU->rows);

	cudaalloctime.toc();

	// comptueSiHost(tempCSRCPU, tempCSCCPU, Si_host);

	TicToc rowcolsum("Row Col abs sum");
	rowcolsum.tic();

	int number_of_blocks = (tempCSRCPU->rows + 1024 - 1) / 1024;
	int number_of_threads = 1024;
	computeRowColAbsSum <<<number_of_blocks, number_of_threads>>> (tempCSR, tempCSC, ising0, 8.0);
	cudaDeviceSynchronize();

	rowcolsum.toc();

	TicToc sicomputation("Si computation");
	sicomputation.tic();

	comptueSi<<<number_of_blocks, number_of_threads>>> (tempCSR, tempCSC, Si);
	// debugmuij<<<1,1>>> (tempCSR, Si);
	cudaDeviceSynchronize();

	sicomputation.toc();

	TicToc sortcomputation("Sort computation");
	sortcomputation.tic();

	sortNeighbourList<<<number_of_blocks, number_of_threads>>> (tempCSR, neighbour_list, Si, allowed, 8, ising0);
	//printNeighbourList<<<1,1>>> (tempCSR, neighbour_list, Si);
	cudaDeviceSynchronize();

	sortcomputation.toc();

	mis<<<1,1>>> (tempCSR, inmis);

	aggregation_initial<<<number_of_blocks, number_of_threads>>> (tempCSRCPU->rows, paired_with);
	aggregation<<<number_of_blocks, number_of_threads>>> (tempCSRCPU->rows, inmis, neighbour_list, paired_with, allowed, tempCSR, Si);
	cudaDeviceSynchronize();
	// for(int i = 0; i < tempCSRCPU->rows; i++) {
	// 	for(int j = 0; j < tempCSRCPU->rows; j++) {
	// 		printf("%d %d %lf\n", i, j, muij(i, j, tempCSRCPU, Si_host));
	// 	}
	// }
	
	return 0;
}