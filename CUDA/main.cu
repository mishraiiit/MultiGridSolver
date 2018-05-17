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

	int * paired_with_cpu = (int *) malloc(tempCSRCPU->rows * sizeof(int));

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

	TicToc bfstime("BFS time...");
	bfstime.tic();
	int * bfs_distance = bfs(tempCSRCPU->rows, tempCSR);
	bfstime.toc();
	cudaDeviceSynchronize();

	TicToc sortcomputation("Sort computation");
	sortcomputation.tic();
	sortNeighbourList<<<number_of_blocks, number_of_threads>>> (tempCSR, neighbour_list, Si, allowed, 8, ising0);
	cudaDeviceSynchronize();

	sortcomputation.toc();

	aggregation_initial<<<number_of_blocks, number_of_threads>>> (tempCSRCPU->rows, paired_with);
	for(int i = 0; i < 200; i++) {
		aggregation<<<number_of_blocks, number_of_threads>>> (tempCSRCPU->rows, neighbour_list, paired_with, allowed, tempCSR, Si, i, ising0, bfs_distance);
	}

	cudaMemcpy(paired_with_cpu, paired_with, sizeof(int) * tempCSRCPU->rows, cudaMemcpyDeviceToHost);
	int ans = 0;
	for(int i = 0; i < tempCSRCPU->rows; i++) {
		if(paired_with[i] == -1) {
			
		} else {
			if(i == paired_with[i]) {
				printf("%d %d 1\n", i + 1, ans + 1);
				ans++;
			} else if(i < paired_with[i]) {
				printf("%d %d 1\n", i + 1, ans + 1);
				printf("%d %d 1\n", paired_with[i] + 1, ans + 1);
				ans++;
			}
		}
	}
	return 0;
}