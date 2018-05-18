#include "Matrix.cu"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cusparse.h>
#include <string>

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

__global__ void debugmuij(MatrixCSR * matrix, float * Si) {
	for(int i = 0; i < matrix->rows; i++) {
		for(int j = 0; j < matrix->cols; j++) {
			printf("%d %d %lf\n", i, j, muij(i, j, matrix, Si));
		}
	}
}

int main() {

	TicToc readtime("Read time total");
	readtime.tic();

	std::string filename = "../matrices/poisson10000.mtx";

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
	int * aggregations_cpu = (int *) malloc(tempCSRCPU->rows * sizeof(int));

	float * Si;
	cudaMalloc(&Si, sizeof(float) * tempCSRCPU->rows);

	bool * ising0;
	cudaMalloc(&ising0, sizeof(bool) * tempCSRCPU->rows);

	bool * allowed;
	cudaMalloc(&allowed, sizeof(bool) * tempCSRCPU->nnz);

	float * Si_host = (float *) malloc(sizeof(float) * tempCSRCPU->rows);

	int * paired_with;
	cudaMalloc(&paired_with, sizeof(int) * tempCSRCPU->rows);

	int * useful_pairs;
	cudaMalloc(&useful_pairs, sizeof(int) * tempCSRCPU->rows);

	int * useful_pairs_cpu_prefix = (int *) malloc(tempCSRCPU->rows * sizeof(int));

	int * aggregations;
	cudaMalloc(&aggregations, sizeof(int) * tempCSRCPU->rows);

	int * aggregation_count;
	cudaMalloc(&aggregation_count, sizeof(int) * tempCSRCPU->rows);

	cudaalloctime.toc();

	
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

	TicToc aggregationtime("Aggregation time");
	aggregationtime.tic();
	aggregation_initial<<<number_of_blocks, number_of_threads>>> (tempCSRCPU->rows, paired_with);
	aggregation<<<number_of_blocks, number_of_threads>>> (tempCSRCPU->rows, neighbour_list, paired_with, allowed, tempCSR, Si, 0, ising0, bfs_distance);
	aggregation<<<number_of_blocks, number_of_threads>>> (tempCSRCPU->rows, neighbour_list, paired_with, allowed, tempCSR, Si, 1, ising0, bfs_distance);
	cudaDeviceSynchronize();
	aggregationtime.toc();

	TicToc get_usefule_pairs_time("Get useful_pairs time");
	get_usefule_pairs_time.tic();
	get_useful_pairs<<<number_of_blocks, number_of_threads>>> (tempCSRCPU->rows, paired_with, useful_pairs);
	get_usefule_pairs_time.toc();


	TicToc prefix_sum("Sum kernel");
	prefix_sum.tic();
	gpu_prefix_sum(tempCSRCPU->rows, useful_pairs);
	cudaMemcpy(useful_pairs_cpu_prefix, useful_pairs, sizeof(int) * tempCSRCPU->rows, cudaMemcpyDeviceToHost);
	prefix_sum.toc();

	int nc;
	cudaMemcpy(&nc, useful_pairs + tempCSRCPU->rows - 1, sizeof(int), cudaMemcpyDeviceToHost);
	mark_aggregations <<<number_of_blocks, number_of_threads>>> (tempCSRCPU->rows, aggregations, useful_pairs);
	cudaDeviceSynchronize();
	cudaMemcpy(aggregations_cpu, aggregations, sizeof(int) * tempCSRCPU->rows, cudaMemcpyDeviceToHost);
	cudaMemcpy(paired_with_cpu, paired_with, sizeof(int) * tempCSRCPU->rows, cudaMemcpyDeviceToHost);

	get_aggregations_count <<< (nc + 1024 - 1) / 1024, 1024 >>> (nc, aggregations, paired_with, aggregation_count);
	cudaDeviceSynchronize();
	gpu_prefix_sum(nc, aggregation_count);
	int nnz_in_p_matrix;
	cudaMemcpy(&nnz_in_p_matrix, aggregation_count + nc - 1, sizeof(int), cudaMemcpyDeviceToHost);
	

	MatrixCSR * P_transpose_cpu = (MatrixCSR *) malloc(sizeof(MatrixCSR));
	P_transpose_cpu->rows = nc;
	P_transpose_cpu->cols = tempCSRCPU->rows;
	P_transpose_cpu->nnz = nnz_in_p_matrix;
	cudaMalloc(&P_transpose_cpu->i, sizeof(int) * (P_transpose_cpu->rows + 1));
	cudaMalloc(&P_transpose_cpu->j, sizeof(int) * (P_transpose_cpu->nnz));
	cudaMalloc(&P_transpose_cpu->val, sizeof(float) * (P_transpose_cpu->nnz));
	create_p_matrix_transpose <<< (nc + 1024 - 1) / 1024, 1024>>> (nc, aggregations, paired_with, aggregation_count, P_transpose_cpu->i, P_transpose_cpu->j, P_transpose_cpu->val);
	cudaDeviceSynchronize();

	MatrixCSR * P_transpose_gpu;
	cudaMalloc(&P_transpose_gpu, sizeof(MatrixCSR));	
	cudaMemcpy(P_transpose_gpu, P_transpose_cpu, sizeof(MatrixCSR), cudaMemcpyHostToDevice);

	P_transpose_cpu = deepCopyMatrixCSRGPUtoCPU(P_transpose_gpu);

	// printCSRCPU(P_transpose_cpu);

	MatrixCSR * shallow_cpu = shallowCopyMatrixCSRGPUtoCPU(P_transpose_gpu);
	
	int * new_i;
	int * new_j;
	float * new_val;

	std::cout << P_transpose_cpu->rows << " " << P_transpose_cpu->cols << " " << P_transpose_cpu->nnz << std::endl;

	cudaMalloc(&new_i, sizeof(int) * (P_transpose_cpu->cols + 1));
	cudaMalloc(&new_j, sizeof(int) * (P_transpose_cpu->nnz));
	cudaMalloc(&new_val, sizeof(float) * (P_transpose_cpu->nnz));

	assert(P_transpose_cpu->rows == shallow_cpu->rows);
	assert(P_transpose_cpu->cols == shallow_cpu->cols);
	assert(P_transpose_cpu->nnz == shallow_cpu->nnz);

	cudaDeviceSynchronize();
	TicToc time_transpose("Time taken csr2csc");
	time_transpose.tic();
	cusparseHandle_t  handle;
    cusparseCreate(&handle);
	cusparseStatus_t status = cusparseScsr2csc(handle, P_transpose_cpu->rows, P_transpose_cpu->cols, P_transpose_cpu->nnz, shallow_cpu->val, shallow_cpu->i, shallow_cpu->j, new_val, new_j, new_i, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
	cudaDeviceSynchronize();
	time_transpose.toc();

	MatrixCSR * P_cpu = (MatrixCSR *) malloc(sizeof(MatrixCSR));
	P_cpu->rows = P_transpose_cpu->cols;
	P_cpu->cols = P_transpose_cpu->rows;
	P_cpu->nnz = P_transpose_cpu->nnz;

	P_cpu->i = (int *) malloc(sizeof(int) * (P_cpu->rows + 1));
	P_cpu->j = (int *) malloc(sizeof(int) * P_cpu->nnz);
	P_cpu->val = (float *) malloc(sizeof(float) * P_cpu->nnz);

	cudaMemcpy(P_cpu->i, new_i, sizeof(int) * (P_cpu->rows + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(P_cpu->j, new_j, sizeof(int) * P_cpu->nnz, cudaMemcpyDeviceToHost);
	cudaMemcpy(P_cpu->val, new_val, sizeof(float) * P_cpu->nnz, cudaMemcpyDeviceToHost);

	// printCSRCPU(P_cpu);
	

	return 0;
}