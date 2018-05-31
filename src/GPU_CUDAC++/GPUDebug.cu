/*
    Description : This file contains some debug functions for the GPU.

    @author : mishraiiit
*/

#ifndef GPU_DEBUG
#define GPU_DEBUG
#include <stdio.h>
#include "MatrixIO.cu"
#include "MatrixAccess.cu"


/*
    Description : Prints a matrix on GPU in COO format, only nnz entries.

    Parameters : 
        MatrixCSR * matrix : Matrix on GPU.
	
	Comments : Launch with kernel parameters <<<1,1>>> and do device
	synchronization after launching.
*/

__global__ void debugCOO(MatrixCOO * matrix) {
	for(int i = 0; i < matrix->nnz; i++) {
		printf("%d %d %lf\n", matrix->i[i], matrix->j[i], matrix->val[i]);
	}
}


/*
    Description : Prints a matrix on GPU in CSR format, in dense fashion.

    Parameters : 
        MatrixCSR * matrix : Matrix on GPU.

	Comments : Launch with kernel parameters <<<1,1>>> and do device
	synchronization after launching.
*/

__global__ void debugCSR(MatrixCSR * matrix) {
	for(int i = 0; i < matrix->rows; i++) {
		for(int j = 0; j < matrix->cols; j++) {
			printf("%lf ", getElementMatrixCSR(matrix, i, j));
		}
		printf("\n");
	}
}


/*
    Description : Prints a matrix on GPU in CSC format, in dense fashion.

    Parameters : 
        MatrixCSR * matrix : Matrix on GPU.

	Comments : Launch with kernel parameters <<<1,1>>> and do device
	synchronization after launching.        
*/

__global__ void debugCSC(MatrixCSC * matrix) {
	for(int i = 0; i < matrix->rows; i++) {
		for(int j = 0; j < matrix->cols; j++) {
			printf("%lf ", getElementMatrixCSC(matrix, i, j));
		}
		printf("\n");
	}
}

/*
    Description : Prints a variable on GPU.

    Parameters : 
        T * u : address of the variable on GPU.
*/

template<typename T>
__global__ void print_gpu_variable_kernel(T * u) {
    fprintf(stderr, "%d\n", u);
}

template<typename T>
void print_gpu_variable(T * u) {
    print_gpu_variable_kernel <<<1,1>>> (u);
    cudaDeviceSynchronize();
}


/*
    Description : Swaps two variable on GPU.

    Parameters : 
        T & u : first variable to swap.
        T & v : second variable to swap.
*/

template<typename T>
__host__ __device__ void swap_variables(T & u, T & v) {
    T temp = u;
    u = v;
    v = temp;
}


/*
    Description : Can be used to change value at an address on GPU from CPU.

    Parameters : 
        T * node : Address of the variable.
        U value : Value to change it to.
    
    Comments : Launch this kernel with parameters <<<1,1>>>.
*/

template<typename T, typename U>
__global__ void assign(T * node, U value) {
    * node = value;
}

/*
    Description : Assign a value to array on GPU.

    Parameters : 
        int n : Array size.
        T * arr : Array.
        U value : Value to change it to.
*/

template<typename T, typename U>
__global__ void initialize_array_kernel(int n, T * arr, U value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    arr[i] = value;
}

template<typename T, typename U>
void initialize_array(int n, T * arr, U value) {
    cudaMemset(arr, value, sizeof(T) * n);
    // initialize_array_kernel <<< (n + 1024 - 1) / 1024, 1024 >>> (n, arr, value);
}

#endif