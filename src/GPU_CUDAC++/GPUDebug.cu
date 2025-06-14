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
void initialize_array(int n, T * arr, U value) {
    cudaMemset(arr, value, sizeof(T) * n);
}

#endif