/*
    Description : This file contains the functions to access an element in
    the matrix formats present in MatrixIo.cu file.

    @author : mishraiiit
*/
    
#ifndef MATRIX_ACCESS
#include "MatrixIO.cu"
#define MATRIX_ACCESS

/*
    Description : This function can be used to access a particular element  in
    a given CSR matrix by it's row and column. This function works both on
    CPU and GPU.

    Returns : The value at the element at (i, j).

    Parameters : 
        MatrixCSR * matrix : The input matrix.
        int i : The row of the element to access.
        int j : The col of the element to access.

    Comments : It's expected from the user that i and j will be in bounds
    of the matrix.
*/

__host__ __device__ float getElementMatrixCSR(MatrixCSR * matrix, int i, int j) {
    int l = matrix->i[i];
    int r = matrix->i[i + 1];
    if(l == r) return 0.0;

    int boundary = r;
    while(l != r) {
        int mid = l + (r - l) / 2;
        if(matrix->j[mid] < j) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }

    if(l < boundary && matrix->j[l] == j)
        return matrix->val[l];
    else
        return 0.0;
}


/*
    Description : This function can be used to access a particular element  in
    a given CSC matrix by it's row and column. This function works both on
    CPU and GPU.

    Returns : The value at the element at (i, j).

    Parameters : 
        MatrixCSC * matrix : The input matrix.
        int i : The row of the element to access.
        int j : The col of the element to access.

    Comments : It's expected from the user that i and j will be in bounds
    of the matrix.
*/

__host__ __device__ float getElementMatrixCSC(MatrixCSC * matrix, int i, int j) {
    int l = matrix->j[j];
    int r = matrix->j[j + 1];
    if(l == r) return 0.0;

    int boundary = r;
    while(l != r) {
        int mid = l + (r - l) / 2;
        if(matrix->i[mid] < i) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }

    if(l < boundary && matrix->i[l] == i)
        return matrix->val[l];
    else
        return 0.0;
}

#endif