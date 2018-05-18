#ifndef MatrixAccess
#include "MatrixIO.cu"
#define MatrixAccess

__host__ __device__ float getElementMatrixCSR(MatrixCSR * matrix, int i, int j) {
    int l = matrix->i[i];
    int r = matrix->i[i + 1];
    if(l == r) return 0.0;

    while(l != r) {
        int mid = (l + r) / 2;
        if(matrix->j[mid] < j) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }

    if(matrix->j[l] == j)
        return matrix->val[l];
    else
        return 0.0;
}

__host__ __device__ float getElementMatrixCSC(MatrixCSC * matrix, int i, int j) {
    int l = matrix->j[j];
    int r = matrix->j[j + 1];
    if(l == r) return 0.0;

    while(l != r) {
        int mid = (l + r) / 2;
        if(matrix->i[mid] < i) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }

    if(matrix->i[l] == i)
        return matrix->val[l];
    else
        return 0.0;
}

#endif