/*
    Description : This file contains functions related to matrix operations.

    @author : mishraiiit
*/

#ifndef MATRIX_OPERATIONS
#define MATRIX_OPERATIONS
#include "MatrixIO.cu"
#include "GPUDebug.cu"
#include <cusparse.h>

/*
    Description : It takes a matrix in CSR format and returns it's transpose.

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format.

    Returns : The transpose of input matrix in CSR format.

    @author : mishraiiit
*/

MatrixCSR * transposeCSRCPU(const MatrixCSR * const matrix) {

    MatrixCSR * matrix_trans = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    matrix_trans->rows = matrix->cols;
    matrix_trans->cols = matrix->rows;
    matrix_trans->nnz = matrix->nnz;
    matrix_trans->i = (int *) malloc(sizeof(int) * (matrix_trans->rows + 1));
    matrix_trans->j = (int *) malloc(sizeof(int) * matrix_trans->nnz);
    matrix_trans->val = (float *) malloc(sizeof(float) * matrix_trans->nnz);

    int * col_sum = (int *) calloc(matrix->cols, sizeof(int));
    int * col_freq = (int *) calloc(matrix->cols, sizeof(int));

    for(int i = 0; i < matrix->rows; i++) {
        for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
            assert(matrix->j[j] < matrix->cols);
            col_sum[matrix->j[j]]++;
        }
    }

    matrix_trans->i[0] = 0;
    matrix_trans->i[1] = col_sum[0];
    for(int i = 1; i < matrix->cols; i++) {
        col_sum[i] += col_sum[i - 1];
        matrix_trans->i[i + 1]  = col_sum[i];
    }
    
    assert(matrix_trans->i[matrix->cols] == matrix->nnz);

    for(int i = 0; i < matrix->rows; i++) {
        for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
            int pos = matrix_trans->i[matrix->j[j]] + col_freq[matrix->j[j]];
            assert(pos < matrix_trans->nnz);
            matrix_trans->j[pos] = i;
            matrix_trans->val[pos] = matrix->val[j];
            col_freq[matrix->j[j]]++;
        }
    }

    free(col_sum);
    free(col_freq);

    return matrix_trans;
}

/*
    Description : Prints a matrix on stdout.

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format.

    @author : mishraiiit
*/

void printCSRCPU(const MatrixCSR * const matrix) {
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
            std::cout << i + 1 << " " << matrix->j[j] + 1 << " " << \
            matrix->val[j] << std::endl;
        }
    }
}


/*
    Description : Copies a matrix on GPU to CPU. It will also create a copy
    of the contents of the pointers inside the struct on GPU.

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format on GPU.

    Returns : Matrix in CSR format on CPU.

    @author : mishraiiit
*/

MatrixCSR * deepCopyMatrixCSRGPUtoCPU(const MatrixCSR * const gpu_matrix) {
    MatrixCSR * cpu_matrix = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    cudaMemcpy(cpu_matrix, gpu_matrix, sizeof(MatrixCSR), cudaMemcpyDeviceToHost);
    int * cpu_i = (int *) malloc(sizeof(int) * (cpu_matrix->rows + 1));
    int * cpu_j = (int *) malloc(sizeof(int) * (cpu_matrix->nnz));
    float * cpu_val = (float *) malloc(sizeof(float) * (cpu_matrix->nnz));

    cudaMemcpy(cpu_i, cpu_matrix->i,
        sizeof(int) * (cpu_matrix->rows + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_j, cpu_matrix->j,
        sizeof(int) * (cpu_matrix->nnz), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_val, cpu_matrix->val,
        sizeof(float) * (cpu_matrix->nnz), cudaMemcpyDeviceToHost);

    cpu_matrix->i = cpu_i;
    cpu_matrix->j = cpu_j;
    cpu_matrix->val = cpu_val;

    return cpu_matrix;
}


/*
    Description : Copies a matrix on CPU to GPU. It will not copy the contents
    of the underlying pointers, will just copy the pointers as is (i.e the
    address, not the values).

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format on CPU.

    Returns : Matrix in CSR format on GPU.

    @author : mishraiiit
*/

MatrixCSR * shallowCopyMatrixCSRCPUtoGPU(const MatrixCSR * const my_cpu) {
    MatrixCSR * cpu_matrix = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    memcpy(cpu_matrix, my_cpu, sizeof(MatrixCSR));
    MatrixCSR * gpu_matrix;
    cudaMalloc(&gpu_matrix, sizeof(MatrixCSR));
    cudaMemcpy(gpu_matrix, cpu_matrix, sizeof(MatrixCSR), cudaMemcpyHostToDevice);
    free(cpu_matrix);
    return gpu_matrix;
}


/*
    Description : Copies a matrix on GPU to GPU. It will also create a copy
    of the contents of the pointers inside the struct on GPU.

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format on GPU.

    Returns : Matrix in CSR format on GPU.

    @author : mishraiiit
*/

MatrixCSR * deepCopyMatrixCSRGPUtoGPU(const MatrixCSR * const gpu_matrix) {
    MatrixCSR * cpu_matrix = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    cudaMemcpy(cpu_matrix, gpu_matrix, sizeof(MatrixCSR), cudaMemcpyDeviceToHost);

    int * gpu_i;
    int * gpu_j;
    float * gpu_val;

    cudaMalloc(&gpu_i, sizeof(int) * (cpu_matrix->rows + 1));
    cudaMalloc(&gpu_j, sizeof(int) * (cpu_matrix->nnz));
    cudaMalloc(&gpu_val, sizeof(float) * (cpu_matrix->nnz));

    cudaMemcpy(gpu_i, cpu_matrix->i,
        sizeof(int) * (cpu_matrix->rows + 1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_j, cpu_matrix->j,
        sizeof(int) * (cpu_matrix->nnz), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_val, cpu_matrix->val,
        sizeof(float) * (cpu_matrix->nnz), cudaMemcpyDeviceToDevice);

    cpu_matrix->i = gpu_i;
    cpu_matrix->j = gpu_j;
    cpu_matrix->val = gpu_val;

    MatrixCSR * gpu_copy = shallowCopyMatrixCSRCPUtoGPU(cpu_matrix);
    free(cpu_matrix);

    return gpu_copy;
}


/*
    Description : Copies a matrix on GPU to CPU. It will also create a copy
    of the contents of the pointers inside the struct on GPU.

    Parameters : 
        MatrixCSC * matrix : Matrix in CSC format on GPU.

    Returns : Matrix in CSC format on CPU.

    @author : mishraiiit
*/

MatrixCSC * deepCopyMatrixCSCRGPUtoCPU(const MatrixCSC * const gpu_matrix) {
    MatrixCSC * cpu_matrix = (MatrixCSC *) malloc(sizeof(MatrixCSC));
    cudaMemcpy(cpu_matrix, gpu_matrix, sizeof(MatrixCSC), cudaMemcpyDeviceToHost);
    int * cpu_i = (int *) malloc(sizeof(int) * (cpu_matrix->nnz));
    int * cpu_j = (int *) malloc(sizeof(int) * (cpu_matrix->cols + 1));
    float * cpu_val = (float *) malloc(sizeof(float) * (cpu_matrix->nnz));

    cudaMemcpy(cpu_i, cpu_matrix->i,
        sizeof(int) * (cpu_matrix->rows + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_j, cpu_matrix->j,
        sizeof(int) * (cpu_matrix->nnz), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_val, cpu_matrix->val,
        sizeof(float) * (cpu_matrix->nnz), cudaMemcpyDeviceToHost);

    cpu_matrix->i = cpu_i;
    cpu_matrix->j = cpu_j;
    cpu_matrix->val = cpu_val;

    return cpu_matrix;
}


/*
    Description : Copies a matrix on GPU to CPU. It will not copy the contents
    of the underlying pointers, will just copy the pointers as is (i.e the
    address, not the values).

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format on GPU.

    Returns : Matrix in CSR format on CPU.

    @author : mishraiiit
*/

MatrixCSR * shallowCopyMatrixCSRGPUtoCPU(const MatrixCSR * const gpu_matrix) {
    MatrixCSR * cpu_matrix = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    cudaMemcpy(cpu_matrix, gpu_matrix, sizeof(MatrixCSR), cudaMemcpyDeviceToHost);
    return cpu_matrix;
}


/*
    Description : Copies a matrix on CPU to GPU. It will also create a copy
    of the contents of the pointers inside the struct on CPU.

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format on CPU.

    Returns : Matrix in CSR format on GPU.

    @author : mishraiiit
*/

MatrixCSR * deepCopyMatrixCSRCPUtoGPU(const MatrixCSR * const my_cpu) {
    MatrixCSR * cpu_matrix = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    memcpy(cpu_matrix, my_cpu, sizeof(MatrixCSR));
    MatrixCSR * gpu_matrix;
    cudaMalloc(&gpu_matrix, sizeof(MatrixCSR));

    int * gpu_i;
    int * gpu_j;
    float * gpu_val;

    cudaMalloc(&gpu_i, sizeof(int) * (cpu_matrix->rows + 1));
    cudaMalloc(&gpu_j, sizeof(int) * (cpu_matrix->nnz));
    cudaMalloc(&gpu_val, sizeof(float) * (cpu_matrix->nnz));

    cudaMemcpy(gpu_i, cpu_matrix->i,
    	sizeof(int) * (cpu_matrix->rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_j, cpu_matrix->j,
    	sizeof(int) * (cpu_matrix->nnz), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_val, cpu_matrix->val,
    	sizeof(float) * (cpu_matrix->nnz), cudaMemcpyHostToDevice);

    cpu_matrix->i = gpu_i;
    cpu_matrix->j = gpu_j;
    cpu_matrix->val = gpu_val;

    cudaMemcpy(gpu_matrix, cpu_matrix, sizeof(MatrixCSR), cudaMemcpyHostToDevice);
    free(cpu_matrix);
    return gpu_matrix;
}


/*
    Description : Copies a matrix on CPU to GPU. It will not copy the contents
    of the underlying pointers, will just copy the pointers as is (i.e the
    address, not the values).

    Parameters : 
        MatrixCSC * matrix : Matrix in CSC format on CPU.

    Returns : Matrix in CSC format on GPU.

    @author : mishraiiit
*/

MatrixCSC * shallowCopyMatrixCSCCPUtoGPU(const MatrixCSC * const my_cpu) {
    MatrixCSC * cpu_matrix = (MatrixCSC *) malloc(sizeof(MatrixCSC));
    memcpy(cpu_matrix, my_cpu, sizeof(MatrixCSC));
    MatrixCSC * gpu_matrix;
    cudaMalloc(&gpu_matrix, sizeof(MatrixCSC));
    cudaMemcpy(gpu_matrix, cpu_matrix, sizeof(MatrixCSC), cudaMemcpyHostToDevice);
    free(cpu_matrix);
    return gpu_matrix;
}

/*
    Description : It takes a matrix in CSR which is on GPU and returns it's
    transpose on GPU.

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format on GPU.
        cusparseHandle_t & handle : cudaSparse handle.

    Returns : The transpose of input matrix in CSR format on GPU.

    @author : mishraiiit
*/

MatrixCSR * transposeCSRGPU_cudaSparse(MatrixCSR * matrix_gpu, cusparseHandle_t & handle) {

    MatrixCSR * shallow_cpu = shallowCopyMatrixCSRGPUtoCPU(matrix_gpu);

    int * new_i;
    int * new_j;
    float * new_val;

    cudaMalloc(&new_i, sizeof(int) * (shallow_cpu->cols + 1));
    cudaMalloc(&new_j, sizeof(int) * (shallow_cpu->nnz));
    cudaMalloc(&new_val, sizeof(float) * (shallow_cpu->nnz));

    
    cusparseStatus_t status = cusparseScsr2csc(handle, shallow_cpu->rows,
        shallow_cpu->cols, shallow_cpu->nnz, shallow_cpu->val, shallow_cpu->i,
        shallow_cpu->j, new_val, new_j, new_i,
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

    swap_variables(shallow_cpu->rows, shallow_cpu->cols);

    assert(status == CUSPARSE_STATUS_SUCCESS);

    shallow_cpu->i = new_i;
    shallow_cpu->j = new_j;
    shallow_cpu->val = new_val;

    MatrixCSR * to_return = shallowCopyMatrixCSRCPUtoGPU(shallow_cpu);
    free(shallow_cpu);
    return to_return;
}


/*
    Description : It takes a matrix in CSR which is on GPU and returns it's
    corresponding CSC format on GPU.

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format on GPU.
        cusparseHandle_t & handle : cudaSparse handle.

    Returns : The CSC format of the given matrix on GPU.

    @author : mishraiiit
*/

MatrixCSC * convertCSRGPU_cudaSparse(MatrixCSR * matrix_gpu, cusparseHandle_t & handle) {

    MatrixCSR * shallow_cpu = shallowCopyMatrixCSRGPUtoCPU(matrix_gpu);
    MatrixCSC * shallow_cpu_csc = (MatrixCSC *) malloc(sizeof(MatrixCSC));

    int * new_i;
    int * new_j;
    float * new_val;

    cudaMalloc(&new_i, sizeof(int) * (shallow_cpu->cols + 1));
    cudaMalloc(&new_j, sizeof(int) * (shallow_cpu->nnz));
    cudaMalloc(&new_val, sizeof(float) * (shallow_cpu->nnz));

    
    cusparseStatus_t status = cusparseScsr2csc(handle, shallow_cpu->rows,
        shallow_cpu->cols, shallow_cpu->nnz, shallow_cpu->val, shallow_cpu->i,
        shallow_cpu->j, new_val, new_j, new_i,
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

    assert(status == CUSPARSE_STATUS_SUCCESS);

    shallow_cpu_csc->rows = shallow_cpu->cols;
    shallow_cpu_csc->cols = shallow_cpu->rows;
    shallow_cpu_csc->nnz = shallow_cpu->nnz;
    shallow_cpu_csc->j = new_i;
    shallow_cpu_csc->i = new_j;
    shallow_cpu_csc->val = new_val;

    MatrixCSC * to_return = shallowCopyMatrixCSCCPUtoGPU(shallow_cpu_csc);
    free(shallow_cpu_csc);
    return to_return;
}


/*
    Description : It takes two sparse matrices on GPU and multiplies them.

    Parameters : 
        MatrixCSR * A : Matrix in CSR format on GPU.
        MatrixCSR * B : Matrix in CSR format on GPU.
        cusparseHandle_t & handle : cudaSparse handle.

    Returns : Sparse matrix C = A * B

    @author : mishraiiit
*/

MatrixCSR * spmatrixmult_cudaSparse(MatrixCSR * a, MatrixCSR * b, cusparseHandle_t & handle ) {

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);

    MatrixCSR * shallow_a = shallowCopyMatrixCSRGPUtoCPU(a);
    MatrixCSR * shallow_b = shallowCopyMatrixCSRGPUtoCPU(b);

    int nnzC;
    // nnzTotalDevHostPtr points to host memory
    int * nnzTotalDevHostPtr = &nnzC;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    int * csrRowPtrC;
    cudaMalloc(&csrRowPtrC, sizeof(int)*(shallow_a->rows + 1));

    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, shallow_a->rows,
        shallow_b->cols, shallow_a->cols, 
        descr, shallow_a->nnz, 
        shallow_a->i, shallow_a->j,
        descr, shallow_b->nnz, shallow_b->i, shallow_b->j,
        descr, csrRowPtrC, nnzTotalDevHostPtr );

    assert(nnzTotalDevHostPtr != NULL);
    assert(nnzC == *nnzTotalDevHostPtr);


    int * csrColIndC;
    float * csrValC;

    cudaMalloc(&csrColIndC, sizeof(int) * nnzC);
    cudaMalloc(&csrValC, sizeof(float) * nnzC);

    cusparseStatus_t status =  cusparseScsrgemm(handle,
         CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
         shallow_a->rows, shallow_b->cols, shallow_a->cols,
         descr, shallow_a->nnz,
         shallow_a->val, shallow_a->i, shallow_a->j,
         descr, shallow_b->nnz,
         shallow_b->val, shallow_b->i, shallow_b->j,
         descr,
         csrValC, csrRowPtrC, csrColIndC);
    
    assert(status == CUSPARSE_STATUS_SUCCESS);

    MatrixCSR * shallow_c = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    shallow_c->rows = shallow_a->rows;
    shallow_c->cols = shallow_b->cols;
    shallow_c->nnz = nnzC;
    shallow_c->i = csrRowPtrC;
    shallow_c->j = csrColIndC;
    shallow_c->val = csrValC;
    MatrixCSR * c = shallowCopyMatrixCSRCPUtoGPU(shallow_c);
    free(shallow_c);
    return c;
}

#endif