/*
    Description : This file contains functions related to matrix operations.

    @author : mishraiiit
*/

#ifndef MATRIX_OPERATIONS
#define MATRIX_OPERATIONS
#include "MatrixIO.cu"
#include "GPUDebug.cu"
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>

// Error checking macros
#define CHECK_CUDA(call) do { \
    cudaError_t cuda_check_error = call; \
    if (cuda_check_error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(cuda_check_error)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t cusparse_check_status = call; \
    if (cusparse_check_status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

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
    std::cout << "rows: " << matrix->rows << " cols: " << matrix->cols << " nnz: " << matrix->nnz << std::endl;
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
    assert(cudaMemcpy(cpu_matrix, gpu_matrix,
        sizeof(MatrixCSR), cudaMemcpyDeviceToHost) == cudaSuccess);
    int * cpu_i = (int *) malloc(sizeof(int) * (cpu_matrix->rows + 1));
    int * cpu_j = (int *) malloc(sizeof(int) * (cpu_matrix->nnz));
    float * cpu_val = (float *) malloc(sizeof(float) * (cpu_matrix->nnz));

    assert(cudaMemcpy(cpu_i, cpu_matrix->i,
        sizeof(int) * (cpu_matrix->rows + 1),
        cudaMemcpyDeviceToHost) == cudaSuccess);

    assert(cudaMemcpy(cpu_j, cpu_matrix->j,
        sizeof(int) * (cpu_matrix->nnz),
        cudaMemcpyDeviceToHost) == cudaSuccess);

    assert(cudaMemcpy(cpu_val, cpu_matrix->val,
        sizeof(float) * (cpu_matrix->nnz),
        cudaMemcpyDeviceToHost) == cudaSuccess);

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
    assert(cudaMalloc(&gpu_matrix, sizeof(MatrixCSR)) == cudaSuccess);
    assert(cudaMemcpy(gpu_matrix, cpu_matrix,
        sizeof(MatrixCSR), cudaMemcpyHostToDevice) == cudaSuccess);
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
    assert(cudaMemcpy(cpu_matrix, gpu_matrix,
        sizeof(MatrixCSR), cudaMemcpyDeviceToHost) == cudaSuccess);

    int * gpu_i;
    int * gpu_j;
    float * gpu_val;

    assert(cudaMalloc(&gpu_i,
        sizeof(int) * (cpu_matrix->rows + 1)) == cudaSuccess);
    assert(cudaMalloc(&gpu_j,
        sizeof(int) * (cpu_matrix->nnz)) == cudaSuccess);
    assert(cudaMalloc(&gpu_val,
        sizeof(float) * (cpu_matrix->nnz)) == cudaSuccess);

    assert(cudaMemcpy(gpu_i, cpu_matrix->i, 
        sizeof(int) * (cpu_matrix->rows + 1),
        cudaMemcpyDeviceToDevice) == cudaSuccess);
    assert(cudaMemcpy(gpu_j, cpu_matrix->j,
        sizeof(int) * (cpu_matrix->nnz),
        cudaMemcpyDeviceToDevice) == cudaSuccess);
    assert(cudaMemcpy(gpu_val, cpu_matrix->val,
        sizeof(float) * (cpu_matrix->nnz),
        cudaMemcpyDeviceToDevice) == cudaSuccess);

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

MatrixCSC * deepCopyMatrixCSCGPUtoCPU(const MatrixCSC * const gpu_matrix) {
    MatrixCSC * cpu_matrix = (MatrixCSC *) malloc(sizeof(MatrixCSC));
    assert(cudaMemcpy(cpu_matrix, gpu_matrix,
        sizeof(MatrixCSC), cudaMemcpyDeviceToHost) == cudaSuccess);
    int * cpu_i = (int *) malloc(sizeof(int) * (cpu_matrix->nnz));
    int * cpu_j = (int *) malloc(sizeof(int) * (cpu_matrix->cols + 1));
    float * cpu_val = (float *) malloc(sizeof(float) * (cpu_matrix->nnz));

    assert(cudaMemcpy(cpu_i, cpu_matrix->i,
        sizeof(int) * (cpu_matrix->nnz),
        cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(cpu_j, cpu_matrix->j,
        sizeof(int) * (cpu_matrix->cols + 1),
        cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(cpu_val, cpu_matrix->val,
        sizeof(float) * (cpu_matrix->nnz),
        cudaMemcpyDeviceToHost) == cudaSuccess);

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
    assert(cudaMemcpy(cpu_matrix, gpu_matrix,
        sizeof(MatrixCSR), cudaMemcpyDeviceToHost) == cudaSuccess);
    return cpu_matrix;
}


/*
    Description : Copies a matrix on GPU to CPU. It will not copy the contents
    of the underlying pointers, will just copy the pointers as is (i.e the
    address, not the values).

    Parameters : 
        MatrixCSC * matrix : Matrix in CSC format on GPU.

    Returns : Matrix in CSC format on CPU.

    @author : mishraiiit
*/

MatrixCSC * shallowCopyMatrixCSCGPUtoCPU(const MatrixCSC * const gpu_matrix) {
    MatrixCSC * cpu_matrix = (MatrixCSC *) malloc(sizeof(MatrixCSC));
    assert(cudaMemcpy(cpu_matrix, gpu_matrix,
        sizeof(MatrixCSC), cudaMemcpyDeviceToHost) == cudaSuccess);
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
    assert(cudaMalloc(&gpu_matrix, sizeof(MatrixCSR)) == cudaSuccess);

    int * gpu_i;
    int * gpu_j;
    float * gpu_val;

    assert(cudaMalloc(&gpu_i,
        sizeof(int) * (cpu_matrix->rows + 1)) == cudaSuccess);
    assert(cudaMalloc(&gpu_j,
        sizeof(int) * (cpu_matrix->nnz)) == cudaSuccess);
    assert(cudaMalloc(&gpu_val,
        sizeof(float) * (cpu_matrix->nnz)) == cudaSuccess);

    assert(cudaMemcpy(gpu_i, cpu_matrix->i,
    	sizeof(int) * (cpu_matrix->rows + 1),
        cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(gpu_j, cpu_matrix->j,
    	sizeof(int) * (cpu_matrix->nnz),
        cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(gpu_val, cpu_matrix->val,
    	sizeof(float) * (cpu_matrix->nnz),
         cudaMemcpyHostToDevice) == cudaSuccess);

    cpu_matrix->i = gpu_i;
    cpu_matrix->j = gpu_j;
    cpu_matrix->val = gpu_val;

    assert(cudaMemcpy(gpu_matrix, cpu_matrix,
        sizeof(MatrixCSR), cudaMemcpyHostToDevice) == cudaSuccess);
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
    assert(cudaMalloc(&gpu_matrix, sizeof(MatrixCSC)) == cudaSuccess);
    assert(cudaMemcpy(gpu_matrix, cpu_matrix,
        sizeof(MatrixCSC), cudaMemcpyHostToDevice) == cudaSuccess);
    free(cpu_matrix);
    return gpu_matrix;
}

/*
    Description : It takes a matrix in CSR format which is on the GPU and returns its
    transpose on the GPU.

    Parameters :
        MatrixCSR * matrix_gpu : Matrix in CSR format on the GPU.
        cusparseHandle_t & handle : cuSPARSE handle.

    Returns : The transpose of the input matrix in CSR format on the GPU.

    @author : mishraiiit
*/
MatrixCSR * transposeCSRGPU_cudaSparse(MatrixCSR * matrix_gpu, cusparseHandle_t & handle) {

    MatrixCSR * shallow_cpu = shallowCopyMatrixCSRGPUtoCPU(matrix_gpu);

    int * new_i;
    int * new_j;
    float * new_val;

    assert(cudaMalloc(&new_i,
        sizeof(int) * (shallow_cpu->cols + 1)) == cudaSuccess);
    assert(cudaMalloc(&new_j,
        sizeof(int) * (shallow_cpu->nnz)) == cudaSuccess);
    assert(cudaMalloc(&new_val,
        sizeof(float) * (shallow_cpu->nnz)) == cudaSuccess);


    void* d_buffer = nullptr;
    size_t bufferSize = 0;

    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(
        handle,
        shallow_cpu->rows, // rows
        shallow_cpu->cols, // cols
        shallow_cpu->nnz,  // nnz
        shallow_cpu->val,  // nnz
        shallow_cpu->i,    // rows+1
        shallow_cpu->j,    // nnz
        new_val,           // nnz
        new_i,             // cols+1
        new_j,             // nnz
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &bufferSize);
    assert(status == CUSPARSE_STATUS_SUCCESS);


    assert(cudaMalloc(&d_buffer, bufferSize) == cudaSuccess);

    status = cusparseCsr2cscEx2(
        handle,
        shallow_cpu->rows, // rows
        shallow_cpu->cols, // cols
        shallow_cpu->nnz,  // nnz
        shallow_cpu->val,  // nnz
        shallow_cpu->i,    // rows+1
        shallow_cpu->j,    // nnz
        new_val,           // nnz
        new_i,             // cols+1
        new_j,             // nnz
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        d_buffer);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    swap_variables(shallow_cpu->rows, shallow_cpu->cols);


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

    assert(cudaMalloc(&new_i,
        sizeof(int) * (shallow_cpu->nnz)) == cudaSuccess);
    assert(cudaMalloc(&new_j,
        sizeof(int) * (shallow_cpu->cols + 1)) == cudaSuccess);
    assert(cudaMalloc(&new_val,
        sizeof(float) * (shallow_cpu->nnz)) == cudaSuccess);

    
    void* d_buffer = nullptr;
    size_t bufferSize = 0;

    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(
        handle,
        shallow_cpu->rows, // rows
        shallow_cpu->cols, // cols
        shallow_cpu->nnz,  // nnz
        shallow_cpu->val,  // nnz
        shallow_cpu->i,    // rows+1
        shallow_cpu->j,    // nnz
        new_val,           // nnz
        new_j,             // cols+1
        new_i,             // nnz
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &bufferSize);
    assert(status == CUSPARSE_STATUS_SUCCESS);


    assert(cudaMalloc(&d_buffer, bufferSize) == cudaSuccess);

    status = cusparseCsr2cscEx2(
        handle,
        shallow_cpu->rows, // rows
        shallow_cpu->cols, // cols
        shallow_cpu->nnz,  // nnz
        shallow_cpu->val,  // nnz
        shallow_cpu->i,    // rows+1
        shallow_cpu->j,    // nnz
        new_val,           // nnz
        new_j,             // cols+1
        new_i,             // nnz
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        d_buffer);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    cudaFree(d_buffer);

    shallow_cpu_csc->rows = shallow_cpu->rows;
    shallow_cpu_csc->cols = shallow_cpu->cols;
    shallow_cpu_csc->nnz = shallow_cpu->nnz;
    shallow_cpu_csc->i = new_i;
    shallow_cpu_csc->j = new_j;
    shallow_cpu_csc->val = new_val;

    MatrixCSC * to_return = shallowCopyMatrixCSCCPUtoGPU(shallow_cpu_csc);
    free(shallow_cpu_csc);
    return to_return;
}


/*
    Description : It takes a matrix in CSC which is on GPU and returns it's
    corresponding CSR format on GPU.

    Parameters : 
        MatrixCSC * matrix : Matrix in CSC format on GPU.
        cusparseHandle_t & handle : cudaSparse handle.

    Returns : The CSR format of the given matrix on GPU.

    @author : mishraiiit
*/

MatrixCSR * convertCSCGPU_cudaSparse(MatrixCSC * matrix_gpu, cusparseHandle_t & handle) {

    MatrixCSC * shallow_cpu = shallowCopyMatrixCSCGPUtoCPU(matrix_gpu);
    MatrixCSR * shallow_cpu_csr = (MatrixCSR *) malloc(sizeof(MatrixCSR));

    int * new_i;
    int * new_j;
    float * new_val;

    assert(cudaMalloc(&new_i,
        sizeof(int) * (shallow_cpu->rows + 1)) == cudaSuccess);
    assert(cudaMalloc(&new_j,
        sizeof(int) * (shallow_cpu->nnz)) == cudaSuccess);
    assert(cudaMalloc(&new_val,
        sizeof(float) * (shallow_cpu->nnz)) == cudaSuccess);

    size_t bufferSize = 0;
    void* d_buffer = nullptr;

    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(
        handle,
        shallow_cpu->rows, // rows
        shallow_cpu->cols, // cols
        shallow_cpu->nnz,  // nnz
        shallow_cpu->val,  // nnz
        shallow_cpu->j,    // cols+1
        shallow_cpu->i,    // nnz
        new_val,           // nnz
        new_i,             // rows+1
        new_j,             // nnz
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &bufferSize);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    assert(cudaMalloc(&d_buffer, bufferSize) == cudaSuccess);

    status = cusparseCsr2cscEx2(
        handle,
        shallow_cpu->rows, // rows
        shallow_cpu->cols, // cols
        shallow_cpu->nnz,  // nnz
        shallow_cpu->val,  // nnz
        shallow_cpu->j,    // cols+1
        shallow_cpu->i,    // nnz
        new_val,           // nnz
        new_i,             // rows+1
        new_j,             // nnz
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        d_buffer);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    cudaFree(d_buffer);

    shallow_cpu_csr->rows = shallow_cpu->rows;
    shallow_cpu_csr->cols = shallow_cpu->cols;
    shallow_cpu_csr->nnz = shallow_cpu->nnz;
    shallow_cpu_csr->i = new_i;
    shallow_cpu_csr->j = new_j;
    shallow_cpu_csr->val = new_val;

    MatrixCSR * to_return = shallowCopyMatrixCSRCPUtoGPU(shallow_cpu_csr);
    free(shallow_cpu_csr);
    return to_return;
}

/*
    Description: It takes two sparse matrices on GPU and multiplies them using the modern cuSPARSE SpGEMM API.

    Parameters:
        MatrixCSR * A: Matrix in CSR format on GPU.
        MatrixCSR * B: Matrix in CSR format on GPU.
        cusparseHandle_t& handle: cuSPARSE handle.

    Returns: Sparse matrix C = A * B on the GPU.

    @author: mishraiiit (Corrected by Gemini)
*/
MatrixCSR* spmatrixmult_cudaSparse(MatrixCSR* a, MatrixCSR* b, cusparseHandle_t& handle) {
    MatrixCSR* shallow_a = shallowCopyMatrixCSRGPUtoCPU(a);
    MatrixCSR* shallow_b = shallowCopyMatrixCSRGPUtoCPU(b);

    float alpha = 1.0f;
    float beta  = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;

    cusparseSpMatDescr_t matA, matB, matC;
    
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, shallow_a->rows, shallow_a->cols, shallow_a->nnz,
        shallow_a->i,
        shallow_a->j,
        shallow_a->val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matB, shallow_b->rows, shallow_b->cols, shallow_b->nnz,
        shallow_b->i,
        shallow_b->j, 
        shallow_b->val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    int* d_csrRowPtrC;
    CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtrC, (shallow_a->rows + 1) * sizeof(int)));

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matC, shallow_a->rows, shallow_b->cols, 0,
        d_csrRowPtrC, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    void*  dBuffer1    = NULL;
    size_t bufferSize1 = 0;
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        handle, opA, opB, &alpha, matA, matB, &beta, matC,
        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL));
    CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        handle, opA, opB, &alpha, matA, matB, &beta, matC,
        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));


    void*  dBuffer2    = NULL;
    size_t bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseSpGEMM_compute(
        handle, opA, opB, &alpha, matA, matB, &beta, matC,
        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL));
    CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2));

    CHECK_CUSPARSE(cusparseSpGEMM_compute(
        handle, opA, opB, &alpha, matA, matB, &beta, matC,
        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));

    int64_t c_rows, c_cols, c_nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &c_rows, &c_cols, &c_nnz));
    
    int*   d_csrColIndC;
    float* d_csrValC;
    CHECK_CUDA(cudaMalloc((void**)&d_csrColIndC, c_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_csrValC,   c_nnz * sizeof(float)));
    
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, d_csrRowPtrC, d_csrColIndC, d_csrValC));

    CHECK_CUSPARSE(cusparseSpGEMM_copy(
        handle, opA, opB, &alpha, matA, matB, &beta, matC,
        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));


    MatrixCSR* c_device;
    CHECK_CUDA(cudaMalloc((void**)&c_device, sizeof(MatrixCSR)));

    MatrixCSR c_host;
    c_host.rows = c_rows;
    c_host.cols = c_cols;
    c_host.nnz  = c_nnz;
    c_host.i    = d_csrRowPtrC;
    c_host.j    = d_csrColIndC;
    c_host.val  = d_csrValC;

    CHECK_CUDA(cudaMemcpy(c_device, &c_host, sizeof(MatrixCSR), cudaMemcpyHostToDevice));

    free(shallow_a);
    free(shallow_b);
    
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);

    return c_device;
}

/*
    Description : Frees a sparse CSR Matrix on GPU.

    Parameters : 
        MatrixCSR * matrix : Matrix in CSR format on GPU.

    @author : mishraiiit
*/

void freeMatrixCSRGPU(MatrixCSR * matrix) {
    MatrixCSR * shallow_cpu = shallowCopyMatrixCSRGPUtoCPU(matrix);
    assert(cudaFree(shallow_cpu->i) == cudaSuccess);
    assert(cudaFree(shallow_cpu->j) == cudaSuccess);
    assert(cudaFree(shallow_cpu->val) == cudaSuccess);
    assert(cudaFree(matrix) == cudaSuccess);
    free(shallow_cpu);
}


/*
    Description : Frees a sparse CSC Matrix on GPU.

    Parameters : 
        MatrixCSC * matrix : Matrix in CSC format on GPU.

    @author : mishraiiit
*/

void freeMatrixCSCGPU(MatrixCSC * matrix) {
    MatrixCSC * shallow_cpu = shallowCopyMatrixCSCGPUtoCPU(matrix);
    assert(cudaFree(shallow_cpu->i) == cudaSuccess);
    assert(cudaFree(shallow_cpu->j) == cudaSuccess);
    assert(cudaFree(shallow_cpu->val) == cudaSuccess);
    assert(cudaFree(matrix) == cudaSuccess);
    free(shallow_cpu);
}

#endif
