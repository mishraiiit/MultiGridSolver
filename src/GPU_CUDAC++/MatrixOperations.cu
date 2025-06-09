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

MatrixCSC * deepCopyMatrixCSCRGPUtoCPU(const MatrixCSC * const gpu_matrix) {
    MatrixCSC * cpu_matrix = (MatrixCSC *) malloc(sizeof(MatrixCSC));
    assert(cudaMemcpy(cpu_matrix, gpu_matrix,
        sizeof(MatrixCSC), cudaMemcpyDeviceToHost) == cudaSuccess);
    int * cpu_i = (int *) malloc(sizeof(int) * (cpu_matrix->nnz));
    int * cpu_j = (int *) malloc(sizeof(int) * (cpu_matrix->cols + 1));
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


    int rows = matrix_gpu->rows;
    int cols = matrix_gpu->cols;
    int nnz = matrix_gpu->nnz;

    int * new_i;
    int * new_j;
    float * new_val;

    assert(cudaMalloc(&new_i, sizeof(int) * (cols + 1)) == cudaSuccess);
    assert(cudaMalloc(&new_j, sizeof(int) * nnz) == cudaSuccess);
    assert(cudaMalloc(&new_val, sizeof(float) * nnz) == cudaSuccess);

    size_t bufferSize = 0;
    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(
        handle,
        rows,
        cols,
        nnz,
        matrix_gpu->val,
        matrix_gpu->i,
        matrix_gpu->j,
        new_val,
        new_i,
        new_j,
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &bufferSize
    );
    assert(status == CUSPARSE_STATUS_SUCCESS);

    void * pBuffer = nullptr;
    assert(cudaMalloc(&pBuffer, bufferSize) == cudaSuccess);

    // Perform the CSR to CSC conversion
    status = cusparseCsr2cscEx2(
        handle,
        rows,
        cols,
        nnz,
        matrix_gpu->val,
        matrix_gpu->i,
        matrix_gpu->j,
        new_val,
        new_i,
        new_j,
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        pBuffer
    );
    assert(status == CUSPARSE_STATUS_SUCCESS);

    cudaFree(pBuffer);

    MatrixCSR * transposed_matrix_gpu = new MatrixCSR();
    transposed_matrix_gpu->rows = cols;
    transposed_matrix_gpu->cols = rows;
    transposed_matrix_gpu->nnz = nnz;
    transposed_matrix_gpu->i = new_i;
    transposed_matrix_gpu->j = new_j;
    transposed_matrix_gpu->val = new_val;

    return transposed_matrix_gpu;
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

MatrixCSC* convertCSRGPU_cudaSparse(MatrixCSR* matrix_gpu, cusparseHandle_t& handle) {

    MatrixCSR* shallow_cpu = shallowCopyMatrixCSRGPUtoCPU(matrix_gpu);
    MatrixCSC* shallow_cpu_csc = (MatrixCSC*)malloc(sizeof(MatrixCSC));

    int*   d_cscColPtr;
    int*   d_cscRowInd;
    float* d_cscVal;

    assert(cudaMalloc(&d_cscColPtr, sizeof(int)   * (shallow_cpu->cols + 1)) == cudaSuccess);
    assert(cudaMalloc(&d_cscRowInd, sizeof(int)   * (shallow_cpu->nnz))      == cudaSuccess);
    assert(cudaMalloc(&d_cscVal,   sizeof(float) * (shallow_cpu->nnz))      == cudaSuccess);

    size_t bufferSize = 0;
    void*  dBuffer    = NULL;
    
    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(
        handle,
        shallow_cpu->rows,
        shallow_cpu->cols,
        shallow_cpu->nnz,
        shallow_cpu->val,
        shallow_cpu->i,
        shallow_cpu->j,
        d_cscVal,
        d_cscColPtr,
        d_cscRowInd,
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &bufferSize);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    if (bufferSize > 0) {
        assert(cudaMalloc(&dBuffer, bufferSize) == cudaSuccess);
    }

    status = cusparseCsr2cscEx2(
        handle,
        shallow_cpu->rows,
        shallow_cpu->cols,
        shallow_cpu->nnz,
        shallow_cpu->val,
        shallow_cpu->i,
        shallow_cpu->j,
        d_cscVal,
        d_cscColPtr,
        d_cscRowInd,
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        dBuffer);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    if (dBuffer) {
        cudaFree(dBuffer);
    }

    shallow_cpu_csc->rows = shallow_cpu->cols;
    shallow_cpu_csc->cols = shallow_cpu->rows;
    shallow_cpu_csc->nnz  = shallow_cpu->nnz;
    shallow_cpu_csc->j    = d_cscColPtr;
    shallow_cpu_csc->i    = d_cscRowInd;
    shallow_cpu_csc->val  = d_cscVal;

    free(shallow_cpu);

    MatrixCSC* to_return = shallowCopyMatrixCSCCPUtoGPU(shallow_cpu_csc);
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

MatrixCSR* spmatrixmult_cudaSparse(MatrixCSR* a, MatrixCSR* b, cusparseHandle_t& handle) {

    // Create host-side shallow copies to access matrix metadata (dims, nnz)
    // and GPU data pointers. The actual matrix data remains on the GPU.
    MatrixCSR* shallow_a = shallowCopyMatrixCSRGPUtoCPU(a);
    MatrixCSR* shallow_b = shallowCopyMatrixCSRGPUtoCPU(b);

    // SpGEMM computes C = alpha * op(A) * op(B) + beta * D.
    // We want C = A * B, so alpha = 1.0, beta = 0.0, and D can be null.
    float alpha = 1.0f;
    float beta  = 0.0f;


    cusparseSpMatDescr_t matA, matB, matC;
    
    // Create descriptor for matrix A
    assert(cusparseCreateCsr(
        &matA, shallow_a->rows, shallow_a->cols, shallow_a->nnz,
        shallow_a->i,   // CSR row offsets
        shallow_a->j,   // CSR column indices
        shallow_a->val, // CSR values
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) == CUSPARSE_STATUS_SUCCESS);

    // Create descriptor for matrix B
    assert(cusparseCreateCsr(
        &matB, shallow_b->rows, shallow_b->cols, shallow_b->nnz,
        shallow_b->i,   // CSR row offsets
        shallow_b->j,   // CSR column indices
        shallow_b->val, // CSR values
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) == CUSPARSE_STATUS_SUCCESS);

    // Create descriptor for the result matrix C
    // Its dimensions are (A's rows, B's cols). Pointers are NULL initially.
    assert(cusparseCreateCsr(
        &matC, shallow_a->rows, shallow_b->cols, 0, // nnz is unknown, set to 0
        NULL, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) == CUSPARSE_STATUS_SUCCESS);

    // Determine buffer sizes and allocate workspace
    size_t bufferSize1 = 0;
    void*  dBuffer1    = NULL;
    assert(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, &bufferSize1, NULL) == CUSPARSE_STATUS_SUCCESS);
    assert(cudaMalloc(&dBuffer1, bufferSize1) == cudaSuccess);

    // Compute the NNZ of C and its row pointers
    assert(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, &bufferSize1, dBuffer1) == CUSPARSE_STATUS_SUCCESS);

    // Get the size and row pointers of C from its descriptor
    int64_t c_rows, c_cols, c_nnz;
    int* d_csrRowPtrC;
    assert(cusparseSpMatGetSize(matC, &c_rows, &c_cols, &c_nnz) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseCsrGet(matC, NULL, NULL, NULL, (void**)&d_csrRowPtrC, NULL, NULL, NULL, NULL, NULL) == CUSPARSE_STATUS_SUCCESS);
    
    // Allocate memory for C's column indices and values
    int*   d_csrColIndC;
    float* d_csrValC;
    assert(cudaMalloc((void**)&d_csrColIndC, c_nnz * sizeof(int))   == cudaSuccess);
    assert(cudaMalloc((void**)&d_csrValC,   c_nnz * sizeof(float)) == cudaSuccess);
    
    // Update the C descriptor with the newly allocated pointers
    assert(cusparseCsrSetPointers(matC, d_csrRowPtrC, d_csrColIndC, d_csrValC) == CUSPARSE_STATUS_SUCCESS);

    // Re-run the computation phase to compute C's column indices and values
    size_t bufferSize2 = 0;
    void*  dBuffer2    = NULL;
    assert(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, &bufferSize2, NULL) == CUSPARSE_STATUS_SUCCESS);
    assert(cudaMalloc(&dBuffer2, bufferSize2) == cudaSuccess);

    assert(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, &bufferSize2, dBuffer2) == CUSPARSE_STATUS_SUCCESS);

    // Create the final output matrix struct
    MatrixCSR* shallow_c = (MatrixCSR*) malloc(sizeof(MatrixCSR));
    shallow_c->rows = c_rows;
    shallow_c->cols = c_cols;
    shallow_c->nnz  = c_nnz;
    shallow_c->i    = d_csrRowPtrC;
    shallow_c->j    = d_csrColIndC;
    shallow_c->val  = d_csrValC;

    MatrixCSR* c = shallowCopyMatrixCSRCPUtoGPU(shallow_c);

    // Clean up resources
    free(shallow_a);
    free(shallow_b);
    free(shallow_c);
    
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);

    return c;
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
