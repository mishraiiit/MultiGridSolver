/*
    Description : This file contains the functions to access an element in
    the matrix formats present in MatrixIo.cu file.

    @author : mishraiiit
*/

#ifndef MATRIX_IO
#define MATRIX_IO
#include <assert.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <fstream>

/*
    Description : Struct declaration for a matrix in CSR format. The matrix is
    assumed to be zero-indexed i.e rows start from 0 and not 1.

    Parameters : 
        int rows : The number of rows in the matrix.
        int cols : The number of cols in the matrix.
        int nnz : The number of non-zero entries in the matrix.
        int * i : rowPtr for the matrix. Size is (rows + 1).
        int * j : colInd for the matrix. Size is nnz.
        float * val : The values at the corresponding colInd. Size is nnz.

    @author : mishraiiit
*/

struct MatrixCSR {
    int rows, cols, nnz;
    int * i, * j;
    float * val;
};


/*
    Description : Struct declaration for a matrix in CSC format. The matrix is
    assumed to be zero-indexed i.e rows start from 0 and not 1.

    Parameters : 
        int rows : The number of rows in the matrix.
        int cols : The number of cols in the matrix.
        int nnz : The number of non-zero entries in the matrix.
        int * i : rowInd for the matrix. Size is nnz.
        int * j : colPtr for the matrix. Size is (cols + 1).
        float * val : The values at the corresponding rowInd. Size is nnz.

    @author : mishraiiit
*/

struct MatrixCSC {
    int rows, cols, nnz;
    int * i, * j;
    float * val;
};


std::string itoa(int number) {
    if (number == 0) {
        return "0";
    }
    std::string ret;
    while(number != 0) {
        ret = ((char)('0' + (number % 10))) + ret;
        number = number / 10;
    }
    return ret;
}

/*
    Description : Print newlines for seperating stuff.
*/

void printLines() {
    fprintf(stderr, "\n");
    fprintf(stderr, "---------------------------------------------------------------------------\n");
    fprintf(stderr, "\n");
}

void printInfo(const char * s, int level) {
    std::string to_print = "";

    to_print = "\033[1;32m[info]\033[0m";
    for(int i = 0; i < level; i++) {
        to_print = ' ' + to_print;
    }
    fprintf(stderr, "%s %s\n", to_print.c_str(), s);
}

void printInfo(std::string s, int level) {
    std::string to_print = "";

    to_print = "\033[1;32m[info]\033[0m";
    for(int i = 0; i < level; i++) {
        to_print = ' ' + to_print;
    }
    fprintf(stderr, "%s %s\n", to_print.c_str(), s.c_str());
}

/*
    Description : It reads the matrix in the file to Unified Memory 
    in CSR format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCSR * readMatrixUnifiedMemoryCSR(std::string filename) {
    std::ifstream fin(filename);
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;
    // Read the data

    MatrixCSR * matrix_csr;
    cudaMallocManaged(&matrix_csr, sizeof(MatrixCSR));

    matrix_csr->rows = M;
    matrix_csr->cols = N;
    matrix_csr->nnz = L;

    cudaMallocManaged(&matrix_csr->i, sizeof(int) * (matrix_csr->rows + 1));
    cudaMallocManaged(&matrix_csr->j, sizeof(int) * L);
    cudaMallocManaged(&matrix_csr->val, sizeof(float) * L);

    int filled = 0;

    auto start = std::chrono::system_clock::now();

    std::vector< std::vector <std::pair<int, float> > > matrix_data(M);
    for (int l = 0; l < L; l++) {
        int m, n;
        float data;
        fin >> m >> n >> data;
        matrix_data[m - 1].push_back({n - 1, data});
    }

    for(int i = 0; i < M; i++) {
        std::sort(matrix_data[i].begin(), matrix_data[i].end());
        matrix_csr->i[i] = filled;
        for(auto tp : matrix_data[i]) {
            matrix_csr->j[filled] = tp.first;
            matrix_csr->val[filled] = tp.second;
            filled++;
        }
    }

    matrix_csr->i[M] = filled;

    assert(filled == L);
    fin.close();
    printInfo("Read matrix from file: " + filename, 4);
    printInfo("Matrix size : " + itoa(M) + " x " + itoa(N) + ".", 4);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;

    return matrix_csr;
}

/*
    Description : It reads the matrix in the file to CPU Memory 
    in CSR format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCSR * readMatrixCPUMemoryCSR(std::string filename) {
    std::ifstream fin(filename);
     if (!fin.is_open()) {
        printf("Error: Could not open file %s\n", filename.c_str());
        exit(1);
    }
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;
    // Read the data

    MatrixCSR * matrix_csr;
    matrix_csr = (MatrixCSR *) malloc(sizeof(MatrixCSR));

    matrix_csr->rows = M;
    matrix_csr->cols = N;
    matrix_csr->nnz = L;

    matrix_csr->i = (int *) malloc(sizeof(int) * (matrix_csr->rows + 1));
    matrix_csr->j = (int *) malloc(sizeof(int) * L);
    matrix_csr->val = (float *) malloc(sizeof(float) * L);

    int filled = 0;

    auto start = std::chrono::system_clock::now();

    // Using a dynamically allocated array of vectors to avoid stack overflow for large M
    std::vector<std::pair<int, float>> *matrix_data_ptr = new std::vector<std::pair<int, float>>[M];

    for (int l = 0; l < L; l++) {
        int m, n;
        float data;
        fin >> m >> n >> data;
        if (m - 1 >= 0 && m - 1 < M) { // Bounds check
            matrix_data_ptr[m - 1].push_back({n - 1, data});
        } else {
            // Handle error: row index out of bounds
            fprintf(stderr, "Warning: Row index %d out of bounds for matrix of size %d in file %s\n", m, M, filename.c_str());
        }
    }

    for(int i = 0; i < M; i++) {
        std::sort(matrix_data_ptr[i].begin(), matrix_data_ptr[i].end());
        matrix_csr->i[i] = filled;
        for(auto tp : matrix_data_ptr[i]) {
            if (filled < L) { // Bounds check
                matrix_csr->j[filled] = tp.first;
                matrix_csr->val[filled] = tp.second;
                filled++;
            } else {
                 // Handle error: too many non-zero elements
                fprintf(stderr, "Warning: More non-zero elements found than specified L=%d in file %s\n", L, filename.c_str());
                break; 
            }
        }
         if (filled >= L && i < M -1) { // If L is filled but not all rows processed
            for (int k = i + 1; k < M; ++k) { // Ensure subsequent row_ptr are also filled
                if (!matrix_data_ptr[k].empty()){
                     fprintf(stderr, "Warning: Non-zero elements found after L limit in file %s\n", filename.c_str());
                     break;
                }
            }
            if (filled > L) { // Correct L if more elements were processed somehow (should ideally not happen with checks)
                fprintf(stderr, "Warning: Actual nnz %d exceeded reported L %d in file %s. Updating nnz.\n", filled, L, filename.c_str());
                matrix_csr->nnz = filled; // This line might be problematic if L was used for allocation sizes strictly
            }
            // break; // Exiting outer loop too
        }
    }
    
    delete[] matrix_data_ptr; // Free the dynamically allocated array of vectors


    matrix_csr->i[M] = filled;

    if (filled != L) {
        // This can happen if L was too small or too large compared to actual data.
        // Adjusting nnz if necessary, though allocations were based on original L.
        // This situation might indicate an issue with the .mtx file or initial L value.
        fprintf(stderr, "Warning: Final nnz count %d does not match L %d for file %s. Using actual count %d for matrix nnz.\n", filled, L, filename.c_str(), filled);
        matrix_csr->nnz = filled;
    }


    assert(filled <= L || matrix_csr->nnz == filled); // Ensure filled is not more than L, or nnz has been updated
    fin.close();
    printInfo("Read matrix from file: " + filename, 4);
    printInfo("Matrix size : " + itoa(M) + " x " + itoa(N) + ".", 4);
    printInfo("NNZ read : " + itoa(matrix_csr->nnz) + ".", 4);


    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;


    return matrix_csr;
}


/*
    Description : It reads the matrix in the file to GPU memory
    in CSR format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCSR * readMatrixGPUMemoryCSR(std::string filename) {

    MatrixCSR * matrix_csr_cpu = readMatrixCPUMemoryCSR(filename);
    if (matrix_csr_cpu == NULL) return NULL; // Propagate error if CPU read failed
    MatrixCSR * matrix_csr;

    int * device_i, * device_j;
    float * device_val;

    // Use matrix_csr_cpu->nnz which might have been adjusted if L was incorrect
    cudaError_t err;
    err = cudaMalloc(&device_i, sizeof(int) * (matrix_csr_cpu->rows + 1));
    if (err != cudaSuccess) { free(matrix_csr_cpu); return NULL;}
    err = cudaMalloc(&device_j, sizeof(int) * matrix_csr_cpu->nnz);
    if (err != cudaSuccess) { cudaFree(device_i); free(matrix_csr_cpu); return NULL;}
    err = cudaMalloc(&device_val, sizeof(float) * matrix_csr_cpu->nnz);
    if (err != cudaSuccess) { cudaFree(device_i); cudaFree(device_j); free(matrix_csr_cpu); return NULL;}


    err = cudaMemcpy(device_i, matrix_csr_cpu->i,
        sizeof(int) * (matrix_csr_cpu->rows + 1), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { /* handle or log error, cleanup */ cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); free(matrix_csr_cpu); return NULL;}
    
    err = cudaMemcpy(device_j, matrix_csr_cpu->j,
        sizeof(int) * matrix_csr_cpu->nnz, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { /* handle or log error, cleanup */ cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); free(matrix_csr_cpu); return NULL;}

    err = cudaMemcpy(device_val, matrix_csr_cpu->val,
        sizeof(float) * matrix_csr_cpu->nnz, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { /* handle or log error, cleanup */ cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); free(matrix_csr_cpu); return NULL;}


    // Free CPU arrays now that data is on GPU
    free(matrix_csr_cpu->i);
    free(matrix_csr_cpu->j);
    free(matrix_csr_cpu->val);

    // Update pointers in the CPU-side struct to point to GPU memory
    matrix_csr_cpu->i = device_i;
    matrix_csr_cpu->j = device_j;
    matrix_csr_cpu->val = device_val;

    // Allocate the MatrixCSR struct itself on the GPU
    err = cudaMalloc(&matrix_csr, sizeof(MatrixCSR));
    if (err != cudaSuccess) { /* handle or log error, cleanup GPU arrays */ cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); free(matrix_csr_cpu); return NULL;}
    
    // Copy the updated CPU-side struct (now holding GPU pointers) to the GPU-allocated struct
    err = cudaMemcpy(matrix_csr, matrix_csr_cpu, sizeof(MatrixCSR),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { /* handle or log error, cleanup GPU arrays and matrix_csr struct */ cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); cudaFree(matrix_csr); free(matrix_csr_cpu); return NULL;}


    free(matrix_csr_cpu); // Free the CPU-side struct shell

    return matrix_csr;
}


/*
    Description : It reads the matrix in the file to Unified Memory 
    in CSC format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCSC * readMatrixUnifiedMemoryCSC(std::string filename) {
    std::ifstream fin(filename);
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;
    // Read the data

    MatrixCSC * matrix_csc;
    cudaMallocManaged(&matrix_csc, sizeof(MatrixCSC));

    matrix_csc->rows = M;
    matrix_csc->cols = N;
    matrix_csc->nnz = L;

    cudaMallocManaged(&matrix_csc->i, sizeof(int) * L);
    cudaMallocManaged(&matrix_csc->j, sizeof(int) * (matrix_csc->cols + 1));
    cudaMallocManaged(&matrix_csc->val, sizeof(float) * L);

    int filled = 0;

    auto start = std::chrono::system_clock::now();

    std::vector< std::vector <std::pair<int, float> > > matrix_data(N); // N for CSC
    for (int l = 0; l < L; l++) {
        int m, n;
        float data;
        fin >> m >> n >> data;
        matrix_data[n - 1].push_back({m - 1, data}); // Store by column
    }

    for(int i = 0; i < N; i++) { // Iterate through columns
        std::sort(matrix_data[i].begin(), matrix_data[i].end());
        matrix_csc->j[i] = filled; // CSC colPtr
        for(auto tp : matrix_data[i]) {
            matrix_csc->i[filled] = tp.first; // CSC rowInd
            matrix_csc->val[filled] = tp.second;
            filled++;
        }
    }

    matrix_csc->j[N] = filled; // Last entry in colPtr

    assert(filled == L);
    fin.close();
    printInfo("Read matrix from file: " + filename, 4);
    printInfo("Matrix size : " + itoa(M) + " x " + itoa(N) + ".", 4);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;


    return matrix_csc;
}


/*
    Description : It reads the matrix in the file to CPU memory 
    in CSC format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCSC * readMatrixCPUMemoryCSC(std::string filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        printf("Error: Could not open file %s\n", filename.c_str());
        exit(1);
    }
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;
    // Read the data

    MatrixCSC * matrix_csc;
    matrix_csc = (MatrixCSC *) malloc(sizeof(MatrixCSC));

    matrix_csc->rows = M;
    matrix_csc->cols = N;
    matrix_csc->nnz = L;

    matrix_csc->i = (int *) malloc(sizeof(int) * L);
    matrix_csc->j = (int *) malloc(sizeof(int) * (matrix_csc->cols + 1));
    matrix_csc->val = (float *) malloc(sizeof(float) * L);

    int filled = 0;

    auto start = std::chrono::system_clock::now();
    
    std::vector<std::pair<int, float>> *matrix_data_ptr = new std::vector<std::pair<int, float>>[N]; // N for CSC

    for (int l = 0; l < L; l++) {
        int m, n_val; // Renamed n to n_val to avoid conflict with parameter n in other contexts
        float data;
        fin >> m >> n_val >> data;
         if (n_val - 1 >= 0 && n_val - 1 < N) { // Bounds check for column index
            matrix_data_ptr[n_val - 1].push_back({m - 1, data});
        } else {
            fprintf(stderr, "Warning: Column index %d out of bounds for matrix with %d columns in file %s\n", n_val, N, filename.c_str());
        }
    }

    for(int col_idx = 0; col_idx < N; col_idx++) { // Iterate through columns
        std::sort(matrix_data_ptr[col_idx].begin(), matrix_data_ptr[col_idx].end());
        matrix_csc->j[col_idx] = filled; // CSC colPtr
        for(auto tp : matrix_data_ptr[col_idx]) {
            if (filled < L) { // Bounds check
                matrix_csc->i[filled] = tp.first; // CSC rowInd
                matrix_csc->val[filled] = tp.second;
                filled++;
            } else {
                fprintf(stderr, "Warning: More non-zero elements found than specified L=%d in file %s for CSC format.\n", L, filename.c_str());
                break;
            }
        }
        if (filled >= L && col_idx < N -1) {
             for (int k = col_idx + 1; k < N; ++k) {
                if (!matrix_data_ptr[k].empty()){
                     fprintf(stderr, "Warning: Non-zero elements found after L limit in file %s for CSC format.\n", filename.c_str());
                     break;
                }
            }
            if (filled > L) {
                 fprintf(stderr, "Warning: CSC Actual nnz %d exceeded reported L %d. Updating nnz.\n", filled, L);
                 matrix_csc->nnz = filled;
            }
            // break; 
        }
    }
    delete[] matrix_data_ptr;

    matrix_csc->j[N] = filled; // Last entry in colPtr

    if (filled != L) {
        fprintf(stderr, "Warning: CSC Final nnz count %d does not match L %d. Using actual count %d.\n", filled, L, filled);
        matrix_csc->nnz = filled;
    }
    assert(filled <= L || matrix_csc->nnz == filled);
    fin.close();
    printInfo("Read matrix (CSC on CPU) from file: " + filename, 4);
    printInfo("Matrix size : " + itoa(M) + " x " + itoa(N) + ".", 4);
    printInfo("NNZ read : " + itoa(matrix_csc->nnz) + ".", 4);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;


    return matrix_csc;
}


/*
    Description : It reads the matrix in the file to GPU memory 
    in CSC format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCSC * readMatrixGPUMemoryCSC(std::string filename) {

    MatrixCSC * matrix_csc_cpu = readMatrixCPUMemoryCSC(filename);
    if (matrix_csc_cpu == NULL) return NULL;
    MatrixCSC * matrix_csc;

    int * device_i, * device_j;
    float * device_val;
    
    cudaError_t err;

    err = cudaMalloc(&device_i, sizeof(int) * matrix_csc_cpu->nnz);
    if (err != cudaSuccess) { free(matrix_csc_cpu); return NULL; }
    err = cudaMalloc(&device_j, sizeof(int) * (matrix_csc_cpu->cols + 1));
    if (err != cudaSuccess) { cudaFree(device_i); free(matrix_csc_cpu); return NULL; }
    err = cudaMalloc(&device_val, sizeof(float) * matrix_csc_cpu->nnz);
    if (err != cudaSuccess) { cudaFree(device_i); cudaFree(device_j); free(matrix_csc_cpu); return NULL; }

    err = cudaMemcpy(device_i, matrix_csc_cpu->i,
        sizeof(int) * matrix_csc_cpu->nnz, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); free(matrix_csc_cpu); return NULL; }
        
    err = cudaMemcpy(device_j, matrix_csc_cpu->j,
        sizeof(int) * (matrix_csc_cpu->cols + 1), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); free(matrix_csc_cpu); return NULL; }

    err = cudaMemcpy(device_val, matrix_csc_cpu->val,
        sizeof(float) * matrix_csc_cpu->nnz, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); free(matrix_csc_cpu); return NULL; }


    free(matrix_csc_cpu->i);
    free(matrix_csc_cpu->j);
    free(matrix_csc_cpu->val);

    matrix_csc_cpu->i = device_i;
    matrix_csc_cpu->j = device_j;
    matrix_csc_cpu->val = device_val;

    err = cudaMalloc(&matrix_csc, sizeof(MatrixCSC));
    if (err != cudaSuccess) { cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); free(matrix_csc_cpu); return NULL; }
    
    err = cudaMemcpy(matrix_csc, matrix_csc_cpu, sizeof(MatrixCSC),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(device_i); cudaFree(device_j); cudaFree(device_val); cudaFree(matrix_csc); free(matrix_csc_cpu); return NULL; }


    free(matrix_csc_cpu);

    return matrix_csc;
}


/*
    Description : Writes given CSR matrix to file.

    Parameters : 
        string filename : Path to file.
        MatrixCSR * matrix : Matrix to write.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

void writeMatrixCSRCPU(std::string filename, MatrixCSR * matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    int nnz = matrix->nnz;
    std::ofstream fout;
    fout.open(filename);
    fout << "%%MatrixMarket matrix coordinate real general " << std::endl;
    fout << rows << " " << cols << " " << nnz << std::endl;
    for(int i = 0; i < rows; i++) {
        for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
            fout << i + 1 << " " << matrix->j[j] + 1 << " " <<             matrix->val[j] << std::endl;
            // Adding 1 for exporting to .mtx format.
        }
    }
    fout.close();
}


/*
    Description : Print configuration of #directives.
*/

void printConfig() {
    printInfo("Configuration", 4);

    #ifdef BLELLOCH
        printInfo("Prallel prefix sum using CUB : YES", 8);
    #else
        printInfo("Prallel prefix sum using CUB : NO", 8);
    #endif

    #ifdef BFS_WORK_EFFICIENT
        printInfo("Work efficient Merill's BFS  : YES", 8);
    #else
        printInfo("Work efficient Merill's BFS  : NO", 8);
    #endif

    #ifdef AGGREGATION_WORK_EFFICIENT
        printInfo("Aggregation work efficient   : YES", 8);
    #else
        printInfo("Aggregation work efficient   : NO", 8);
    #endif

    #ifdef DEBUG
        printInfo("Debug Mode On                : YES", 8);
    #else
        printInfo("Debug Mode On                : NO", 8);
    #endif

}

#endif
