/*
    Description : This file contains the functions to access an element in
    the matrix formats present in MatrixIo.cu file.

    @author : mishraiiit
*/

#ifndef MATRIX_IO
#define MATRIX_IO
#include <assert.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>

/*
    Description : Struct declaration for a matrix in COO format. The matrix is
    assumed to be zero-indexed i.e rows start from 0 and not 1.

    Parameters : 
        int rows : The number of rows in the matrix.
        int cols : The number of cols in the matrix.
        int nnz : The number of non-zero entries in the matrix.
        int * i : rowInd for the matrix. Size is nnz.
        int * j : colInd for the matrix. Size is nnz.
        float * val : values for the matrix. Size is nnz.

    @author : mishraiiit
*/

struct MatrixCOO {
    int rows, cols, nnz;
    int * i, * j;
    float * val;
};


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
    in COO format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCOO * readMatrixUnifiedMemoryCOO(std::string filename) {
    std::ifstream fin(filename);
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;
    // Read the data

    MatrixCOO * matrix_coo;
    cudaMallocManaged(&matrix_coo, sizeof(MatrixCOO));

    matrix_coo->rows = M;
    matrix_coo->cols = N;
    matrix_coo->nnz = L;

    cudaMallocManaged(&matrix_coo->i, sizeof(int) * L);
    cudaMallocManaged(&matrix_coo->j, sizeof(int) * L);
    cudaMallocManaged(&matrix_coo->val, sizeof(float) * L);

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
        for(auto tp : matrix_data[i]) {
            matrix_coo->i[filled] = i;
            matrix_coo->j[filled] = tp.first;
            matrix_coo->val[filled] = tp.second;
            filled++;
        }
    }

    assert(filled == L);
    fin.close();
    printInfo("Read matrix from file: " + filename, 4);
    printInfo("Matrix size : " + itoa(M) + " x " + itoa(N) + ".", 4);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;


    return matrix_coo;
}


/*
    Description : It reads the matrix in the file to main memory 
    in COO format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCOO * readMatrixCPUMemoryCOO(std::string filename) {
    std::ifstream fin(filename);
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;
    // Read the data

    MatrixCOO * matrix_coo;
    matrix_coo =  (MatrixCOO *) malloc(sizeof(MatrixCOO));

    matrix_coo->rows = M;
    matrix_coo->cols = N;
    matrix_coo->nnz = L;

    matrix_coo->i = (int *) malloc(sizeof(int) * L);
    matrix_coo->j = (int *) malloc(sizeof(int) * L);
    matrix_coo->val = (float *) malloc(sizeof(float) * L);

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
        for(auto tp : matrix_data[i]) {
            matrix_coo->i[filled] = i;
            matrix_coo->j[filled] = tp.first;
            matrix_coo->val[filled] = tp.second;
            filled++;
        }
    }

    assert(filled == L);
    fin.close();
    printInfo("Read matrix from file: " + filename, 4);
    printInfo("Matrix size : " + itoa(M) + " x " + itoa(N) + ".", 4);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;


    return matrix_coo;
}


/*
    Description : It reads the matrix in the file to GPU memory 
    in COO format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCOO * readMatrixGPUMemoryCOO(std::string filename) {

    MatrixCOO * matrix_coo_cpu = readMatrixCPUMemoryCOO(filename);
    MatrixCOO * matrix_coo;

    int * device_i, * device_j;
    float * device_val;

    cudaMalloc(&device_i, sizeof(int) * matrix_coo_cpu->nnz);
    cudaMalloc(&device_j, sizeof(int) * matrix_coo_cpu->nnz);
    cudaMalloc(&device_val, sizeof(float) * matrix_coo_cpu->nnz);

    cudaMemcpy(device_i, matrix_coo_cpu->i,
        sizeof(int) * matrix_coo_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_j, matrix_coo_cpu->j,
        sizeof(int) * matrix_coo_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, matrix_coo_cpu->val,
        sizeof(float) * matrix_coo_cpu->nnz, cudaMemcpyHostToDevice);


    free(matrix_coo_cpu->i);
    free(matrix_coo_cpu->j);
    free(matrix_coo_cpu->val);

    matrix_coo_cpu->i = device_i;
    matrix_coo_cpu->j = device_j;
    matrix_coo_cpu->val = device_val;

    cudaMalloc(&matrix_coo, sizeof(MatrixCOO));
    cudaMemcpy(matrix_coo, matrix_coo_cpu, sizeof(MatrixCOO),
        cudaMemcpyHostToDevice);

    free(matrix_coo_cpu);

    return matrix_coo;
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
    Description : It reads the matrix in the file to Unified Memory 
    in COO format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCSR * readMatrixCPUMemoryCSR(std::string filename) {
    std::ifstream fin(filename);
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
    Description : It reads the matrix in the file to GPU memory
    in CSR format.

    Parameters : 
        string filename : Path to file.

    Comments : File should be in .mtx format.

    @author : mishraiiit
*/

MatrixCSR * readMatrixGPUMemoryCSR(std::string filename) {

    MatrixCSR * matrix_csr_cpu = readMatrixCPUMemoryCSR(filename);
    MatrixCSR * matrix_csr;

    int * device_i, * device_j;
    float * device_val;

    cudaMalloc(&device_i, sizeof(int) * matrix_csr_cpu->nnz);
    cudaMalloc(&device_j, sizeof(int) * matrix_csr_cpu->nnz);
    cudaMalloc(&device_val, sizeof(float) * matrix_csr_cpu->nnz);

    cudaMemcpy(device_i, matrix_csr_cpu->i,
        sizeof(int) * (matrix_csr_cpu->rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_j, matrix_csr_cpu->j,
        sizeof(int) * matrix_csr_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, matrix_csr_cpu->val,
        sizeof(float) * matrix_csr_cpu->nnz, cudaMemcpyHostToDevice);


    free(matrix_csr_cpu->i);
    free(matrix_csr_cpu->j);
    free(matrix_csr_cpu->val);

    matrix_csr_cpu->i = device_i;
    matrix_csr_cpu->j = device_j;
    matrix_csr_cpu->val = device_val;

    cudaMalloc(&matrix_csr, sizeof(MatrixCSR));
    cudaMemcpy(matrix_csr, matrix_csr_cpu, sizeof(MatrixCSR),
        cudaMemcpyHostToDevice);

    free(matrix_csr_cpu);

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

    std::vector< std::vector <std::pair<int, float> > > matrix_data(N);
    for (int l = 0; l < L; l++) {
        int m, n;
        float data;
        fin >> m >> n >> data;
        matrix_data[n - 1].push_back({m - 1, data});
    }

    for(int i = 0; i < N; i++) {
        std::sort(matrix_data[i].begin(), matrix_data[i].end());
        matrix_csc->j[i] = filled;
        for(auto tp : matrix_data[i]) {
            matrix_csc->i[filled] = tp.first;
            matrix_csc->val[filled] = tp.second;
            filled++;
        }
    }

    matrix_csc->j[N] = filled;

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

    std::vector< std::vector <std::pair<int, float> > > matrix_data(N);
    for (int l = 0; l < L; l++) {
        int m, n;
        float data;
        fin >> m >> n >> data;
        matrix_data[n - 1].push_back({m - 1, data});
    }

    for(int i = 0; i < N; i++) {
        std::sort(matrix_data[i].begin(), matrix_data[i].end());
        matrix_csc->j[i] = filled;
        for(auto tp : matrix_data[i]) {
            matrix_csc->i[filled] = tp.first;
            matrix_csc->val[filled] = tp.second;
            filled++;
        }
    }

    matrix_csc->j[N] = filled;

    assert(filled == L);
    fin.close();
    printInfo("Read matrix from file: " + filename, 4);
    printInfo("Matrix size : " + itoa(M) + " x " + itoa(N) + ".", 4);

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
    MatrixCSC * matrix_csc;

    int * device_i, * device_j;
    float * device_val;

    cudaMalloc(&device_i, sizeof(int) * matrix_csc_cpu->nnz);
    cudaMalloc(&device_j, sizeof(int) * matrix_csc_cpu->nnz);
    cudaMalloc(&device_val, sizeof(float) * matrix_csc_cpu->nnz);

    cudaMemcpy(device_i, matrix_csc_cpu->i,
        sizeof(int) * matrix_csc_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_j, matrix_csc_cpu->j,
        sizeof(int) * (matrix_csc_cpu->cols + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, matrix_csc_cpu->val,
        sizeof(float) * matrix_csc_cpu->nnz, cudaMemcpyHostToDevice);

    free(matrix_csc_cpu->i);
    free(matrix_csc_cpu->j);
    free(matrix_csc_cpu->val);

    matrix_csc_cpu->i = device_i;
    matrix_csc_cpu->j = device_j;
    matrix_csc_cpu->val = device_val;

    cudaMalloc(&matrix_csc, sizeof(MatrixCSC));
    cudaMemcpy(matrix_csc, matrix_csc_cpu, sizeof(MatrixCSC),
        cudaMemcpyHostToDevice);

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
            fout << i + 1 << " " << matrix->j[j] + 1 << " " << \
            matrix->val[j] << std::endl;
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
