#include <assert.h>
#include "bits/stdc++.h"
using namespace std;
#include <vector>

struct MatrixCOO {
    int rows, cols, nnz;
    int * i, * j;
    double * val;
};

struct MatrixCSR {
    int rows, cols, nnz;
    int * i, * j;
    double * val;
};

struct MatrixCSC {
    int rows, cols, nnz;
    int * i, * j;
    double * val;
};


MatrixCOO * readMatrixUnifiedMemoryCOO(string filename) {
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
    cudaMallocManaged(&matrix_coo->val, sizeof(double) * L);

    int filled = 0;

    auto start = std::chrono::system_clock::now();

    vector< vector <pair<int, double> > > matrix_data(M);
    for (int l = 0; l < L; l++) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        matrix_data[m - 1].push_back({n - 1, data});
    }
    for(int i = 0; i < M; i++) {
        sort(matrix_data[i].begin(), matrix_data[i].end());
        for(auto tp : matrix_data[i]) {
            matrix_coo->i[filled] = i;
            matrix_coo->j[filled] = tp.first;
            matrix_coo->val[filled] = tp.second;
            filled++;
        }
    }

    assert(filled == L);
    fin.close();
    cerr << "Read matrix from file: " << filename << endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_coo;
}

MatrixCOO * readMatrixCPUMemoryCOO(string filename) {
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
    matrix_coo->val = (double *) malloc(sizeof(double) * L);
    
    int filled = 0;

    auto start = std::chrono::system_clock::now();

    vector< vector <pair<int, double> > > matrix_data(M);
    for (int l = 0; l < L; l++) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        matrix_data[m - 1].push_back({n - 1, data});
    }
    for(int i = 0; i < M; i++) {
        sort(matrix_data[i].begin(), matrix_data[i].end());
        for(auto tp : matrix_data[i]) {
            matrix_coo->i[filled] = i;
            matrix_coo->j[filled] = tp.first;
            matrix_coo->val[filled] = tp.second;
            filled++;
        }
    }

    assert(filled == L);
    fin.close();
    cerr << "Read matrix from file: " << filename << endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_coo;
}

MatrixCOO * readMatrixGPUMemoryCOO(string filename) {

    MatrixCOO * matrix_coo_cpu = readMatrixCPUMemoryCOO(filename);
    MatrixCOO * matrix_coo;

    int * device_i, * device_j;
    double * device_val;

    cudaMalloc(&device_i, sizeof(int) * matrix_coo_cpu->nnz);
    cudaMalloc(&device_j, sizeof(int) * matrix_coo_cpu->nnz);
    cudaMalloc(&device_val, sizeof(double) * matrix_coo_cpu->nnz);

    cudaMemcpy(device_i, matrix_coo_cpu->i, sizeof(int) * matrix_coo_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_j, matrix_coo_cpu->j, sizeof(int) * matrix_coo_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, matrix_coo_cpu->val, sizeof(double) * matrix_coo_cpu->nnz, cudaMemcpyHostToDevice);


    free(matrix_coo_cpu->i);
    free(matrix_coo_cpu->j);
    free(matrix_coo_cpu->val);

    matrix_coo_cpu->i = device_i;
    matrix_coo_cpu->j = device_j;
    matrix_coo_cpu->val = device_val;

    cudaMalloc(&matrix_coo, sizeof(MatrixCOO));
    cudaMemcpy(matrix_coo, matrix_coo_cpu, sizeof(MatrixCOO), cudaMemcpyHostToDevice);

    free(matrix_coo_cpu);

    return matrix_coo;
}


MatrixCSR * readMatrixUnifiedMemoryCSR(string filename) {
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
    cudaMallocManaged(&matrix_csr->val, sizeof(double) * L);

    int filled = 0;

    auto start = std::chrono::system_clock::now();

    vector< vector <pair<int, double> > > matrix_data(M);
    for (int l = 0; l < L; l++) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        matrix_data[m - 1].push_back({n - 1, data});
    }
    
    for(int i = 0; i < M; i++) {
        sort(matrix_data[i].begin(), matrix_data[i].end());
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
    cerr << "Read matrix from file: " << filename << endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_csr;
}

MatrixCSR * readMatrixCPUMemoryCSR(string filename) {
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
    matrix_csr->val = (double *) malloc(sizeof(double) * L);

    int filled = 0;

    auto start = std::chrono::system_clock::now();

    vector< vector <pair<int, double> > > matrix_data(M);
    for (int l = 0; l < L; l++) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        matrix_data[m - 1].push_back({n - 1, data});
    }
    
    for(int i = 0; i < M; i++) {
        sort(matrix_data[i].begin(), matrix_data[i].end());
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
    cerr << "Read matrix from file: " << filename << endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_csr;
}

MatrixCSR * readMatrixGPUMemoryCSR(string filename) {

    MatrixCSR * matrix_csr_cpu = readMatrixCPUMemoryCSR(filename);
    MatrixCSR * matrix_csr;

    int * device_i, * device_j;
    double * device_val;

    cudaMalloc(&device_i, sizeof(int) * matrix_csr_cpu->nnz);
    cudaMalloc(&device_j, sizeof(int) * matrix_csr_cpu->nnz);
    cudaMalloc(&device_val, sizeof(double) * matrix_csr_cpu->nnz);

    cudaMemcpy(device_i, matrix_csr_cpu->i, sizeof(int) * (matrix_csr_cpu->rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_j, matrix_csr_cpu->j, sizeof(int) * matrix_csr_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, matrix_csr_cpu->val, sizeof(double) * matrix_csr_cpu->nnz, cudaMemcpyHostToDevice);


    free(matrix_csr_cpu->i);
    free(matrix_csr_cpu->j);
    free(matrix_csr_cpu->val);

    matrix_csr_cpu->i = device_i;
    matrix_csr_cpu->j = device_j;
    matrix_csr_cpu->val = device_val;

    cudaMalloc(&matrix_csr, sizeof(MatrixCSR));
    cudaMemcpy(matrix_csr, matrix_csr_cpu, sizeof(MatrixCSR), cudaMemcpyHostToDevice);

    free(matrix_csr_cpu);

    return matrix_csr;
}

MatrixCSC * readMatrixUnifiedMemoryCSC(string filename) {
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
    cudaMallocManaged(&matrix_csc->val, sizeof(double) * L);

    int filled = 0;

    auto start = std::chrono::system_clock::now();

    vector< vector <pair<int, double> > > matrix_data(N);
    for (int l = 0; l < L; l++) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        matrix_data[n - 1].push_back({m - 1, data});
    }
    
    for(int i = 0; i < N; i++) {
        sort(matrix_data[i].begin(), matrix_data[i].end());
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
    cerr << "Read matrix from file: " << filename << endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_csc;
}

MatrixCSC * readMatrixCPUMemoryCSC(string filename) {
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
    matrix_csc->val = (double *) malloc(sizeof(double) * L);

    int filled = 0;

    auto start = std::chrono::system_clock::now();

    vector< vector <pair<int, double> > > matrix_data(N);
    for (int l = 0; l < L; l++) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        matrix_data[n - 1].push_back({m - 1, data});
    }
    
    for(int i = 0; i < N; i++) {
        sort(matrix_data[i].begin(), matrix_data[i].end());
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
    cerr << "Read matrix from file: " << filename << endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_csc;
}

MatrixCSC * readMatrixGPUMemoryCSC(string filename) {

    MatrixCSC * matrix_csc_cpu = readMatrixCPUMemoryCSC(filename);
    MatrixCSC * matrix_csc;

    int * device_i, * device_j;
    double * device_val;

    cudaMalloc(&device_i, sizeof(int) * matrix_csc_cpu->nnz);
    cudaMalloc(&device_j, sizeof(int) * matrix_csc_cpu->nnz);
    cudaMalloc(&device_val, sizeof(double) * matrix_csc_cpu->nnz);

    cudaMemcpy(device_i, matrix_csc_cpu->i, sizeof(int) * matrix_csc_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_j, matrix_csc_cpu->j, sizeof(int) * (matrix_csc_cpu->cols + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, matrix_csc_cpu->val, sizeof(double) * matrix_csc_cpu->nnz, cudaMemcpyHostToDevice);

    free(matrix_csc_cpu->i);
    free(matrix_csc_cpu->j);
    free(matrix_csc_cpu->val);

    matrix_csc_cpu->i = device_i;
    matrix_csc_cpu->j = device_j;
    matrix_csc_cpu->val = device_val;

    cudaMalloc(&matrix_csc, sizeof(MatrixCSC));
    cudaMemcpy(matrix_csc, matrix_csc_cpu, sizeof(MatrixCSC), cudaMemcpyHostToDevice);

    free(matrix_csc_cpu);

    return matrix_csc;
}