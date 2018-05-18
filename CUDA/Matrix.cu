#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <fstream>

struct MatrixCOO {
    int rows, cols, nnz;
    int * i, * j;
    float * val;
};

struct MatrixCSR {
    int rows, cols, nnz;
    int * i, * j;
    float * val;
};

struct MatrixCSC {
    int rows, cols, nnz;
    int * i, * j;
    float * val;
};

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
    std::cerr << "Read matrix from file: " << filename << std::endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_coo;
}

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
    std::cerr << "Read matrix from file: " << filename << std::endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_coo;
}

MatrixCOO * readMatrixGPUMemoryCOO(std::string filename) {

    MatrixCOO * matrix_coo_cpu = readMatrixCPUMemoryCOO(filename);
    MatrixCOO * matrix_coo;

    int * device_i, * device_j;
    float * device_val;

    cudaMalloc(&device_i, sizeof(int) * matrix_coo_cpu->nnz);
    cudaMalloc(&device_j, sizeof(int) * matrix_coo_cpu->nnz);
    cudaMalloc(&device_val, sizeof(float) * matrix_coo_cpu->nnz);

    cudaMemcpy(device_i, matrix_coo_cpu->i, sizeof(int) * matrix_coo_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_j, matrix_coo_cpu->j, sizeof(int) * matrix_coo_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, matrix_coo_cpu->val, sizeof(float) * matrix_coo_cpu->nnz, cudaMemcpyHostToDevice);


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
    std::cerr << "Read matrix from file: " << filename << std::endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_csr;
}

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
    std::cerr << "Read matrix from file: " << filename << std::endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_csr;
}

MatrixCSR * readMatrixGPUMemoryCSR(std::string filename) {

    MatrixCSR * matrix_csr_cpu = readMatrixCPUMemoryCSR(filename);
    MatrixCSR * matrix_csr;

    int * device_i, * device_j;
    float * device_val;

    cudaMalloc(&device_i, sizeof(int) * matrix_csr_cpu->nnz);
    cudaMalloc(&device_j, sizeof(int) * matrix_csr_cpu->nnz);
    cudaMalloc(&device_val, sizeof(float) * matrix_csr_cpu->nnz);

    cudaMemcpy(device_i, matrix_csr_cpu->i, sizeof(int) * (matrix_csr_cpu->rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_j, matrix_csr_cpu->j, sizeof(int) * matrix_csr_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, matrix_csr_cpu->val, sizeof(float) * matrix_csr_cpu->nnz, cudaMemcpyHostToDevice);


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
    std::cerr << "Read matrix from file: " << filename << std::endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_csc;
}

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
    std::cerr << "Read matrix from file: " << filename << std::endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = end-start;
    std::cerr << "Time for reading: " << diff.count() << " s\n";

    return matrix_csc;
}

MatrixCSC * readMatrixGPUMemoryCSC(std::string filename) {

    MatrixCSC * matrix_csc_cpu = readMatrixCPUMemoryCSC(filename);
    MatrixCSC * matrix_csc;

    int * device_i, * device_j;
    float * device_val;

    cudaMalloc(&device_i, sizeof(int) * matrix_csc_cpu->nnz);
    cudaMalloc(&device_j, sizeof(int) * matrix_csc_cpu->nnz);
    cudaMalloc(&device_val, sizeof(float) * matrix_csc_cpu->nnz);

    cudaMemcpy(device_i, matrix_csc_cpu->i, sizeof(int) * matrix_csc_cpu->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_j, matrix_csc_cpu->j, sizeof(int) * (matrix_csc_cpu->cols + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, matrix_csc_cpu->val, sizeof(float) * matrix_csc_cpu->nnz, cudaMemcpyHostToDevice);

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

__global__ void computeRowColAbsSum(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, bool * ising0, float ktg) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= matrix_csr->rows)
        return;

    int row_start = matrix_csr->i[id];
    int row_end = matrix_csr->i[id + 1];

    int col_start = matrix_csc->j[id];
    int col_end = matrix_csc->j[id + 1];

    float ans = 0;
    while(row_start < row_end || col_start < col_end) {
        if(row_start < row_end && col_start < col_end) {
            if(matrix_csr->j[row_start] < matrix_csc->i[col_start]) {
                if(matrix_csr->j[row_start] != id)
                    ans += abs(matrix_csr->val[row_start]) / 2;
                row_start++;
            } else if(matrix_csr->j[row_start] > matrix_csc->i[col_start]) {
                if(matrix_csc->i[col_start] != id)
                    ans += abs(matrix_csc->val[col_start]) / 2;
                col_start++;
            } else {
                if(matrix_csr->j[row_start] != id)
                    ans += abs(matrix_csr->val[row_start] + matrix_csc->val[col_start]) / 2;
                row_start++;
                col_start++;
            }
        } 
        else if(row_start < row_end) {
            if(matrix_csr->j[row_start] != id)
                ans += abs(matrix_csr->val[row_start]) / 2;
            row_start++;
        } else {
            if(matrix_csc->i[col_start] != id)
                ans += abs(matrix_csc->val[col_start]) / 2;
            col_start++;
        }
    }
    float aii = getElementMatrixCSR(matrix_csr, id, id);
    float rhs = (ktg / (ktg - 2)) * ans;
    ising0[id] = aii >= rhs;
}

__global__ void comptueSi(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, float * output) {
int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= matrix_csr->rows)
        return;

    int row_start = matrix_csr->i[id];
    int row_end = matrix_csr->i[id + 1];

    int col_start = matrix_csc->j[id];
    int col_end = matrix_csc->j[id + 1];

    float ans = 0;
    while(row_start < row_end || col_start < col_end) {
        if(row_start < row_end && col_start < col_end) {
            if(matrix_csr->j[row_start] < matrix_csc->i[col_start]) {
                if(matrix_csr->j[row_start] != id)
                    ans += matrix_csr->val[row_start] / 2;
                row_start++;
            } else if(matrix_csr->j[row_start] > matrix_csc->i[col_start]) {
                if(matrix_csc->i[col_start] != id)
                    ans += matrix_csc->val[col_start] / 2;
                col_start++;
            } else {
                if(matrix_csr->j[row_start] != id)
                    ans += (matrix_csr->val[row_start] + matrix_csc->val[col_start]) / 2;
                row_start++;
                col_start++;
            }
        } 
        else if(row_start < row_end) {
            if(matrix_csr->j[row_start] != id)
                ans += matrix_csr->val[row_start] / 2;
            row_start++;
        } else {
            if(matrix_csc->i[col_start] != id)
                ans += matrix_csc->val[col_start] / 2;
            col_start++;
        }
    }

    output[id] = -ans;
}

__host__ void comptueSiHost(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, float * output) {    
    for(int id = 0; id < matrix_csr->rows; id++) {

        int row_start = matrix_csr->i[id];
        int row_end = matrix_csr->i[id + 1];

        int col_start = matrix_csc->j[id];
        int col_end = matrix_csc->j[id + 1];

        float ans = 0;
        while(row_start < row_end || col_start < col_end) {
            if(row_start < row_end && col_start < col_end) {
                if(matrix_csr->j[row_start] < matrix_csc->i[col_start]) {
                    if(matrix_csr->j[row_start] != id)
                        ans += matrix_csr->val[row_start] / 2;
                    row_start++;
                } else if(matrix_csr->j[row_start] > matrix_csc->i[col_start]) {
                    if(matrix_csc->i[col_start] != id)
                        ans += matrix_csc->val[col_start] / 2;
                    col_start++;
                } else {
                    if(matrix_csr->j[row_start] != id)
                        ans += (matrix_csr->val[row_start] + matrix_csc->val[col_start]) / 2;
                    row_start++;
                    col_start++;
                }
            } 
            else if(row_start < row_end) {
                if(matrix_csr->j[row_start] != id)
                    ans += matrix_csr->val[row_start] / 2;
                row_start++;
            } else {
                if(matrix_csc->i[col_start] != id)
                    ans += matrix_csc->val[col_start] / 2;
                col_start++;
            }
        }

        output[id] = -ans;
    }
}

__host__ __device__ float muij(int i, int j, MatrixCSR * matrix_csr, float * Si) {
    float aii = getElementMatrixCSR(matrix_csr, i, i);
    float ajj = getElementMatrixCSR(matrix_csr, j, j);
    float aij = getElementMatrixCSR(matrix_csr, i, j);
    float aji = getElementMatrixCSR(matrix_csr, j, i);

    float num = 2 * (1 / ((1 / aii) + (1 / ajj)));
    float den = (- (aij + aji) / 2) + 1 / ( ( 1 / (aii - Si[i])) + (1 / (ajj - Si[j])) );
    return num / den;
}

template<typename T>
__device__ void swap_variables(T & u, T & v) {
    T temp = u;
    u = v;
    v = temp;
}

__global__ void sortNeighbourList(MatrixCSR * matrix, MatrixCSR * neighbour_list, float * Si, bool * allowed, float ktg, bool * ising0) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= matrix->rows)
        return;

    int row_start = matrix->i[id];
    int row_end = matrix->i[id + 1];

    for(int i = row_start; i < row_end; i++) {
        for(int j = i + 1; j < row_end; j++) {
            int id1 = neighbour_list->j[i];
            int id2 = neighbour_list->j[j];
            if(muij(id, id2, matrix, Si) < muij(id, id1, matrix, Si)) {
                swap_variables(neighbour_list->j[i], neighbour_list->j[j]);
                swap_variables(neighbour_list->val[i], neighbour_list->val[j]);
            }
        }
    }

    for(int i = row_start; i < row_end; i++) {
        int id1 = neighbour_list->j[i];
        float mij = muij(id, id1, matrix, Si);
        allowed[i] = (0 < mij && mij <= ktg);
        if(ising0[id] || ising0[id1]) {
            allowed[i] = false;
        }
    }    
}

__global__ void printNeighbourList(MatrixCSR * matrix, MatrixCSR * neighbour_list, float * Si) {

    for(int id = 0; id < neighbour_list->rows; id++) {
        int row_start = neighbour_list->i[id];
        int row_end = neighbour_list->i[id + 1];

        printf("Neighbours for %d\n", id);
        for(int i = row_start; i < row_end; i++) {
            printf("%d %lf\n", neighbour_list->j[i], muij(id, neighbour_list->j[i], matrix, Si));        
        }
    }
    printf("\n");
}

__global__ void mis(MatrixCSR * matrix, int * inmis) {
    for(int i = 0; i < 10000; i++) {
        if(((i / 100) + (i % 100)) % 2 == 0) {
            inmis[i] = true;
        } else {
            inmis[i] = false;
        }
    }
}

__global__ void aggregation_initial(int n, int * paired_with) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < n) {
        paired_with[id] = -1;
    }
}

__device__ bool okay(int i, int j, MatrixCSR * matrix, float * Si) {
    return (getElementMatrixCSR(matrix, i, i) - Si[i] + getElementMatrixCSR(matrix, j, j) - Si[j] >= 0);
}

/*
    Comments:
    This is the main function for initial_std::pairwise_aggregation.
    n is the number of rows in the matrix.
    neighbour_list is the matrix (adjacency matrix) in CSR format.
    allowed marks the positions in the neighbour_list which are not useful
    (the links with which aggregation shouldn't be formed).
    distance tells the distance of the nodes in which aggregation is done on in this kernel. Odd or Even.
    Si is the array containing the Si values in the paper.
    ising0 contains the node which are to be kept out of aggregation.
    bfs_distance tells the distance of a node from node 0.
*/
__global__ void aggregation(int n, MatrixCSR * neighbour_list, int * paired_with, bool * allowed, MatrixCSR * matrix, float * Si, int distance, bool * ising0, int * bfs_distance) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;
    if(ising0[i]) return;
    if(bfs_distance[i]  % 2 != distance) return;
    if(paired_with[i] != -1) return;

    for(int j = neighbour_list->i[i]; j < neighbour_list->i[i + 1]; j++) {
        if(!allowed[j]) continue;
        int possible_j = neighbour_list->j[j];
        if(bfs_distance[i] == bfs_distance[possible_j]) continue;
        if(!okay(i, possible_j, matrix, Si)) continue;
        if(atomicCAS(&paired_with[possible_j], -1, i) == -1) {
            paired_with[i] = possible_j;
            paired_with[possible_j] = i;
            // printf("%d %d\n", i, possible_j);
            return;
        }
    }

    paired_with[i] = i;
    // printf("%d %d\n", i, paired_with[i]);
}

/*
    Comments:
    A class for recording time. tic() for time start. toc() for time end.
    For recording time taken on GPU.
    After the kernel use device synchronization.
*/
class TicToc {
    public:
        std::chrono::time_point<std::chrono::system_clock> start, end;
        std::string s;
        TicToc(std::string s) : s(s) {

        }
        void tic() {
            start = std::chrono::system_clock::now();            
        }

        void toc() {
            end = std::chrono::system_clock::now();
            std::chrono::duration<float> diff = end-start;
            fprintf(stderr, "%s %lf\n", s.c_str(), diff.count());            
        }
};

/*
    Comments:
    A helper function to assign a value to a pointer on GPU.
    Particularly helpful when we need to change a pointer on GPU
    from CPU (saves you from useless memcopies).
*/
template<typename T, typename U>
__global__ void assign(T * node, U value) {
    * node = value;
}

__global__ void bfs_frontier_kernel(MatrixCSR * matrix, bool * visited, int * distance, bool * frontier, int * new_found) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= matrix->rows) return;
    if(frontier[i]) {
        visited[i] = true;
        frontier[i] = false;
        * new_found = 1;
        for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
            int nj = matrix->j[j];
            if(!visited[nj]) {
                frontier[nj] = true;
                distance[nj] = distance[i] + 1;
            }
        }
    }
}

__global__ void get_useful_pairs(int n, int * paired_with, int * useful_pairs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n)
        return;
    if(paired_with[i] == -1) {
        useful_pairs[i] = 0;
    } else if(paired_with[i] < i) {
        useful_pairs[i] = 0;
    } else {
        useful_pairs[i] = 1;
    }
}

__global__ void gpu_prefix_sum_kernel(int n, int * useful_pairs) {
    for(int i = 1; i < n; i++) {
        useful_pairs[i] += useful_pairs[i - 1];
    }
}

void gpu_prefix_sum(int n, int * useful_pairs) {
    gpu_prefix_sum_kernel <<<1,1>>> (n, useful_pairs);
}

int * bfs(int n, MatrixCSR * matrix_gpu) {
    bool * visited;
    cudaMalloc(&visited, sizeof(bool) * n);

    int * distance;
    cudaMalloc(&distance, sizeof(int) * n);

    bool * frontier;
    cudaMalloc(&frontier, sizeof(int) * n);

    assign<<<1,1>>> (&frontier[0], 1);
    assign<<<1,1>>> (&distance[0], 0);

    int * new_found;
    cudaMallocManaged(&new_found, sizeof(int));

    do {
        * new_found = false;
        bfs_frontier_kernel <<<(n + 1024 - 1)/ 1024, 1024>>>(matrix_gpu, visited, distance, frontier, new_found);
        cudaDeviceSynchronize();
    } while(* new_found);

    return distance;

}

__global__ void mark_aggregations(int n, int * aggregations, int * useful_pairs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;
    int curr = useful_pairs[i];
    int prev = (i == 0) ? 0 : useful_pairs[i - 1];
    if(curr != prev) {
        aggregations[curr - 1] = i;
    }
}

__global__ void get_aggregations_count(int nc, int * aggregations, int * paired_with, int *aggregation_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nc) return;
    aggregation_count[i] = 1 + (aggregations[i] != paired_with[aggregations[i]]);
}

__global__ void create_p_matrix_transpose (int nc, int * aggregations, int * paired_with, int * aggregation_count, int * matrix_i, int * matrix_j, float * matrix_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nc) return;
    matrix_i[i + 1] = aggregation_count[i];

    int first = aggregations[i];
    int second = paired_with[first];

    int prev_sum = (i == 0) ? 0 : aggregation_count[i - 1];

    if(first == second) {
        matrix_j[prev_sum] = first;
        matrix_val[prev_sum] = 1;
    } else {
        matrix_j[prev_sum] = first;
        matrix_j[prev_sum + 1] = second;
        matrix_val[prev_sum] = 1;
        matrix_val[prev_sum + 1] = 1;
    }
}

template<typename T>
__global__ void print_gpu_variable_kernel(T * u) {
    printf("%d\n", u);
}


template<typename T>
void print_gpu_variable(T * u) {
    print_gpu_variable_kernel <<<1,1>>> (u);
    cudaDeviceSynchronize();
}

MatrixCSR * deepCopyMatrixCSRGPUtoCPU(const MatrixCSR * const gpu_matrix) {
    MatrixCSR * cpu_matrix = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    cudaMemcpy(cpu_matrix, gpu_matrix, sizeof(MatrixCSR), cudaMemcpyDeviceToHost);
    std::cout << cpu_matrix->rows << "--" << cpu_matrix->cols << "--" << cpu_matrix->nnz << std::endl;
    int * cpu_i = (int *) malloc(sizeof(int) * (cpu_matrix->rows + 1));
    int * cpu_j = (int *) malloc(sizeof(int) * (cpu_matrix->nnz));
    float * cpu_val = (float *) malloc(sizeof(float) * (cpu_matrix->nnz));

    cudaMemcpy(cpu_i, cpu_matrix->i, sizeof(int) * (cpu_matrix->rows + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_j, cpu_matrix->j, sizeof(int) * (cpu_matrix->nnz), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_val, cpu_matrix->val, sizeof(float) * (cpu_matrix->nnz), cudaMemcpyDeviceToHost);

    cpu_matrix->i = cpu_i;
    cpu_matrix->j = cpu_j;
    cpu_matrix->val = cpu_val;

    return cpu_matrix;
}

MatrixCSR * shallowCopyMatrixCSRGPUtoCPU(const MatrixCSR * const gpu_matrix) {
    MatrixCSR * cpu_matrix = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    cudaMemcpy(cpu_matrix, gpu_matrix, sizeof(MatrixCSR), cudaMemcpyDeviceToHost);
    return cpu_matrix;
}

MatrixCSR * deepCopyMatrixCSRCPUtoGPU(const MatrixCSR * const my_cpu) {
    MatrixCSR * cpu_matrix = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    memcpy(cpu_matrix, my_cpu, sizeof(MatrixCSR));
    std::cout << cpu_matrix->rows << "--" << cpu_matrix->cols << "--" << cpu_matrix->nnz << std::endl;
    MatrixCSR * gpu_matrix;
    cudaMalloc(&gpu_matrix, sizeof(MatrixCSR));

    int * gpu_i;
    int * gpu_j;
    float * gpu_val;

    cudaMalloc(&gpu_i, sizeof(int) * (cpu_matrix->rows + 1));
    cudaMalloc(&gpu_j, sizeof(int) * (cpu_matrix->nnz));
    cudaMalloc(&gpu_val, sizeof(float) * (cpu_matrix->nnz));

    cudaMemcpy(gpu_i, cpu_matrix->i, sizeof(int) * (cpu_matrix->rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_j, cpu_matrix->j, sizeof(int) * (cpu_matrix->nnz), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_val, cpu_matrix->val, sizeof(float) * (cpu_matrix->nnz), cudaMemcpyHostToDevice);

    cpu_matrix->i = gpu_i;
    cpu_matrix->j = gpu_j;
    cpu_matrix->val = gpu_val;

    cudaMemcpy(gpu_matrix, cpu_matrix, sizeof(MatrixCSR), cudaMemcpyHostToDevice);
    free(cpu_matrix);
    return gpu_matrix;
}

MatrixCSR * shallowCopyMatrixCSRCPUtoGPU(const MatrixCSR * const my_cpu) {
    MatrixCSR * cpu_matrix = (MatrixCSR *) malloc(sizeof(MatrixCSR));
    memcpy(cpu_matrix, my_cpu, sizeof(MatrixCSR));
    std::cout << cpu_matrix->rows << "--" << cpu_matrix->cols << "--" << cpu_matrix->nnz << std::endl;
    MatrixCSR * gpu_matrix;
    cudaMalloc(&gpu_matrix, sizeof(MatrixCSR));

    int * gpu_i;
    int * gpu_j;
    float * gpu_val;

    cudaMalloc(&gpu_i, sizeof(int) * (cpu_matrix->rows + 1));
    cudaMalloc(&gpu_j, sizeof(int) * (cpu_matrix->nnz));
    cudaMalloc(&gpu_val, sizeof(float) * (cpu_matrix->nnz));

    cudaMemcpy(gpu_i, cpu_matrix->i, sizeof(int) * (cpu_matrix->rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_j, cpu_matrix->j, sizeof(int) * (cpu_matrix->nnz), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_val, cpu_matrix->val, sizeof(float) * (cpu_matrix->nnz), cudaMemcpyHostToDevice);

    cpu_matrix->i = gpu_i;
    cpu_matrix->j = gpu_j;
    cpu_matrix->val = gpu_val;

    cudaMemcpy(gpu_matrix, cpu_matrix, sizeof(MatrixCSR), cudaMemcpyHostToDevice);
    free(cpu_matrix);
    return gpu_matrix;
}

void printCSRCPU(MatrixCSR * matrix) {
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
            std::cout << i + 1 << " " << matrix->j[j] + 1 << " " << matrix->val[j] << std::endl;
        }
    }
}

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