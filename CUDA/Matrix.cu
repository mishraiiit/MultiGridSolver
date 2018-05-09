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

__host__ __device__ double getElementMatrixCSR(MatrixCSR * matrix, int i, int j) {
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

__host__ __device__ double getElementMatrixCSC(MatrixCSC * matrix, int i, int j) {
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

__global__ void comptueRowColumnAbsSum(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, double * output) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= matrix_csr->rows)
        return;


    int row_start = matrix_csr->i[id];
    int row_end = matrix_csr->i[id + 1];

    int col_start = matrix_csc->j[id];
    int col_end = matrix_csc->j[id + 1];

    double ans = 0;
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

    output[id] = ans;
    //printf("output[%d] : %lf\n", id, ans);
}

__global__ void comptueSi(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, double * output) {
int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= matrix_csr->rows)
        return;

    int row_start = matrix_csr->i[id];
    int row_end = matrix_csr->i[id + 1];

    int col_start = matrix_csc->j[id];
    int col_end = matrix_csc->j[id + 1];

    double ans = 0;
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

__host__ void comptueSiHost(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, double * output) {    
    for(int id = 0; id < matrix_csr->rows; id++) {

        int row_start = matrix_csr->i[id];
        int row_end = matrix_csr->i[id + 1];

        int col_start = matrix_csc->j[id];
        int col_end = matrix_csc->j[id + 1];

        double ans = 0;
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

__host__ __device__ double muij(int i, int j, MatrixCSR * matrix_csr, double * Si) {
    double aii = getElementMatrixCSR(matrix_csr, i, i);
    double ajj = getElementMatrixCSR(matrix_csr, j, j);
    double aij = getElementMatrixCSR(matrix_csr, i, j);
    double aji = getElementMatrixCSR(matrix_csr, j, i);

    double num = 2 * (1 / ((1 / aii) + (1 / ajj)));
    double den = (- (aij + aji) / 2) + 1 / ( ( 1 / (aii - Si[i])) + (1 / (ajj - Si[j])) );
    return num / den;
}

template<typename T>
__device__ void swap_variables(T & u, T & v) {
    T temp = u;
    u = v;
    v = temp;
}

__global__ void sortNeighbourList(MatrixCSR * matrix, MatrixCSR * neighbour_list, double * Si) {
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
}

__global__ void printNeighbourList(MatrixCSR * matrix, MatrixCSR * neighbour_list, double * Si) {

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

class TicToc {
    public:
        std::chrono::time_point<std::chrono::system_clock> start, end;
        string s;
        TicToc(string s) : s(s) {

        }
        void tic() {
            start = std::chrono::system_clock::now();            
        }

        void toc() {
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end-start;
            printf("%s %lf\n", s.c_str(), diff.count());            
        }
};