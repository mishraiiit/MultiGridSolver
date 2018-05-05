#include <assert.h>
#include "bits/stdc++.h"
using namespace std;
#include <vector>

struct MatrixCOOUnsorted {
    int rows, cols, nnz;
    int * i, * j;
    double * val;
};

MatrixCOOUnsorted * readMatrix(string filename) {
    std::ifstream fin(filename);
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    matrix_coo->rows = M;
    matrix_coo->cols = N;
    matrix_coo->val = L;
    fin >> M >> N >> L;
    // Read the data

    MatrixCOOUnsorted * matrix_coo;
    cudaMallocManaged(&matrix_coo, sizeof(MatrixCOOUnsorted));

    cudaMallocManaged(&matrix_coo->i, sizeof(int) * L);
    cudaMallocManaged(&matrix_coo->i, sizeof(int) * L);
    cudaMallocManaged(&matrix_coo->j, sizeof(int) * L);
    cudaMallocManaged(&matrix_coo->val, sizeof(double) * L);

    int filled = 0;

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
    return matrix_coo;
}