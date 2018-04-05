#include <iostream>
#include <typeinfo>
#include <Eigen/Sparse>
#include <bits/stdc++.h>
#include <deque>
#include <string>
using namespace std;
using namespace Eigen;

typedef SparseMatrix<double, RowMajor> SMatrix;

SMatrix readMatrix(string filename) {
    std::ifstream fin(filename);
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;
    SMatrix matrix(M, N);
    // Read the data
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
            matrix.insert(i, tp.first) = tp.second;
        }
    }
    fin.close();
    cerr << "Read matrix from file: " << filename << endl;
    return matrix;
}

void writeMatrix(string filename, SMatrix matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    int nnz = matrix.nonZeros();
    ofstream fout;
    fout.open(filename);
    fout << "%%MatrixMarket matrix coordinate real general " << endl;
    fout << rows << " " << cols << " " << nnz << endl;
    for(int i = 0; i < rows; i++) {
        SparseVector<double> row = matrix.row(i);
        for(SparseVector<double>::InnerIterator it(row); it; ++it) {
            int j = it.index();
            double value = it.value();
            fout << i + 1 << " " << j + 1 << " " << value << endl;
            // Adding 1 for exporting to .mtx format.
        }
    }
    fout.close();
}
