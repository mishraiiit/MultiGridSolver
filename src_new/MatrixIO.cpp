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
    for (int l = 0; l < L; l++) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        matrix.insert(m - 1, n - 1) = data;
    }
    fin.close();
    return matrix;
}

void writeMatrix(string filename, SMatrix matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    int nnz = matrix.nonZeros();
    ofstream fout;
    fout.open(filename);
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