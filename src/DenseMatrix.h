#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H
#include <vector>
#include <iostream>
#include "DenseVector.h"

// Forward declarations.
class DenseVector;
class SparseVector;
class DenseMatrix;
class SparseMatrix;

class DenseMatrix {
    private:
        std::vector<std::vector<double> > _matrix;
    public:
        int rows;
        int cols;
        DenseMatrix(int n, int m);
        std::vector<double> & operator[] (int row);
        DenseMatrix operator + (DenseMatrix matrix);
        bool operator== (DenseMatrix matrix);
        DenseMatrix operator * (DenseMatrix matrix);
        DenseMatrix operator * (int number);
        DenseMatrix transpose();
        SparseMatrix toSparseMatrix();
        void print();
};

DenseMatrix operator* (int number, DenseMatrix D);

#endif