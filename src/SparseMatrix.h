#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H
#include <vector>
#include "SparseVector.h"

class DenseVector;
class SparseVector;
class DenseMatrix;
class SparseMatrix;

class SparseMatrix {
    private:
        int rows, cols;
        std::vector<SparseVector> data;
    public:
        SparseMatrix(std::vector<std::pair<std::pair<int, int>, double> > _data, int _rows, int _cols);
        SparseMatrix(std::vector<SparseVector> _data, int _rows, int _cols);
        SparseVector & operator[] (int index);
        SparseMatrix operator * (SparseMatrix matrix);
        SparseMatrix operator * (DenseMatrix matrix);
        DenseVector operator * (DenseVector vec);
        SparseMatrix operator * (SparseVector vec);
        SparseMatrix operator + (SparseMatrix matrix);
        DenseMatrix operator + (DenseMatrix matrix);
        bool operator == (SparseMatrix matrix);
        void print();
        DenseMatrix toDenseMatrix();
        SparseMatrix transpose();
        size_t row_size();
        size_t col_size();
};

#endif