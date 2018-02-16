#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H
#include <vector>
#include <string>
#include "SparseVector.h"

class DenseVector;
class SparseVector;
class DenseMatrix;
class SparseMatrix;

class SparseMatrix {
    private:
        SparseMatrix * transpose_matrix;
        int rows, cols;
        std::vector<SparseVector> data;
        SparseMatrix * computeTranspose();
    public:
        SparseMatrix(std::vector<std::pair<std::pair<int, int>, double> > _data, int _rows, int _cols, SparseMatrix * _transpose=NULL);
        SparseMatrix(std::vector<SparseVector> _data, int _rows, int _cols);
        SparseMatrix(std::string s);
        SparseVector & operator[] (int index);
        SparseVector & getRowVector(int index);
        SparseVector & getColumnVector(int index);
        SparseMatrix transpose();
        void changed();
        SparseMatrix operator * (SparseMatrix matrix);
        SparseMatrix operator * (DenseMatrix matrix);
        DenseVector operator * (DenseVector vec);
        SparseMatrix operator * (SparseVector vec);
        SparseMatrix operator + (SparseMatrix matrix);
        DenseMatrix operator + (DenseMatrix matrix);
        double getRowColAbsSum(int index);
        double getRowColSum(int index);
        bool operator == (SparseMatrix matrix);
        void print();
        DenseMatrix toDenseMatrix();
        size_t row_size();
        size_t col_size();
        int nnz();
};

#endif