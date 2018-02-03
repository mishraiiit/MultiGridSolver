#include "SparseMatrix.h"
#include <utility>
#include <vector>
#include <algorithm>
#include <assert.h>

SparseMatrix::SparseMatrix(std::vector<std::pair<std::pair<int, int>, double> > _data, int _rows, int _cols) {
    rows = _rows;
    cols = _cols;
    data = std::vector<SparseVector> (rows, SparseVector(cols, {}));
    int last_row = -1;
    std::vector<std::pair<int, double> > row_data;
    for(int i = 0; i < _data.size(); i++) {
        if(_data[i].second != 0)
            data[_data[i].first.first].getData().push_back({_data[i].first.second, _data[i].second});
    }
}

SparseMatrix::SparseMatrix(std::vector<SparseVector> _data, int _rows, int _cols) {
    rows = _rows;
    cols = _cols;
    data = _data;
    for(int i = 0; i < rows; i++) {
        assert(data[i].size == cols);
    }
}

SparseMatrix SparseMatrix::transpose() {
    std::vector<std::pair<std::pair<int, int>, double> > new_data;
    for(int i = 0; i < data.size(); i++) {
        std::vector<std::pair<int, double> > & row_data = data[i].getData();
        for(std::pair<int, double> elem : row_data) {
            new_data.push_back({{elem.first, i}, elem.second});
        }
    }
    sort(new_data.begin(), new_data.end());
    return SparseMatrix(new_data, cols, rows);
}

SparseVector & SparseMatrix::operator[] (int index) {
    return data[index];
}

SparseMatrix SparseMatrix::operator * (SparseMatrix matrix) {
    assert(cols == matrix.rows);
    matrix = matrix.transpose();
    std::vector<std::pair<std::pair<int, int>, double> > new_data;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < matrix.rows; j++) {
            new_data.push_back({{i, j}, (*this)[i] * matrix[j]});
        }
    }
    return SparseMatrix(new_data, rows, matrix.rows);
}

DenseMatrix SparseMatrix::toDenseMatrix() {
    DenseMatrix matrix(rows, cols);
    for(int i = 0; i < rows; i++) {
        std::vector<std::pair<int, double> > & row_data = data[i].getData();
        for(std::pair<int, double> elem : row_data) {
            matrix[i][elem.first] = elem.second;
        }
    }
    return matrix;
}

bool SparseMatrix::operator==(SparseMatrix matrix) {
    if(rows != matrix.rows) return false;
    for(int i = 0; i < rows; i++) {
        if(matrix[i] != (*this)[i])
            return false;
    }
    return true;
}

SparseMatrix SparseMatrix::operator+(SparseMatrix matrix) {
    assert(rows == matrix.rows && cols == matrix.cols);
    std::vector<SparseVector> data(rows, SparseVector(cols, {}));
    for(int i = 0; i < rows; i++) {
        data[i] = this->operator[](i) + matrix[i];
    }
    return SparseMatrix(data, rows, cols);
}

DenseMatrix SparseMatrix::operator + (DenseMatrix matrix) {
    return this->toDenseMatrix() + matrix;
}

void SparseMatrix::print() {
    for(int i = 0; i < rows; i++) {
        data[i].print();
    }
}

size_t SparseMatrix::row_size() {
    return rows;
}

size_t SparseMatrix::col_size() {
    return cols;
}