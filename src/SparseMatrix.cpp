#include "SparseMatrix.h"
#include "Utility.h"
#include <utility>
#include <vector>
#include <fstream>
#include <algorithm>
#include <assert.h>

SparseMatrix::SparseMatrix(std::vector<std::pair<std::pair<int, int>, double> > _data, int _rows, int _cols, SparseMatrix * _transpose) {
    rows = _rows;
    cols = _cols;
    transpose_matrix = _transpose;
    data = std::vector<SparseVector> (rows, SparseVector(cols, {}));
    int last_row = -1;
    std::vector<std::pair<int, double> > row_data;
    for(int i = 0; i < _data.size(); i++) {
        if(_data[i].second != 0)
            data[_data[i].first.first].getData().push_back({_data[i].first.second, _data[i].second});
    }
    if(transpose_matrix == NULL) {
        transpose_matrix = computeTranspose();
    }
}

SparseMatrix::SparseMatrix(std::string s) {
    std::ifstream fin(s);
    if(fin.fail()) {
        std::cout << "File " << s << " not found.";
        exit(1);
    }
    int _rows, _cols, _nnz;
    fin >> _rows >> _cols >> _nnz;

    std::vector<std::pair<std::pair<int, int>, double> > _data;
    for(int i = 0; i < _nnz; i++) {
        int x, y;
        double value;
        fin >> x >> y >> value;
        x--; y--;
        _data.push_back({{x, y}, value});
    }
    std::sort(_data.begin(), _data.end());
    rows = _rows;
    cols = _cols;
    transpose_matrix = NULL;
    data = std::vector<SparseVector> (rows, SparseVector(cols, {}));
    int last_row = -1;
    std::vector<std::pair<int, double> > row_data;
    for(int i = 0; i < _data.size(); i++) {
        if(_data[i].second != 0)
            data[_data[i].first.first].getData().push_back({_data[i].first.second, _data[i].second});
    }
    if(transpose_matrix == NULL) {
        transpose_matrix = computeTranspose();
    }
}

void SparseMatrix::changed() {
    transpose_matrix = computeTranspose();
}

SparseMatrix::SparseMatrix(std::vector<SparseVector> _data, int _rows, int _cols) {
    assert(_data.size() == _rows);
    rows = _rows;
    cols = _cols;
    data = _data;
    for(int i = 0; i < rows; i++) {
        assert(data[i].size == cols);
    }
    if(transpose_matrix == NULL) {
        transpose_matrix = computeTranspose();
    }
}

SparseMatrix SparseMatrix::transpose() {
    assert(transpose_matrix != NULL);
    return * transpose_matrix;
}

SparseVector & SparseMatrix::operator[] (int index) {
    return data[index];
}

SparseVector & SparseMatrix::getRowVector(int index) {
    assert(index < data.size());
    assert(data[index].size == col_size());
    return data[index];
}

SparseVector & SparseMatrix::getColumnVector(int index) {
    assert(index < transpose_matrix->data.size());
    assert(transpose_matrix->data[index].size == row_size());
    assert(transpose_matrix != NULL);
    return transpose_matrix->getRowVector(index);
}

SparseMatrix SparseMatrix::operator * (SparseMatrix matrix) {
    assert(cols == matrix.rows);
    matrix = matrix.transpose();
    std::vector<std::pair<std::pair<int, int>, double> > new_data;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < matrix.rows; j++) {
            double temp = (*this)[i] * matrix[j];
            if(temp != 0)
                new_data.push_back({{i, j}, temp});
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

SparseMatrix * SparseMatrix::computeTranspose() {
    std::vector<std::pair<std::pair<int, int>, double> > new_data;
    for(int i = 0; i < data.size(); i++) {
        std::vector<std::pair<int, double> > & row_data = data[i].getData();
        for(std::pair<int, double> elem : row_data) {
            new_data.push_back({{elem.first, i}, elem.second});
        }
    }
    sort(new_data.begin(), new_data.end());
    return new SparseMatrix(new_data, cols, rows, this);
}

double SparseMatrix::getRowColAbsSum(int index) {
    auto & row_data = getRowVector(index).getData();
    auto & col_data = getRowVector(index).getData();
    double sum = 0;
    std::vector<std::pair<int, std::pair<double, double> > > union_lis = \
        union_list(row_data, col_data);
    for(std::pair<int, std::pair<double, double> > index_info : union_lis) {
        if(index_info.first != index)
            sum += abs(index_info.second.first + index_info.second.second) / 2.0;
    }
    return sum;
}

double SparseMatrix::getRowColSum(int index) {

    return ((getRowVector(index).sum() + getColumnVector(index).sum()) / 2.0) - operator[](index)[index];
}

int SparseMatrix::nnz() {
    int ans = 0;
    for(int i = 0; i < rows; i++) {
        ans += data[i].nnz();
    }
    return ans;
}
