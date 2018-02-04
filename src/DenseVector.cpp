#include "DenseVector.h"
#include <vector>
#include <assert.h>

DenseVector::DenseVector(int n) {
    size = n;
    for(int i = 0; i < n; i++) {
        _data.push_back(0);
    }
}

DenseVector DenseVector::operator+(DenseVector vec) {
    assert(size == vec.size);
    DenseVector result(size);
    for(int i = 0; i < size; i++) {
        result[i] = _data[i] + vec[i];
    }
    return result;
}

double DenseVector::operator*(DenseVector vec) {
    assert(size == vec.size);
    double answer = 0.0;
    for(int i = 0; i < size; i++) {
        answer += _data[i] * vec[i];
    }
    return answer;
}

double & DenseVector::operator[](int index) {
    return _data[index];
}

DenseMatrix DenseVector::toDenseMatrix() {
    DenseMatrix result(size, 1);
    for(int i = 0; i < size; i++) {
        result[i][0] = _data[i];
    }
    return result;
}

SparseVector DenseVector::toSparseVector() {
    std::vector<std::pair<int, double> > res_data;
    for(int i = 0; i < size; i++) {
        if(_data[i] != 0) {
            res_data.push_back({i, _data[i]});
        }
    }
    return SparseVector(size, res_data);
}

void DenseVector::print() {
    for(int i = 0; i < size; i++) {
        std::cout << _data[i] << " ";
    }
    std::cout << std::endl;
}

bool DenseVector::operator==(DenseVector B) {
    if(size != B.size)
        return false;
    for(int i = 0; i < size; i++) {
        if((*this)[i] != B[i])
            return false;
    }
    return true;
}