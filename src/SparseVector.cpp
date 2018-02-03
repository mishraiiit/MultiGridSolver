#include "SparseVector.h"
#include "DenseVector.h"
#include <assert.h>

SparseVector::SparseVector(int size, std::vector<std::pair<int, double> > data) : size(size) {
    _data = data;
    int last = -1;
    for(int i = 0; i < data.size(); i++) {
        assert(data[i].first > last);
        last = data[i].first;
        _abs_sum += abs(data[i].second);
        _sum += data[i].second;
    }
}

std::vector<std::pair<int, double> > & SparseVector::getData() {
    return _data;
}

double SparseVector::operator[] (int index) {
    int l = 0, r = _data.size() - 1;
    if(_data[r].first < index)
        return 0;
    while(l != r) {
        int mid = (l + r) / 2;
        if(_data[mid].first < index) {
            l = mid + 1;
        } else {
            r = mid;
        }
        if(_data[l].first > index)
            return 0;
        return _data[l].second;
    }
}

SparseVector SparseVector::operator+ (SparseVector vec) {
    assert(size == vec.size);
    std::vector<std::pair<int, double> > res_data;
    int i = 0, j = 0;
    while(i < _data.size() && j < vec._data.size()) {
        if(_data[i].first < vec._data[j].first) {
            res_data.push_back(_data[i]);
            i++;
        } else if(_data[i].first > vec._data[j].first) {
            res_data.push_back(vec._data[j]);
            j++;
        } else {
            res_data.push_back({_data[i].first, _data[i].second + vec._data[j].second});
            i++; j++;
        }
    }
    while(i < _data.size()) {
        res_data.push_back(_data[i]);
        i++;
    }
    while(j < vec._data.size()) {
        res_data.push_back(vec._data[j]);
        j++;
    }
    return SparseVector(size, res_data);
}

double SparseVector::operator* (SparseVector vec) {
    assert(size == vec.size);
    double result = 0;
    int i = 0, j = 0;
    while(i < _data.size() && j < vec._data.size()) {
        if(_data[i].first < vec._data[j].first) {
            i++;
        } else if(_data[i].first > vec._data[j].first) {
            j++;
        } else {
            result += _data[i].second * vec._data[j].second;
            i++; j++;
        }
    }
    return result;
}

double SparseVector::operator* (DenseVector vec) {
    return (*this) * vec.toSparseVector();
}

bool SparseVector::operator == (SparseVector vec) {
    return (size == vec.size && _data == vec._data);
}

bool SparseVector::operator != (SparseVector vec) {
    return !((*this) == vec);
}

void SparseVector::print() {
    this->toDenseVector().print();
}

DenseVector SparseVector::toDenseVector() {
    DenseVector result(size);
    for(std::pair<int, double> elem : _data) {
        result[elem.first] = elem.second;
    }
    return result;
}

double SparseVector::abs_sum() {
    return _abs_sum;
}

double SparseVector::sum() {
    return _sum;
}