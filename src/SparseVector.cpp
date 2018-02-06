#include "SparseVector.h"
#include "Utility.h"
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
    assert(index < size);
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
    }
    if(_data[l].first > index)
        return 0;
    return _data[l].second;
}

SparseVector SparseVector::operator+ (SparseVector vec) {
    assert(size == vec.size);
    std::vector<std::pair<int, double> > res_data;
    std::vector<std::pair<int, std::pair<double, double> > > union_lis = \
        union_list(getData(), vec.getData());
    for(std::pair<int, std::pair<double, double> > index_info : union_lis) {
        res_data.push_back({index_info.first, index_info.second.first + index_info.second.second});
    }
    return SparseVector(size, res_data);
}

double SparseVector::operator* (SparseVector vec) {
    assert(size == vec.size);
    double result = 0;
    std::vector<std::pair<int, std::pair<double, double> > > intersection = \
        intersection_list(getData(), vec.getData());
    for(std::pair<int, std::pair<double, double> > index_info : intersection) {
        result += index_info.second.first * index_info.second.second;
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