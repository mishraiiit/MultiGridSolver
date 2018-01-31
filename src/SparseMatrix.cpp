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
		data[_data[i].first.first].getData().push_back({_data[i].first.second, _data[i].second});
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