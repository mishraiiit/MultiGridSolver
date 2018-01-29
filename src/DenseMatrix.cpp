#include "DenseMatrix.h"
#include <vector>
#include <assert.h>

DenseMatrix::DenseMatrix(int n, int m) {
	rows = n;
	cols = m;
	_matrix = std::vector<std::vector<double> > (n, std::vector<double> (m, 0));
}

std::vector<double> & DenseMatrix::operator[] (int row) {
	assert(row < rows);
	return _matrix[row];
}

DenseMatrix DenseMatrix::operator * (DenseMatrix matrix) {
	assert(cols == matrix.rows);
	DenseMatrix result(rows, matrix.cols);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < matrix.cols; j++) {
			double & ans = result[i][j];
			for(int k = 0; k < cols; k++) {
				ans +=_matrix[i][k] * matrix._matrix[k][j];
			}
		}
	}
	return result;
}

DenseMatrix DenseMatrix::operator + (DenseMatrix matrix) {
	assert(rows == matrix.rows);
	assert(cols == matrix.cols);
	DenseMatrix result(rows, cols);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			result[i][j] = _matrix[i][j] + matrix._matrix[i][j];
		}
	}
	return result;
}

DenseMatrix DenseMatrix::operator * (int number) {
	DenseMatrix result = * this;
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			result[i][j] = result[i][j] * number;
		}
	}
	return result;
}

DenseMatrix operator* (int number, DenseMatrix D) {
	return D * number;
}

DenseMatrix DenseMatrix::transform () {
	DenseMatrix result = * this;
	for(int i = 0; i < rows; i++) {
		for(int j = i + 1; j < cols; j++) {
			result[i][j] = result[j][i];
		}
	}
	return result;
}