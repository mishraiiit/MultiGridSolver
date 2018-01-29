#include <vector>

#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

class DenseMatrix {
private:
		std::vector<std::vector<double> > _matrix;
	public:
		int rows, cols;
		DenseMatrix(int n, int m);
		std::vector<double> & operator[] (int row);
		DenseMatrix operator + (DenseMatrix matrix);
		DenseMatrix operator * (DenseMatrix matrix);
		DenseMatrix operator * (int number);
		DenseMatrix transform();
};

DenseMatrix operator* (int number, DenseMatrix D);

#endif