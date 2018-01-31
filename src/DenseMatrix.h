#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H
#include <vector>
#include "DenseVector.h"

class DenseVector;
class SparseVector;
class DenseMatrix;
class SparseMatrix;

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
		DenseMatrix transpose();
		DenseVector toDenseVector();
		SparseMatrix toSparseMatrix();
};

DenseMatrix operator* (int number, DenseMatrix D);

#endif