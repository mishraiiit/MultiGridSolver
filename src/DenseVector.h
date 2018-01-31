#ifndef DENSE_VECTOR_H
#define DENSE_VECTOR_H
#include <vector>
#include "DenseMatrix.h"
#include "SparseVector.h"

class DenseVector;
class SparseVector;
class DenseMatrix;
class SparseMatrix;

class DenseVector {
private:
		std::vector<double> _data;
	public:		
		int size;
		DenseVector(int n);
		DenseVector operator+(DenseVector vec);
		double operator*(DenseVector vec);
		double & operator[](int index);
		DenseMatrix toDenseMatrix();
		SparseVector toSparseVector();
};

#endif