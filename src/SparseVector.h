#ifndef SPARSE_VECTOR_H
#define SPARSE_VECTOR_H

class DenseVector;
class SparseVector;
class DenseMatrix;
class SparseMatrix;

class SparseVector {
	private:
		std::vector<std::pair<int, double> > _data;	
	public:
		int size;
		SparseVector(int size, std::vector<std::pair<int, double> > data);
		double operator[] (int index);
		double operator* (SparseVector vec);
		double operator* (DenseVector vec);
};

#endif