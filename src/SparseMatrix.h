#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

class DenseVector;
class SparseVector;
class DenseMatrix;
class SparseMatrix;

class SparseMatrix {
	private:
		int row, col;
		std::vector<std::pair<std::pair<int, int>, double> > _data;
	public:
		SparseMatrix(std::vector<std::pair<std::pair<int, int>, double> > _data);
		SparseMatrix operator * (SparseMatrix matrix);
		SparseMatrix operator * (DenseMatrix matrix);
		DenseVector operator * (DenseVector vec);
		SparseMatrix operator * (SparseVector vec);
		SparseMatrix operator + (SparseMatrix matrix);
		DenseMatrix operator + (DenseMatrix matrix);
};

#endif