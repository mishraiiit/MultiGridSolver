#include "../DenseMatrix.h"
#include <stdlib.h>
#include <vector>
#include "assert.h"

void testAddition() {
	// Test 1.
	{
		DenseMatrix A(4, 4);
		DenseMatrix B(4, 4);
		DenseMatrix C(4, 4);

		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				A[i][j] = rand() % 10;
				B[i][j] = rand() % 10;
				C[i][j] = A[i][j] + B[i][j];
			}
		}

		DenseMatrix D = A + B;
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				assert(C[i][j] == D[i][j]);
			}
		}
	}
}

void testMultiplication() {
	// Test 1.
	{
		DenseMatrix A(4, 4);
		DenseMatrix B(4, 4);
		DenseMatrix C(4, 4);

		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				A[i][j] = rand() % 10;
				B[i][j] = rand() % 10;
			}
		}

		C = A * B;
		for(int i = 0; i < 4; i++) {
			for(int k = 0; k < 4; k++) {
				double ans = 0;
				for(int j = 0; j < 4; j++) {
					ans = ans + A[i][j] * B[j][k];
				}
				assert(C[i][k] == ans);
			}
		}
	}
}

void testRandomRowAccess() {
	// Test 1.
	{
		std::vector<double> _row;
		for(int i = 0; i < 4; i++) {
			_row.push_back(rand() % 100);
		}
		DenseMatrix matrix(4, 4);
		for(int i = 0; i < 4; i++) {
			matrix[0][i] = _row[i];
		}
		std::vector<double> & _rec = matrix[0];
		for(int i = 0; i < 4; i++) {
			assert(_rec[i] == _row[i]);
		}
	}
}

void testMultiplicationDenseMatrixNumber() {
	// Test 1.
	{
		DenseMatrix matrix(4, 4);
		const int multiplier = 4;
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				matrix[i][j] = rand() % 100;
			}
		}
		DenseMatrix result = matrix * multiplier;
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				assert(result[i][j] == matrix[i][j] * multiplier);
			}
		}
	}
}

void testMultiplicationNumberDenseMatrix() {
	// Test 1.
	{
		DenseMatrix matrix(4, 4);
		const int multiplier = 4;
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				matrix[i][j] = rand() % 100;
			}
		}
		DenseMatrix result = multiplier * matrix;
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				assert(result[i][j] == matrix[i][j] * multiplier);
			}
		}
	}
}


int main() {
	testAddition();
	testMultiplication();
	testRandomRowAccess();
	testMultiplicationDenseMatrixNumber();
	testMultiplicationNumberDenseMatrix();
}