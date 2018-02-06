#include "../SparseMatrix.h"
#include "../DenseMatrix.h"
#include <assert.h>

void testAddition() {
	// Test 1.
	{
		DenseMatrix A(10, 10), B(10, 10);
		for(int t = 0; t < 20; t++) {
			A[rand() % 10][rand() % 10] = rand() % 100;
			B[rand() % 10][rand() % 10] = rand() % 100;
		}
		assert(A.toSparseMatrix() + B.toSparseMatrix() == (A + B).toSparseMatrix());
	}
}

void testMultiplication() {
	// Test 1.
	{
		DenseMatrix A(10, 10), B(10, 10);
		for(int t = 0; t < 60; t++) {
			A[rand() % 10][rand() % 10] = rand() % 100;
			B[rand() % 10][rand() % 10] = rand() % 100;
		}
		assert(A.toSparseMatrix() * B.toSparseMatrix() == (A * B).toSparseMatrix());
	}

	// Test 2.
	{
		DenseMatrix A(100, 100), B(100, 100);
		for(int t = 0; t < 600; t++) {
			A[rand() % 10][rand() % 10] = rand() % 100;
			B[rand() % 10][rand() % 10] = rand() % 100;
		}
		assert(A.toSparseMatrix() * B.toSparseMatrix() == (A * B).toSparseMatrix());
	}
	
}

int main() {

	testAddition();
	testMultiplication();
	
}