#include "../SparseVector.h"
#include <stdlib.h>
#include <vector>
#include "assert.h"

void testSize() {
	// Test 1.
	{
		SparseVector vec(10, {});
		assert(vec.size == 10);
	}
}

void testGetData() {
	// Test 1.
	{
		std::vector <std::pair<int, double> > data = {{2, 3}, {3, 4}, {5, 7}};
		SparseVector vec(10, data);
		assert(vec.getData() == data);
	}

	// Test 2.
	{
		std::vector <std::pair<int, double> > data = {{2, 3}, {3, 4}, {5, 7}};
		SparseVector vec(10, data);
		vec.getData() = {};
	}
}

void testRandomAccessOperator() {
	// Test 1.
	{
		std::vector <std::pair<int, double> > data = {{2, 3}, {3, 4}, {5, 7}};
		SparseVector vec(10, data);
		assert(vec[0] == 0);
		assert(vec[1] == 0);
		assert(vec[2] == 3);
		assert(vec[3] == 4);
		assert(vec[5] == 7);
		assert(vec[6] == 0);
		assert(vec[7] == 0);
		assert(vec[8] == 0);
		assert(vec[9] == 0);
	}
}

void testAddition() {
	// Test 1.
	{
		SparseVector A(10, {{0, 2}, {2, 3}, {8, 8}, {9, 2}});
		SparseVector B(10, {{0, 2}, {1, 2}, {5, 1}, {8, 1}});
		SparseVector C = A + B;
		assert(A + B == C);
		assert(A.toDenseVector() + B.toDenseVector() == C.toDenseVector());
	}
}

void testToDenseVector() {
	// Test 1.
	{
		SparseVector A(10, {{0, 2}, {2, 3}, {8, 8}, {9, 2}});
		DenseVector B = A.toDenseVector();
		for(int i = 0; i < 10; i++) {
			assert(A[i] == B[i]);
		}
	}

	// Test 2.
	{
		SparseVector A(10, {{0, 2}, {1, 2}, {5, 1}, {8, 1}});
		DenseVector B = A.toDenseVector();
		for(int i = 0; i < 10; i++) {
			assert(A[i] == B[i]);
		}
	}
}



int main() {

	testSize();
	testGetData();
	testRandomAccessOperator();
	testAddition();
	testToDenseVector();

	return 0;
}