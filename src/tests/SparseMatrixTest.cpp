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
		auto range_random = [](int l, int r) {
			assert(l <= r);
			return l + (rand() % (r - l + 1));
		};

		auto range_random_skew = [range_random](int l, int r) {
			assert(l <= r);
			return range_random(l, range_random(l, range_random(l, r)));
		};

		const int N = 500;
		int max_size = 0;
		SparseMatrix A(std::vector<std::pair<std::pair<int, int>, double> > (), N, N);
		for(int t = 0; t < N; t++) {
			int index = rand() % N;
			std::vector<std::pair<int, double> > & current_row = A[index].getData();
			int last_index = -1;
			if(!current_row.empty()) {
				last_index = current_row.back().first;
			}
			int other_index = range_random_skew(last_index + 1, N - 1);	
			current_row.push_back({other_index, rand() % 100});
			if(current_row.size() > max_size)
				max_size = current_row.size();
		}
		assert((A.toDenseMatrix() * A.toDenseMatrix()).toSparseMatrix() == A * A);
	}
}

int main() {

	testAddition();
	testMultiplication();
	
}