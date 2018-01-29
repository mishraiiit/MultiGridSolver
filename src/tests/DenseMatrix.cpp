#include "../DenseMatrix.h"
#include <stdlib.h>
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

int main() {
	testAddition();
	testMultiplication();
}