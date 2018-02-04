#include "../DenseVector.h"
#include <stdlib.h>
#include <vector>
#include "assert.h"

void testSize() {
    // Test 1.
    {
        DenseVector vec(4);
        assert(vec.size == 4);
    }

    // Test 2.
    {
        DenseVector vec(1);
        assert(vec.size == 1);
    }
}

void testAddition() {
    // Test 1.
    {
        DenseVector A(4), B(4), C(4);
        std::vector<int> C_vec(4);
        for(int i = 0; i < 4; i++) {
            A[i] = rand() % 100;
            B[i] = rand() % 100;
            C_vec[i] = A[i] + B[i];
        }
        C = A + B;
        for(int i = 0; i < 4; i++) {
            assert(C[i] == C_vec[i]);
        }
    }
}

void testMultiplication() {
    // Test 1.
    {
        DenseVector A(4), B(4);
        double result = 0;
        for(int i = 0; i < 4; i++) {
            A[i] = rand() % 100;
            B[i] = rand() % 100;
            result += A[i] * B[i];
        }
        assert(A * B == result);
    }
}

void testRandomAccessOperator() {
    // Test 1.
    {
        DenseVector A(4);
        std::vector<int> B(4);
        for(int i = 0; i < 4; i++) {
            B[i] = rand() % 100;
            A[i] = B[i];
        }
        for(int i = 0; i < 4; i++) {
            assert(A[i] == B[i]);
        }
    }
}

void testToDenseMatrix() {
    // Test 1.
    {
        DenseVector A(4);
        for(int i = 0; i < 4; i++) {
            A[i] = rand() % 100;
        }
        DenseMatrix B = A.toDenseMatrix();
        for(int i = 0; i < 4; i++) {
            assert(B[i][0] == A[i]);
        }
    }
}

int main() {
    
    testSize();
    testAddition();
    testMultiplication();
    testRandomAccessOperator();
    testToDenseMatrix();

    return 0;
}