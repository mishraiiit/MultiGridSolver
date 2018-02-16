#include "../SparseMatrix.h"
#include "../DenseMatrix.h"
#include <assert.h>
#include <string>
#include <map>
#include <set>

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

    // Test 3.
    {
        std::map<std::pair<int, int>, double> M;
        int N = 500;
        int entries = 10000;
        while(entries--) {
            M[{rand() % N, rand() % N}] = rand() % N;
        }
        SparseMatrix S(std::vector<std::pair<std::pair<int, int>, double> > (M.begin(), M.end()), N, N);
        assert(S * S == (S.toDenseMatrix() * S.toDenseMatrix()).toSparseMatrix());
    }
}

void testTranspose() {
    // Test 1.
    {
        std::map<std::pair<int, int>, double> M;
        int N = 500;
        int entries = 10000;
        while(entries--) {
            M[{rand() % N, rand() % N}] = rand() % N;
        }
        SparseMatrix S_SM(std::vector<std::pair<std::pair<int, int>, double> > (M.begin(), M.end()), N, N);
        DenseMatrix S_DM = S_SM.toDenseMatrix();
        assert(S_SM.transpose().toDenseMatrix() == S_DM.transpose());
    }
}

void testNnz() {
    // Test 1.
    {
        DenseMatrix A(10, 10);

        std::set<std::pair<int, int> > S;
        for(int t = 0; t < 20; t++) {
            int i = rand() % 10;
            int j = rand() % 10;
            A[i][j] = 1;
            S.insert({i, j});
        }
        assert(A.toSparseMatrix().nnz() == S.size());
    }
}

void testFileConstructor() {
    // Test 1.
    {
        std::string inp_file("sparse_matrix.txt");
        SparseMatrix T(inp_file);
        assert(T.row_size() == 4);
        assert(T.col_size() == 4);
        assert(T.nnz() == 6);
    }
}

int main() {
    testAddition();
    testMultiplication();
    testTranspose();
    testNnz();
    testFileConstructor();
    return 0;
}