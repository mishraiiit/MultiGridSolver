main: main.cu Aggregation.cu MatrixIO.cu MatrixOperations.cu GPUDebug.cu TicToc.cpp PrefixSum.cu BFS.cu
	nvcc --expt-extended-lambda -std=c++11 -I ../../lib main.cu -o main -lcusparse

test_matrix: test_matrix_operations.cu MatrixIO.cu MatrixOperations.cu GPUDebug.cu MatrixAccess.cu
	nvcc --expt-extended-lambda -std=c++11 -I ../../lib test_matrix_operations.cu -o test_matrix -lcusparse

clean:
	rm -rf ./main ./test_matrix
