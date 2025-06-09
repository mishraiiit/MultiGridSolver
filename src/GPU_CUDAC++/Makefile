
main: main.cu Aggregation.cu MatrixIO.cu MatrixOperations.cu GPUDebug.cu TicToc.cpp PrefixSum.cu BFS.cu
	nvcc --expt-extended-lambda -std=c++11 -I ../../lib main.cu -o main -lcusparse
clean:
	rm -rf ./main
