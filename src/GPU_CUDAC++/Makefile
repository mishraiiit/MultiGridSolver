
main: main.cu Aggregation.cu MatrixIO.cu MatrixOperations.cu GPUDebug.cu TicToc.cpp PrefixSum.cu BFS.cu
	nvcc --expt-extended-lambda -std=c++11 -I ../../lib -I -lcusparse main.cu -o main
clean:
	rm -rf ./main
