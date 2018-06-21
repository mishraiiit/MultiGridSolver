clean:
	rm -rf ./main
compile:
	./compile_mkl.sh main.cpp
	mv a.out main
	g++ -std=c++11 -msse4 -O4 -I ../../lib ../common/gridPlot.cpp -o ../common/gridPlot
