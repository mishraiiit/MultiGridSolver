clean:
	rm -rf ./main
compilefgmres:
	./compile_mkl.sh main.cpp
	cp a.out main
	g++ -std=c++11 -msse4 -O4 -I ../../lib ../common/gridPlot.cpp -o ../common/gridPlot
compilebicgstab:
	./compile_mkl.sh main_bicg.cpp
	cp a.out main_bicg
	g++ -std=c++11 -msse4 -O4 -I ../../lib ../common/gridPlot.cpp -o ../common/gridPlot
