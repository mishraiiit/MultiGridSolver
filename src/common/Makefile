bicg:
	g++ -std=c++11 bicg.cpp -o bicg -msse4 -O4 \
		-I/usr/include/eigen3 \
		-L/usr/lib/x86_64-linux-gnu \
		-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
		-lpthread -lm -ldl \
		-DMKL_ILP64 -m64 \
		-I/usr/include/mkl

clean:
	rm -rf ./poisson ./bicg

poisson:
	g++ -std=c++11 poisson.cpp -o poisson -msse4 -O4