main:
	# Check if source file exists
	if [ ! -f main.cpp ]; then \
		echo "Error: main.cpp not found" && exit 1; \
	fi
	# Compile with Intel MKL
	g++ -std=c++11 -msse4 -O3 \
		main.cpp \
		-o main \
		-L/usr/lib/x86_64-linux-gnu \
		-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
		-lpthread -lm -ldl \
		-DMKL_ILP64 -m64 \
		-I/usr/include/mkl \
		-I../../lib || (echo "Error: Compilation failed" && exit 1)
	g++ -std=c++11 -msse4 -O4 -I ../../lib ../common/gridPlot.cpp -o ../common/gridPlot

clean:
	rm -rf ./main
