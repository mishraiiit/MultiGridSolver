#/bin/bash
/usr/local/bin/mex -f ./myopts.sh -O -largeArrayDims dmtlagtwolev.c dagtwolev_mex.f90 -lmwlapack -lmwblas
