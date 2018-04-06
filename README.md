
# MultiGrid Solver

This project implements the algorithm given [here](https://github.com/mishraiiit/MultiGridSolver/blob/master/docs/AGMG_For_Convection_Diffusion.pdf)  in serial.

Steps to run:

1) Go to src directory and make.

    cd ./src; make;

2) `./run.sh matrixname kappatg npass tou`
Here the matrixname should be present in the matrices directory in the .mtx (matrix market) format. kappatg, npass and tou are the parameters given in the paper.

Status :
Serial implementation is completed (uses Eigen library).

Currently working:
Effects of CMK on the final prolongation matrix.

Next steps:
Start parallel implementation.
