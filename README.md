
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
Started parallel.
Kernel functions written for Si and comptueRowColAbsSum.
Tested MIS in serial. Works.

Next steps:
Kernel for parallel matching.
Implement MIS in parallel. Read [here](http://on-demand.gputechconf.com/gtc/2017/presentation/s7286-martin-burtscher-a-high-quality-and-fast-maximal-independent-set-algorithm-for-gpus.pdf).
