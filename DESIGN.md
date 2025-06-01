# Introduction

This document outlines the design of a software project aimed at solving large sparse linear systems of equations using Algebraic Multigrid (AMG) methods. The primary goal of this project is to provide efficient and scalable AMG solvers.

The repository hosts implementations of the AMG solvers in three different programming environments:

*   **C++:** A high-performance version suitable for production environments.
*   **CUDA:** A version optimized for NVIDIA GPUs, leveraging parallel computing capabilities.
*   **MATLAB:** A version for rapid prototyping, algorithm development, and visualization.

This design document will detail the common architectural components, the specific design choices for each implementation, and the overall project structure.

# Algorithm

The specific Algebraic Multigrid (AMG) algorithm implemented in this project is based on the methods and techniques described in the document `docs/AGMG_For_Convection_Diffusion.pdf`. Readers are encouraged to consult this document for a detailed theoretical background and algorithmic specifics.

Conceptually, a typical AMG algorithm involves the following main steps, which are iteratively applied to create and navigate a hierarchy of coarser representations of the original linear system:

*   **Coarsening or Aggregation:** This step involves selecting a subset of the fine-grid variables to serve as variables on the next coarser grid. The selection process is crucial and is typically based on the strength of connections between variables in the system matrix. This reduces the size of the problem at each subsequent level.

*   **Prolongation/Interpolation:** This operator defines how solution values (or corrections) from a coarser grid are mapped (interpolated) to the corresponding fine grid. The design of the prolongation operator is critical for the efficiency of the multigrid cycle, ensuring that coarse-grid corrections effectively reduce fine-grid errors.

*   **Restriction:** This operator transfers the residual (the error in the current approximation) from a fine grid to the next coarser grid. It is often related to the transpose of the prolongation operator.

*   **Solver on Coarsest Grid:** Once the coarsening process reaches a grid that is sufficiently small, the linear system on this coarsest level is typically solved using a direct solver (e.g., LU or Cholesky decomposition) or a robust iterative solver.

*   **Smoothing:** On each grid level, before and/or after the coarse-grid correction step, a few iterations of a simple iterative solver (e.g., Gauss-Seidel, Jacobi, or a weighted Jacobi method) are applied. The purpose of smoothing is to reduce the high-frequency components of the error, which are not effectively handled by the coarse-grid correction.

The interplay between these components, cycling through different grid levels (e.g., V-cycle, W-cycle), allows AMG methods to achieve fast convergence rates.

# Code Structure

The repository is organized into several top-level directories:

*   `src/`: Contains the source code for the different AMG implementations.
*   `lib/`: Contains third-party libraries used by the project.
*   `matrices/`: Contains sample sparse matrices for testing and benchmarking.
*   `docs/`: Contains documentation, including the theoretical basis for the implemented AMG algorithm.

## Source Code (`src/`)

The `src/` directory is further divided to separate the different implementations:

*   `src/CPU_C++/`: Holds the C++ source code designed to run on standard CPUs. This version prioritizes performance and portability across different CPU architectures.
*   `src/GPU_CUDAC++/`: Contains the CUDA C++ source code optimized for execution on NVIDIA GPUs. This implementation leverages the parallel processing capabilities of GPUs for significant speedups on suitable problems.
*   `src/CPU_Matlab/`: Includes the MATLAB scripts and functions. This version is primarily used for algorithm development, rapid prototyping, and visualization due to MATLAB's interactive environment and rich set of built-in tools.
*   `src/common/`: Contains utility code, such as matrix I/O routines or common data structures, that may be shared across the different implementations.

## Libraries (`lib/`)

The `lib/` directory includes essential third-party libraries:

*   **Eigen:** Located in `lib/Eigen`. Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. It is heavily used in the C++ CPU implementation for matrix manipulations and linear algebra operations.
*   **CUB:** Located in `lib/cub`. CUB is a flexible and reusable software library of CUDA C++ primitives and collective algorithms for high-performance GPU programming. It is utilized in the `GPU_CUDAC++` implementation to implement efficient parallel operations on the GPU.

Additionally, the C++ CPU version (`src/CPU_C++/`) relies on **Intel Math Kernel Library (MKL)** for optimized BLAS and sparse linear algebra routines. While the MKL source code is not directly included in the `lib/` directory (as it's typically installed system-wide or provided as pre-compiled libraries), its linkage and usage are integral to the C++ CPU build process for achieving high performance.

# CPU C++ Implementation

The C++ implementation for CPUs, located in `src/CPU_C++/`, provides a robust solver for large, sparse linear systems.

The primary entry point and control logic for this version are found in `src/CPU_C++/main.cpp`. This file handles:
*   Parsing command-line arguments.
*   Reading the input matrix and right-hand side vector.
*   Setting up the solver and preconditioner based on configuration parameters.
*   Executing the iterative solver.
*   Reporting solution statistics (e.g., iteration count, residual norm).

## Iterative Solver: FGMRES

The core iterative solver employed is Flexible GMRES (FGMRES). FGMRES is chosen for its robustness and effectiveness, particularly when combined with a powerful preconditioner that can vary from iteration to iteration, which is characteristic of AMG.

## Preconditioning

Effective preconditioning is key to the performance of FGMRES. This implementation utilizes a two-level preconditioning strategy:
1.  **AMG Preconditioner:** The primary preconditioner is an Algebraic Multigrid method. The core aggregation logic and multigrid cycle are implemented in `AGMG.cpp` (and related files). This preconditioner constructs a hierarchy of coarser grids and applies smoothing and coarse-grid corrections to approximate the inverse of the system matrix.
2.  **Optional ILU0 Smoother:** Within the AMG cycle, an Incomplete LU factorization with zero fill-in (ILU0) can be used as a smoother on each grid level. This helps to dampen high-frequency errors efficiently. The availability and configuration of ILU0 are typically controlled by parameters.

## Sparse Linear Algebra: Intel MKL

To achieve high performance for sparse matrix operations (e.g., sparse matrix-vector products, sparse triangular solves within ILU0), this implementation leverages the Intel Math Kernel Library (MKL). MKL provides highly optimized routines for these operations, tailored for Intel CPUs.

## Building and Running

To build the CPU C++ version:
*   Navigate to the `src/CPU_C++/` directory.
*   Use the provided `Makefile` (e.g., by running `make`). Ensure that the Intel MKL environment is correctly set up for successful compilation and linking.

To run the solver:
*   Execute the compiled binary (e.g., `main` or `test_AGMG`).
*   Command-line arguments are used to specify the input matrix, right-hand side, and other parameters. Refer to the `README.md` file in the root directory or in `src/CPU_C++/` for detailed information on command-line options.
*   Solver parameters, such as FGMRES tolerance, maximum iterations, and AMG settings, are often configured using JSON files (e.g., `input_fgmres.json`, `input_amg.json`). The path to these configuration files is typically provided via command-line arguments.

# GPU CUDA C++ Implementation

The CUDA C++ implementation, found in `src/GPU_CUDAC++/`, focuses on accelerating the computationally intensive setup phase of the Algebraic Multigrid method using NVIDIA GPUs.

The main control flow and GPU kernel launches are managed by `src/GPU_CUDAC++/main.cu`. This program reads an input matrix and performs the initial steps of the AMG algorithm on the GPU.

## GPU-Accelerated AMG Components

This implementation offloads several key parts of the AMG setup to the GPU:

*   **Aggregation Process:**
    *   **Strength of Connection:** Determining the "strong connections" between variables based on matrix entries is parallelized.
    *   **Breadth-First Search (BFS):** The BFS algorithm, used for identifying initial aggregates or for other graph traversal tasks within coarsening, is implemented with CUDA.
    *   **Pairing:** The process of pairing remaining free nodes to form aggregates is also GPU-accelerated.
*   **Prolongation Operator (P) Construction:** The computation and assembly of the prolongation operator, which maps coarse-grid values to the fine grid, are performed on the GPU. This often involves parallel gather/scatter operations.
*   **Coarse Grid Matrix (Ac) Formation:** The Galerkin product (Ac = P<sup>T</sup> * A * P), which forms the operator for the next coarser grid, is computed on the GPU. This typically involves sparse matrix-matrix multiplications.

## Sparse Linear Algebra: cuSPARSE

For GPU-accelerated sparse linear algebra operations, such as sparse matrix-vector multiplication (SpMV) and sparse matrix-matrix multiplication (SpMM) required in the AMG setup (especially for forming Ac), the implementation relies on the **NVIDIA cuSPARSE library**. cuSPARSE provides highly optimized routines for these tasks on NVIDIA GPUs.

## Building and Output

To build this version:
*   Navigate to the `src/GPU_CUDAC++/` directory.
*   Use the provided `Makefile` (e.g., by running `make`). This will require the CUDA Toolkit (nvcc compiler) to be installed and configured correctly.

The primary output of this `main.cu` executable is typically the prolongation operator matrix (`promatrix.mtx`) written to disk in Matrix Market format. This implementation primarily serves to demonstrate and benchmark the GPU-accelerated AMG setup phase. The generated prolongation operator could then potentially be read and utilized by a separate solver component (which might run on the CPU or GPU).

It is important to note that this specific CUDA implementation is heavily focused on the **AMG setup phase** (coarsening and operator construction) rather than providing a full iterative solver cycle on the GPU.

# MATLAB Implementation

The `src/CPU_Matlab/` directory contains the MATLAB implementation of the AMG solver. This version is primarily intended for:
*   Rapid prototyping of new algorithmic ideas.
*   Testing and verification of components or specific behaviors.
*   Comparison of results against the C++ and CUDA implementations.
*   Visualization of matrices, grids, and solver convergence.

The directory consists of various `.m` files, which are MATLAB scripts and functions implementing different parts of the AMG algorithm. It may also include MEX files (e.g., files like `dagtwolev_mex.f90` which appears to be a Fortran source for a MEX function, and compiled versions like `dmtlagtwolev.mexa64`). MEX files allow MATLAB to call functions written in C, C++, or Fortran directly, which can be useful for performance-critical sections or for integrating existing compiled code.

While this MATLAB version might not offer the same raw performance as the C++ or CUDA versions, its interactive environment and extensive toolbox support make it a valuable tool for research and development purposes within the project.

# Usage

This section provides guidance on how to compile and run the different implementations of the AMG solver available in this repository.

## C++ CPU Version

The primary way to build and run the C++ CPU version is described in the main `README.md` file.
1.  **Build:**
    *   Navigate to the `src/` directory: `cd ./src`
    *   Compile the code using `make`. This command likely invokes the `Makefile` in `src/CPU_C++/` either directly or via a target in `src/Makefile`. Ensure your environment is configured for Intel MKL.
2.  **Run:**
    *   Execute the solver using the `run.sh` script: `./run.sh matrixname kappatg npass tou`
    *   **Parameters:**
        *   `matrixname`: The name of the matrix file (without the `.mtx` extension) located in the `matrices/` directory (e.g., `apbl3D`).
        *   `kappatg`: A parameter related to the strength of connection threshold in AMG (e.g., `3.0`).
        *   `npass`: Number of smoothing passes (e.g., `2`).
        *   `tou`: A parameter, possibly related to a tolerance or threshold (e.g., `0.01`).
    *   For more detailed control over solver parameters (e.g., FGMRES tolerance, max iterations, AMG cycle type), modify the JSON configuration files (e.g., `input_fgmres.json`, `input_amg.json`) located in the `src/CPU_C++/` directory. These are typically read by `main.cpp`.

## GPU CUDA C++ Version

This version focuses on the AMG setup phase and outputs the prolongation operator.
1.  **Build:**
    *   Navigate to the CUDA source directory: `cd src/GPU_CUDAC++/`
    *   Compile using `make`. This uses the `src/GPU_CUDAC++/Makefile` and requires the NVIDIA CUDA Toolkit (nvcc) to be installed.
2.  **Run:**
    *   Execute the compiled binary: `./main matrixfile kappatg npass tou`
    *   **Parameters:**
        *   `matrixfile`: Name of the matrix file from the `matrices/` directory (e.g., `poisson10000`).
        *   `kappatg`, `npass`, `tou`: Similar parameters as used in the C++ CPU version, influencing the aggregation and prolongation operator construction.
    *   The main output will be `promatrix.mtx` in the execution directory.

## MATLAB Version

The MATLAB code is designed for interactive use and prototyping.
1.  **Environment:** Ensure you have MATLAB installed.
2.  **Run:**
    *   Open MATLAB and navigate to the `src/CPU_Matlab/` directory.
    *   A likely entry point for running the solver or specific tests is the `solve.m` script, or other individual `.m` files. You may need to set up paths or provide input matrices as per the script's requirements.
    *   If MEX files are used (e.g., `dmtlagtwolev.mexa64`), they should be automatically callable if they are compiled for your system and MATLAB version. If `dagtwolev_mex.f90` is present, it might need to be compiled using `mex` command in MATLAB, linked against appropriate libraries.

Refer to individual `README.md` files within specific subdirectories for any additional detailed instructions.

# Dependencies

The following lists the main external libraries and tools required to build and run the different parts of this project.

## C++ CPU Implementation

*   **C++ Compiler:** A compiler supporting C++11 or later (e.g., GCC, Clang, Intel C++ Compiler).
*   **GNU Make:** For processing the Makefiles (e.g., `src/CPU_C++/Makefile`).
*   **Intel Math Kernel Library (MKL):** Required for optimized BLAS and sparse linear algebra routines. Ensure MKL is installed and environment variables (e.g., `MKLROOT`) are set correctly.
*   **Eigen Library:** Included in `lib/Eigen`. This is a header-only library, so no separate compilation is needed, but it must be available in the include path.

## GPU CUDA C++ Implementation

*   **C++ Compiler:** A host C++ compiler compatible with `nvcc` (e.g., GCC, Clang).
*   **NVIDIA CUDA Toolkit:** Includes the `nvcc` compiler, CUDA runtime libraries, and development tools. Specifically, the **cuSPARSE** library (part of the CUDA Toolkit) is used for sparse matrix operations on the GPU.
*   **CUB Library:** Included in `lib/cub-1.8.0` (or a similar path like `lib/cub`). This is a header-only library for CUDA algorithm primitives.
*   **GNU Make:** For processing `src/GPU_CUDAC++/Makefile`.

## MATLAB Implementation

*   **MATLAB:** A licensed MATLAB installation.
*   **Fortran Compiler (Optional):** Needed if you intend to recompile Fortran-based MEX files (e.g., `dagtwolev_mex.f90`). A common choice is `gfortran`, which must be compatible with your MATLAB version's MEX setup. Compiled MEX files (e.g., `.mexa64` for 64-bit Linux) might be provided, potentially negating this requirement if they match your system.

## Common / Optional Tools

*   **Python:** May be required for utility scripts, such as `src/common/plotMatrix.py` for visualizing matrices. Standard Python libraries like Matplotlib might be needed for such scripts.
*   **`nlohmann/json.hpp`:** Included in `src/common/json.hpp`. This is a header-only library used by the C++ CPU implementation for parsing JSON configuration files. No separate installation is typically needed as it's part of the repository.
```
