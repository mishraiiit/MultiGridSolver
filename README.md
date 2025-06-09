# MultiGrid Solver

A serial implementation of the aggregation-based algebraic multigrid method for convection-diffusion problems.

## Overview

This project implements the algorithm described in [AGMG For Convection Diffusion](https://github.com/mishraiiit/MultiGridSolver/blob/master/docs/AGMG_For_Convection_Diffusion.pdf) paper in serial execution mode.

## System Requirements

### Development Environment
This project was developed and tested on:
- **OS**: Ubuntu 24.04.2 LTS (Noble Numbat)
- **Kernel**: 6.11.0-1015-gcp
- **Platform**: Google Cloud Platform VM
- **GPU**: NVIDIA L4
- **Driver Version**: 535.230.02
- **CUDA Version**: 12.2
- **GPU Memory**: 23,034 MiB

## Installation & Usage

### Building the Project

1. Navigate to the source directory and compile:
   ```bash
   cd ./src
   make
   ```

### Running the Solver

Execute the solver using the provided shell script:

```bash
./run.sh <matrixname> <kappatg> <npass> <tou>
```

#### Parameters

| Parameter | Description |
|-----------|-------------|
| `matrixname` | Name of the matrix file (must be present in the `matrices/` directory in Matrix Market `.mtx` format) |
| `kappatg` | Aggregation parameter as defined in the paper |
| `npass` | Number of passes for the algorithm |
| `tou` | Tolerance parameter for convergence |

### Example

```bash
./run.sh example_matrix 0.5 10 1e-6
```

This would run the solver on `matrices/example_matrix.mtx` with kappatg=0.5, npass=10, and tou=1e-6.

## Directory Structure

```
MultiGridSolver/
├── src/           # Source code files
├── matrices/      # Input matrix files (.mtx format)
├── docs/          # Documentation and papers
└── run.sh         # Execution script
```

## Matrix Format

Input matrices must be in Matrix Market (`.mtx`) format and placed in the `matrices/` directory.

## License

[Add license information here]

## Contact

ayushmishra.iit@gmail.com


