#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include "AGMG.cpp" // Assuming this defines SMatrix or includes necessary Eigen headers
#include "../common/MatrixIO.cpp" // For readMatrix, writeMatrix
#include "TicToc.cpp"
#include <typeinfo>
#include <string>
#define EIGEN_USE_MKL_ALL
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

class MultiGridPrecond {
  private:
    SMatrix A;
    SMatrix P;
    SMatrix Ptrans;
    SparseLU <SparseMatrix<double> > solver;
    IncompleteLUT<double> ilut;
    bool multiplicative_precond;
    bool use_preconditioner;
  public:
    MultiGridPrecond(const SMatrix & A, const double ktg, const int npass,\
     const double tou, const bool multiplicative_precond, const bool use_preconditioner, string matrixname) {
      TicToc AGMGTimer("AGMGTimer", 4);
      TicToc IncompleteLUTimer("IncompleteLUTimer", 4);
      TicToc SparseLUTimer("SparseLUTimer", 4);

      this->A = A;
      AGMGTimer.tic();
      this->P = AGMG::multiple_pairwise_aggregation(A, ktg, npass, tou, 0);
      AGMGTimer.toc();
      writeMatrix(string("../../matrices/") + matrixname + string("promatrix_cpu.mtx"), this->P);

      this->Ptrans = P.transpose();
      const SMatrix Ac = this->Ptrans * this->A * this->P;

      SparseLUTimer.tic();
      this->solver.analyzePattern(Ac);
      this->solver.factorize(Ac);
      SparseLUTimer.toc();

      IncompleteLUTimer.tic();
      //this->ilut.setFillfactor(7);
      this->ilut.setDroptol(1e-2);
      this->ilut.compute(A);
      IncompleteLUTimer.toc();

      this->multiplicative_precond = multiplicative_precond;
      this->use_preconditioner = use_preconditioner;
    };

    template<typename T>
    T multigrid_solve(const T & vec) const {
      return (this->P * (this->solver.solve(this->Ptrans * vec)));
    };

    template<typename T>
    T solve(const T & vec) const {
      if(!use_preconditioner)
        return vec;
      if(multiplicative_precond) {
        T res = multigrid_solve(vec);
        return res + this->ilut.solve(vec) - this->ilut.solve(A * res);
      } else {
        return multigrid_solve(vec) + this->ilut.solve(vec);
      }
    };
};

template<typename V>
inline double dot(const V & A, const V & B) {
  return A.dot(B);
}

template<typename V>
inline double norm(const V & A) {
  return A.norm();
}

template < class Matrix, class Vector, class Preconditioner, class Real >
int
BiCGSTABiml(const Matrix &A, Vector &x, const Vector &b, const Preconditioner &M, int &max_iter, Real &tol) {
  Real resid;
  Vector rho_1(1), rho_2(1), alpha(1), beta(1), omega(1);
  Vector p, phat, s, shat, t, v;

  Real normb = norm(b);
  Vector r = b - A * x;
  Vector rtilde = r;

  if (normb == 0.0)
    normb = 1;

  if ((resid = norm(r) / normb) <= tol) {
    tol = resid;
    max_iter = 0;
    return 0;
  }

  for (int i = 1; i <= max_iter; i++) {
    rho_1(0) = dot(rtilde, r);
    if (rho_1(0) == 0) {
      tol = norm(r) / normb;
      return 2;
    }
    if (i == 1)
      p = r;
    else {
      beta(0) = (rho_1(0)/rho_2(0)) * (alpha(0)/omega(0));
      p = r + beta(0) * (p - omega(0) * v);
    }
    phat = M.solve(p);
    v = A * phat;
    alpha(0) = rho_1(0) / dot(rtilde, v);
    s = r - alpha(0) * v;
    if ((resid = norm(s)/normb) < tol) {
      x += alpha(0) * phat;
      max_iter = i;
      tol = resid;
      return 0;
    }
    shat = M.solve(s);
    t = A * shat;
    omega(0) = dot(t,s) / dot(t,t);
    x += alpha(0) * phat + omega(0) * shat;
    r = s - omega(0) * t;

    rho_2(0) = rho_1(0);
    if ((resid = norm(r) / normb) < tol) {
      tol = resid;
      max_iter = i;
      return 0;
    }
    if (omega(0) == 0) {
      tol = norm(r) / normb;
      return 3;
    }
  }

  tol = resid;
  return 1;
}

int main (int argc, char ** argv) {

  if(argc != 5) {
    printf("Invalid arguments.\n");
    printf("Usage: %s <matrix_basename> <ktg> <npass> <tou>\n", argv[0]);
    printf("Example: %s mymatrix 10.0 2 4.0\n", argv[0]);
    printf("  <matrix_basename>: string, e.g., 'mymatrix' (reads ../../matrices/mymatrix.mtx)\n");
    printf("  <ktg>: double, e.g., 10.0\n");
    printf("  <npass>: integer, e.g., 2\n");
    printf("  <tou>: double, e.g., 4.0\n");
    exit(1);
  }

  std::string matrixname_arg = argv[1];
  double ktg_val = 0.0;
  int npass_val = 0;
  double tou_val = 0.0;

  try {
    ktg_val = std::stod(argv[2]);
    npass_val = std::stoi(argv[3]);
    tou_val = std::stod(argv[4]);
  } catch (const std::invalid_argument& ia) {
    std::cerr << "Invalid argument type: " << ia.what() << std::endl;
    // Consider re-printing usage instructions here if desired
    exit(1);
  } catch (const std::out_of_range& oor) {
    std::cerr << "Argument out of range: " << oor.what() << std::endl;
    exit(1);
  }

  std::cout << "Starting AGMG process with parameters:" << std::endl;
  std::cout << "  Matrix basename: " << matrixname_arg << std::endl;
  std::cout << "  ktg: " << ktg_val << std::endl;
  std::cout << "  npass: " << npass_val << std::endl;
  std::cout << "  tou: " << tou_val << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  std::string matrix_path = std::string("../../matrices/") + matrixname_arg + std::string(".mtx");
  std::cout << "Reading matrix from: " << matrix_path << std::endl;
  
  SMatrix A;
  try {
    A = readMatrix(matrix_path);
  } catch (const std::exception& e) {
    std::cerr << "Error reading matrix: " << e.what() << std::endl;
    exit(1);
  }
  
  // Assuming SMatrix has methods rows(), cols(), nonZeros() (typical for Eigen::SparseMatrix)
  std::cout << "Matrix A loaded: " << A.rows() << "x" << A.cols();
  if (A.nonZeros() >= 0) { // nonZeros() might return 0 for an empty matrix, check if it's a valid call
      std::cout << ", " << A.nonZeros() << " non-zeros.";
  }
  std::cout << std::endl;


  TicToc AGMGTimer("AGMG Core Algorithm Time", 4);
  AGMGTimer.tic();
  // The '0' as the last parameter corresponds to final_size_factor_activerow,
  // implying reliance on npass and tou for coarsening control.
  SMatrix P = AGMG::multiple_pairwise_aggregation(A, ktg_val, npass_val, tou_val, 0);
  AGMGTimer.toc();
  
  std::cout << "AGMG aggregation complete." << std::endl;
  std::cout << "Prolongation matrix P created: " << P.rows() << "x" << P.cols();
  if (P.nonZeros() >= 0) {
      std::cout << ", " << P.nonZeros() << " non-zeros.";
  }
  std::cout << std::endl;

  std::string output_path = std::string("../../matrices/") + matrixname_arg + std::string("promatrix_cpu.mtx");
  std::cout << "Writing P matrix to: " << output_path << std::endl;
  
  try {
    writeMatrix(output_path, P);
  } catch (const std::exception& e) {
    std::cerr << "Error writing matrix: " << e.what() << std::endl;
    exit(1);
  }
  std::cout << "P matrix successfully written." << std::endl;

  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "Process finished." << std::endl;

  return 0;
}
