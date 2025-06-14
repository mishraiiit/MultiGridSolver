#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include "../common/MatrixIO.cpp"
#include "../common/json.hpp"
#include "../CPU_C++/TicToc.cpp"
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
    MultiGridPrecond(const SMatrix & A_in, const SMatrix & P_in) {
      this->A = A_in;
      this->P = P_in;
      this->Ptrans = P.transpose();
      const SMatrix Ac = this->Ptrans * this->A * this->P;

      this->solver.analyzePattern(Ac);
      this->solver.factorize(Ac);

      //this->ilut.setFillfactor(7);
      this->ilut.setDroptol(1e-2);
      this->ilut.compute(this->A); // Use this->A (which is A_in)

      this->multiplicative_precond = true;
      this->use_preconditioner = true;
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
  srand(0);
  if(argc != 3) {
    printf("Incorrect number of arguments.\n");
    printf("Usage: $ ./main_bicg <matrix_name> <cpu|gpu>\n");
    exit(1);
  }

  string matrix_name = argv[1];
  string device = argv[2];
  double tol = 1e-6;

  SMatrix A = readMatrix(string("../../matrices/") + matrix_name + string(".mtx"));
  SMatrix P_matrix = readMatrix(string("../../matrices/") + matrix_name + string("promatrix_") + device + string(".mtx"));


  MultiGridPrecond precond(A, P_matrix);

  VectorXd x(A.rows());
  x.setZero(); // Initialize x to zero for consistent starting point

  VectorXd b(A.rows());
  for(int i = 0; i < A.rows(); i++) {
    b[i] = rand() / (RAND_MAX + 0.0);
  }

  int max_iter = 10000;

  TicToc solverTimer("BiCGStab_SolveTimer", 4);
  solverTimer.tic();
  int status = BiCGSTABiml(A, x, b, precond, max_iter, tol);
  solverTimer.toc();

  if(status == 0) {
    printScreen(4, "Tolerance ", tol);
    printScreen(4, "Number of iterations BICG", max_iter);
  } else {
    std::cout << "BiCGSTABiml encountered a problem with status code: " << status << std::endl;
  }


  return 0;
}
