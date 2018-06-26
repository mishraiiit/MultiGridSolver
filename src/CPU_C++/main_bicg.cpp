#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include "AGMG.cpp"
#include "../common/MatrixIO.cpp"
#include "../common/termcolor.hpp"
#include "TicToc.cpp"
#include <typeinfo>
#include <string>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

class MultiGridPrecond {
  private:
    SMatrix A;
    SMatrix Ac;
    SMatrix P;
    SMatrix Ptrans;
    SparseLU  <SparseMatrix<double> > solver;
    IncompleteLUT<double> ilut;
    double ktg;
    int npass;
    double tou;
    bool multiplicative_precond;
    bool use_preconditioner;
  public:
    MultiGridPrecond(const SMatrix & A, const double ktg, const int npass,\
     const double tou, const bool multiplicative_precond, const bool use_preconditioner) {
      TicToc AGMGTimer("AGMGTimer", 4);
      TicToc IncompleteLUTimer("IncompleteLUTimer", 4);
      TicToc SparseLUTimer("SparseLUTimer", 4);

      this->A = A;
      this->ktg = ktg;
      this->npass = npass;
      this->tou = tou;

      AGMGTimer.tic();
      this->P = AGMG::multiple_pairwise_aggregation(A, ktg, npass, tou, 0);
      AGMGTimer.toc();
      this->Ptrans = P.transpose();
      this->Ac = this->Ptrans * this->A * this->P;

      SparseLUTimer.tic();
      this->solver.compute(Ac);
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

void printArgumentsInfo() {
    std::cout << std::endl;
    std::cout << termcolor::red << "Invalid arguments.\n" << std::endl;

    std::cout << termcolor::green << "Usage:" << termcolor::bold << " ./main_bicg Arg1 Arg2 Arg3 Arg4 Arg5 Arg6 Arg7\n\n" << termcolor::reset;
    std::cout << termcolor::reset << std::endl;

    std::cout << termcolor::yellow << "Argument 1: " << termcolor::reset;
    std::cout << "Name of matrix file present in /matrices folder,\n            do not put .mtx after the filename in argument.";
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << termcolor::yellow << "Argument 2: " << termcolor::reset;
    std::cout << "The parameter ktg in AGMG, default value is 8.";
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << termcolor::yellow << "Argument 3: " << termcolor::reset;
    std::cout << "Aggregate based on npass or final size of Ac, 0 for npass,\n            1 for final size of Ac.";
    std::cout << std::endl;
    std::cout << std::endl;


    std::cout << termcolor::yellow << "Argument 4: " << termcolor::reset;
    std::cout << "If Arg3 is 0, Arg4 = npass, else Arg4 = final size of Ac\n            For final size, aggreg. is done till size becomes less than finalsz";
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << termcolor::yellow << "Argument 5: " << termcolor::reset;
    std::cout << "Whether to use preconditioner with BICG                \n             1 for yes, 0 for no";
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << termcolor::yellow << "Argument 6: " << termcolor::reset;
    std::cout << "Additive preconditioner or multiplicative preconditoner\n            0 for  Additive, 1 for Multiplicative";
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << termcolor::yellow << "Argument 7: " << termcolor::reset;
    std::cout << "Relative tolerance";
    std::cout << std::endl;
    std::cout << std::endl;
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

  if(argc != 8) {
    printArgumentsInfo();
    exit(1);
  }


  string matrixname;
  double ktg;
  int npass = 1000000000;
  double tou = 1e18;

  matrixname = argv[1];
  ktg = stod(argv[2]);

  int npassorfinalsz = stoi(argv[3]);
  int arg4 = stod(argv[4]);
  if(npassorfinalsz == 0) {
    npass = arg4;
    arg4 = 0;
  }
  bool use_preconditioner = stoi(argv[5]);
  bool multiplicative_precond = stoi(argv[6]);
  double tol = stod(argv[7]);

  SMatrix A = readMatrix(string("../../matrices/") + matrixname + string(".mtx"));


  MultiGridPrecond precond(A, ktg, npass, tou, multiplicative_precond, use_preconditioner);

  VectorXd x(A.rows());

  VectorXd b(A.rows());
  for(int i = 0; i < A.rows(); i++) {
    b[i] = rand() / (RAND_MAX + 0.0);
  }

  int max_iter = 10000;

  if(BiCGSTABiml(A, x, b, precond, max_iter, tol) == 0) {
    std::cout << tol << std::endl;
    std::cout << max_iter << std::endl;
  } else {
    std::cout << "BiCGSTABiml encountered a problem\n" << std::endl;
  }


  return 0;
}
