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
#include "../common/json.hpp";
#include "TicToc.cpp"
#include <typeinfo>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include "mkl_dss.h"
#include "mkl_rci.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_service.h"

#define SZ 128

std::string itoa(int number) {
    std::string ret;
    while(number != 0) {
        ret = ((char)('0' + (number % 10))) + ret;
        number = number / 10;
    }
    return ret;
}

struct MatrixCSR {
    MKL_INT rows, cols, nnz;
    MKL_INT * rowPtr, * colIdx;
    double * val;
};

MatrixCSR * readMatrixCPUMemoryCSR(std::string filename) {
    std::ifstream fin(filename);
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;

    // Read the data
    MatrixCSR * matrix_csr;
    matrix_csr = (MatrixCSR *) malloc(sizeof(MatrixCSR));

    matrix_csr->rows = M;
    matrix_csr->cols = N;
    matrix_csr->nnz = L;

    matrix_csr->rowPtr= (MKL_INT *) malloc(sizeof(MKL_INT) * (matrix_csr->rows + 1));
    matrix_csr->colIdx = (MKL_INT *) malloc(sizeof(MKL_INT) * L);
    matrix_csr->val = (double *) malloc(sizeof(double) * L);

    int filled = 0;

    std::vector< std::vector <std::pair<int, double> > > matrix_data(M);
    for (int l = 0; l < L; l++) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        matrix_data[m - 1].push_back({n - 1, data});
    }

    for(int i = 0; i < M; i++) {
        std::sort(matrix_data[i].begin(), matrix_data[i].end());
        matrix_csr->rowPtr[i] = filled;
        for(auto tp : matrix_data[i]) {
            matrix_csr->colIdx[filled] = tp.first;
            matrix_csr->val[filled] = tp.second;
            filled++;
        }
    }

    matrix_csr->rowPtr[M] = filled;

    assert(filled == L);
    fin.close();
    
    return matrix_csr;
}

MatrixCSR * convertEigenToMKL(const SMatrix & matrix) {
  MatrixCSR * to_return = (MatrixCSR *) malloc(sizeof(MatrixCSR));
  to_return->rows = matrix.rows();
  to_return->cols = matrix.cols();
  to_return->nnz = matrix.nonZeros();
  to_return->rowPtr = (MKL_INT *) malloc(sizeof(MKL_INT) * (to_return->rows + 1));
  to_return->colIdx = (MKL_INT *) malloc(sizeof(MKL_INT) * (to_return->nnz));
  to_return->val = (double *) malloc(sizeof(double) * (to_return->nnz));
  int filled = 0;
  for(int u = 0; u < matrix.rows(); u++) {
    const SparseVector<double> rowvec = matrix.row(u);
    to_return->rowPtr[u] = filled;
    for(SparseVector<double>::InnerIterator i(rowvec); i; ++i) {
      int v = i.index();
      to_return->colIdx[filled] = v;
      to_return->val[filled] = i.value();
      filled++;
    }
  }
  to_return->rowPtr[to_return->rows] = filled;
  return to_return;
}

void createMKLCSRHandle(sparse_matrix_t * csrA, MatrixCSR * matrix) {
  mkl_sparse_d_create_csr (csrA, SPARSE_INDEX_BASE_ZERO, matrix->rows, matrix->cols, matrix->rowPtr, matrix->rowPtr + 1, matrix->colIdx, matrix->val);
}

void apply_prec_ILU0(MKL_INT n, double *bilu0, MKL_INT * ia, MKL_INT * ja, double * v, double * pv) {
  char cvar1, cvar, cvar2;
  double *tmp = (double*)malloc(n * sizeof(double) );

  cvar1 = 'L';
  cvar  = 'N';
  cvar2 = 'U';
  mkl_dcsrtrsv (&cvar1, &cvar, &cvar2, &n, bilu0, ia, ja, v, tmp);

  cvar1 = 'U';
  cvar = 'N';
  cvar2 = 'N';
  mkl_dcsrtrsv (&cvar1, &cvar, &cvar2, &n, bilu0, ia, ja, tmp, pv);

  free(tmp);

  return;
}

MatrixCSR * convertToOneBased(MatrixCSR * matrix) {
  MatrixCSR * to_return = (MatrixCSR * ) malloc(sizeof(MatrixCSR));
  to_return->rows = matrix->rows;
  to_return->cols = matrix->cols;
  to_return->nnz = matrix->nnz;
  to_return->rowPtr = (MKL_INT *) malloc((to_return->rows + 1) * sizeof(MKL_INT));
  to_return->colIdx = (MKL_INT *) malloc((to_return->nnz) * sizeof(MKL_INT));

  for(int i = 0; i < matrix->rows + 1; i++) {
    to_return->rowPtr[i] = matrix->rowPtr[i] + 1;
  }
  for(int i = 0; i < matrix->nnz; i++) {
    to_return->colIdx[i] = matrix->colIdx[i] + 1;
  }
  to_return->val = NULL;
  return to_return;
}

int main (int argc, char ** argv) {
  srand(0);

  if(argc != 2) {
    printf("No input json configuration file given\n");
    printf("Usage: $ ./main input.json\n");
    exit(1);
  }

  ifstream inp_stream(argv[1]);
  nlohmann::json j;
  inp_stream >> j;

  string matrixname;
  double ktg;
  int npass = 1000000000;
  double tou = 1e18;
  
  matrixname = j["matrixName"];
  ktg = j["kappa"];

  bool npassorfinalsz = j["aggregateByAcSize"];
  int arg4 = j["acSizeOrNpassValue"];

  if(npassorfinalsz == 0) {
    npass = arg4;
    arg4 = 0;
  }

  int PRECODITIONER_USE = j["usePreconditioner"];
  int GMRES_NON_RESTART_ITERATIONS = j["restartIterationsGMRES"];
  int ILU_PRECONDITIONER = j["ILUSmoother"];
  int ADDITIVE_PRECONDITIONER = 1 - ((int) j["multiplicativePreconditioner"]);
  double RELATIVE_TOLERANCE = j["relativeTolerance"];

  TicToc MatrixReadTime("Matrix Read Time", 4);
  TicToc AggregationTime("Aggregation Time", 4);
  TicToc EigenToMKLTime("Eigen to MKL Time", 4);
  TicToc DssInitLUTime("DssInitLU Time", 4);
  TicToc IncompleteLUTime("IncompleteLUTime", 4);
  TicToc DGMRESInitTime("DGMRESInitTime", 4);
  TicToc DGMRESMainTime("DGMRESMainTime", 4);

  MatrixReadTime.tic();
  SMatrix T = readMatrix(string("../../matrices/") + matrixname + string(".mtx"));
  MatrixReadTime.toc();

  AggregationTime.tic();
  auto pro_matrix = AGMG::multiple_pairwise_aggregation(T, ktg, npass, tou, arg4);
  AggregationTime.toc();

  // writeMatrix(string("../../matrices/") + matrixname + string("promatrix.mtx"), pro_matrix);
  

  EigenToMKLTime.tic();
  SMatrix pro_matrix_transpose = pro_matrix.transpose();
  SMatrix coarse_grid_matrix_csr = (pro_matrix_transpose * T * pro_matrix);
  MatrixCSR * matrix = convertEigenToMKL(T);
  MatrixCSR * matrix_one_based = convertToOneBased(matrix);
  MatrixCSR * matrix_compressed = convertEigenToMKL(coarse_grid_matrix_csr);
  MatrixCSR * matrix_p = convertEigenToMKL(pro_matrix);
  MatrixCSR * matrix_p_transpose = convertEigenToMKL(pro_matrix_transpose);
  EigenToMKLTime.toc();


  DssInitLUTime.tic();
  _MKL_DSS_HANDLE_t dss_handle;
  MKL_INT opt = MKL_DSS_ZERO_BASED_INDEXING;
  MKL_INT opt_defaults = MKL_DSS_DEFAULTS;
  if(dss_create(dss_handle, opt) != MKL_DSS_SUCCESS) {
  	printf("dss_create() didn't work correctly\n");
  	exit(1);
  }

  MKL_INT non_sym = MKL_DSS_NON_SYMMETRIC;
  if(dss_define_structure(dss_handle, non_sym, matrix_compressed->rowPtr,\
    matrix_compressed->rows, matrix_compressed->cols, matrix_compressed->colIdx, matrix_compressed->nnz) != MKL_DSS_SUCCESS) {
  	printf("dss_define_structure() didn't work correctly\n");
    exit(1);
  }

  MKL_INT reorder_opt = MKL_DSS_AUTO_ORDER;
  if(dss_reorder(dss_handle, reorder_opt, NULL) != MKL_DSS_SUCCESS) {
  	printf("dss_reorder() didn't work correctly\n");
  	exit(1);
  }

  MKL_INT dss_indefinte = MKL_DSS_INDEFINITE;
  if(dss_factor_real(dss_handle, dss_indefinte, matrix_compressed->val) != MKL_DSS_SUCCESS) {
  	printf("dss_factor_real() didn't work correctly\n");
  	exit(1);
  }
  DssInitLUTime.toc();



  DGMRESInitTime.tic();
  MKL_INT N = matrix->rows;
  int nnz = matrix->nnz;

  /*---------------------------------------------------------------------------
  * Allocate storage for the ?par parameters and the solution/rhs/residual vectors
  *---------------------------------------------------------------------------*/
  MKL_INT ipar[SZ];
  int total = ((2* GMRES_NON_RESTART_ITERATIONS + 1) * N + GMRES_NON_RESTART_ITERATIONS*(GMRES_NON_RESTART_ITERATIONS + 9)/2 + 1);
  double * dpar = new double[SZ];
  double * tmp = new double[total];
  double * expected_solution = new double[N];
  double * buffer = new double[N];
  double * buffer1 = new double[N];
  double * diag = new double[N];
  double * rhs = new double[N];
  double * b = new double[N];
  double * computed_solution = new double[N];
  double * residual = new double[N];
  double * bilu0 = new double[nnz];
  double * inp, * out;
  double nrm;

  if(!ILU_PRECONDITIONER) {
	  for(int i = 0; i < N; i++) {
	    diag[i] = 0;
	    for(int j = matrix->rowPtr[i]; j < matrix->rowPtr[i + 1]; j++) {
	      if(i == matrix->colIdx[j]) {
	        diag[i] = matrix->val[j];
	      }
	    }
	  }
   }

  for(int i = 0; i < N; i++) {
    expected_solution[i] = rand() / (RAND_MAX + 0.0);
  }
  /*---------------------------------------------------------------------------
  * Some additional variables to use with the RCI (P)FGMRES solver
  *---------------------------------------------------------------------------*/
  MKL_INT itercount;
  MKL_INT RCI_request, i, ivar;
  double dvar;

  // Descriptor of main sparse matrix properties
  struct matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  descrA.mode = SPARSE_FILL_MODE_UPPER;
  descrA.diag = SPARSE_DIAG_NON_UNIT;

  // Structure with sparse matrix stored in CSR format
  sparse_matrix_t csrA;
  createMKLCSRHandle(&csrA, matrix);

  sparse_matrix_t csrP_matrix;
  createMKLCSRHandle(&csrP_matrix, matrix_p);

  sparse_matrix_t csrP_matrix_transpose;
  createMKLCSRHandle(&csrP_matrix_transpose, matrix_p_transpose);
  
  /*---------------------------------------------------------------------------
  * Initialize variables and the right hand side through matrix-vector product
  *---------------------------------------------------------------------------*/

  ivar = N;
  mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA, expected_solution, 0.0, rhs);
  /*---------------------------------------------------------------------------
  * Save the right-hand side in vector b for future use
  *---------------------------------------------------------------------------*/
  i = 1;
  dcopy (&ivar, rhs, &i, b, &i);
  /*---------------------------------------------------------------------------
  * Initialize the initial guess
  *---------------------------------------------------------------------------*/
  for (i = 0; i < N; i++)
    {
      computed_solution[i] = 0.0;
    }
  /*---------------------------------------------------------------------------
  * Initialize the solver
  *---------------------------------------------------------------------------*/
  dfgmres_init (&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp);
  if (RCI_request != 0)
    goto FAILED;
  /*---------------------------------------------------------------------------
  * Set the desired parameters:
  * -------------------------------------------------------------------------**/
  ipar[7] = 0; // to have maximum iterations stoppage or not.
  ipar[4] = 5000; // maxmimum number of iterations, if ipar[7] is 0, ipar[4] doesn't matter.
  ipar[14] = GMRES_NON_RESTART_ITERATIONS;
  ipar[10] = PRECODITIONER_USE;
  ipar[30] = 0;
  dpar[0] = 1.0E-3;
  DGMRESInitTime.toc();

  if(ILU_PRECONDITIONER) {
  	IncompleteLUTime.tic();
    MKL_INT ierr;
    dcsrilu0(&N, matrix->val, matrix_one_based->rowPtr, matrix_one_based->colIdx, bilu0, ipar, dpar, &ierr);
    if(ierr != 0) {
      printf("Incomplete ILU didn't work as expected.\n");
      printf("Error code: %d\n", ierr);
      return 1;
    }
    IncompleteLUTime.toc();
  }

  DGMRESMainTime.tic();
  /*---------------------------------------------------------------------------
  * Check the correctness and consistency of the newly set parameters
  *---------------------------------------------------------------------------*/
  dfgmres_check (&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp);
  if (RCI_request != 0)
    goto FAILED;

  // printIparInfo(ipar);
  /*---------------------------------------------------------------------------
  * Compute the solution by RCI (P)FGMRES solver with preconditioning
  * Reverse Communication starts here
  *---------------------------------------------------------------------------*/
  ONE:

  dfgmres (&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp);

  switch (RCI_request) {
    case 0: /* The solution was found with the required precision */
      goto COMPLETE;

    case 1:
      mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA, &tmp[ipar[21] - 1], 0.0, &tmp[ipar[22] - 1]);
      goto ONE;

    case 2: /* Stopping criteria */
      ipar[12] = 1;
      /* Get the current FGMRES solution in the vector b[N] */
      dfgmres_get (&ivar, computed_solution, b, &RCI_request, ipar, dpar, tmp, &itercount);
      /* Compute the current true residual via Intel(R) MKL (Sparse) BLAS routines */
      mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA, b, 0.0, residual);
      dvar = -1.0E0;
      i = 1;
      daxpy (&ivar, &dvar, rhs, &i, residual, &i);
      dvar = dnrm2 (&ivar, residual, &i);
      nrm = dnrm2 (&ivar, rhs, &i);
      if (dvar / nrm < RELATIVE_TOLERANCE)
        goto COMPLETE;
      else
        goto ONE;

    case 3:
      if(ADDITIVE_PRECONDITIONER) {
        if(!ILU_PRECONDITIONER) {
          // DIAG_PRECONDITIONER
          inp = tmp + ipar[21] - 1;
          out = tmp + ipar[22] - 1;
          mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrP_matrix_transpose, descrA, inp, 0.0, buffer);
          MKL_INT nrhs = 1;
          dss_solve_real(dss_handle, opt_defaults, out, nrhs, buffer);
          mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrP_matrix, descrA, out, 0.0, buffer);
          for(int i = 0; i < N; i++) {
            out[i] = buffer[i] + (inp[i] / diag[i]);
          }
        } else {
          // ILU_PRECONDITIONER
          inp = tmp + ipar[21] - 1;
          out = tmp + ipar[22] - 1;
          mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrP_matrix_transpose, descrA, inp, 0.0, out);
          MKL_INT nrhs = 1;
          dss_solve_real(dss_handle, opt_defaults, out, nrhs, buffer);
          mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrP_matrix, descrA, buffer, 0.0, out);
          apply_prec_ILU0( N, bilu0, matrix_one_based->rowPtr, matrix_one_based->colIdx, inp, buffer);
          for(int i = 0; i < N; i++) {
            out[i] += buffer[i];
          }
        }
      } else {
        if(!ILU_PRECONDITIONER) {
          printf("Diagonal smoother with Multiplicative preconditioner isn't implemented yet.\n");
          exit(1);
        } else {
          // ILU_PRECONDITIONER
          inp = tmp + ipar[21] - 1;
          out = tmp + ipar[22] - 1;
          mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrP_matrix_transpose, descrA, inp, 0.0, out);
          // out = P^ x
          MKL_INT nrhs = 1;
          dss_solve_real(dss_handle, opt_defaults, out, nrhs, buffer);
          // buffer = Ac-1 PT x
          mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrP_matrix, descrA, buffer, 0.0, out);
          // out = P Ac-1 PT x
          apply_prec_ILU0( N, bilu0, matrix_one_based->rowPtr, matrix_one_based->colIdx, inp, buffer);
          // buffer = ILU_Solve(x)
          mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA, out, 0.0, buffer1);
          // buffer1 = A P Ac-1 PT x
          for(int i = 0; i < N; i++) {
            out[i] += buffer[i];
          }
          // out = P Ac-1 PT x + ILU_Solve(x)
          apply_prec_ILU0( N, bilu0, matrix_one_based->rowPtr, matrix_one_based->colIdx, buffer1, buffer);
          // buffer = ILU_Solve(A P Ac-1 PT x)
          for(int i = 0; i < N; i++) {
            out[i] = out[i] - buffer[i];
          }
          // out = P Ac-1 PT x + ILU_Solve(x) - ILU_Solve(A P Ac-1 PT x)
        }
      }
      goto ONE;

    case 4:
      if (dpar[6] < 1.0E-12)
        goto COMPLETE;
      else
        goto ONE;

    default:
      goto FAILED;
  }
  

COMPLETE:ipar[12] = 0;
  dfgmres_get (&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp, &itercount);
  /*---------------------------------------------------------------------------
  * Print solution vector: computed_solution[N] and the number of iterations: itercount
  *---------------------------------------------------------------------------*/
  // printf (" The system has been solved \n");
  // printf ("\n The following solution has been obtained: \n");
  // for (i = 0; i < N; i++)
  //   {
  //     printf ("computed_solution[%d]=", i);
  //     printf ("%e\n", computed_solution[i]);
  //   }
  // printf ("\n The expected solution is: \n");
  // for (i = 0; i < N; i++)
  //   {
  //     printf ("expected_solution[%d]=", i);
  //     printf ("%e\n", expected_solution[i]);
  //   }
  DGMRESMainTime.toc();
  printf("\n");
  printScreen(4, "Number of FGMRES iterations", itercount);

  for(int i = 0; i < N; i++) {
    expected_solution[i] -= computed_solution[i];
  }
  i = 1;
  dvar = dnrm2 (&ivar, expected_solution, &i);


  MKL_Free_Buffers ();
  free(matrix);
  free(matrix_one_based);
  free(matrix_p);
  free(matrix_p_transpose);

  delete [] dpar;
  delete [] tmp;
  delete [] expected_solution;
  delete [] buffer;
  delete [] rhs;
  delete [] b;
  delete [] computed_solution;
  delete [] residual;

  if (dvar < 1.0e-14)
    {
      printf("iterations : %d\n", itercount);
      printf ("\nThis example has successfully PASSED through all steps of ");
      printf ("computation!\n");
      return 0;
    }
  else
    {
      printf ("The computed solution\n");
      printf ("differs much from the expected solution (Euclidean norm is %e), or both.\n", dvar);
      return 1;
    }

  FAILED:
  printf ("\nThis example FAILED as the solver has returned the ERROR code %d", RCI_request);
  mkl_sparse_destroy(csrA);
  MKL_Free_Buffers ();
  return 1;
    
}
