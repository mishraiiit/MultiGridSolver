#include <iostream>
#include "AGMG.cpp"
#include "MatrixIO.cpp"
#include <typeinfo>
#include <Eigen/Sparse>
#include <bits/stdc++.h>
using namespace std;
using namespace Eigen;

typedef SparseMatrix<double, RowMajor> SMatrix;

int main() {

	SMatrix T = readMatrix("../matrices/CSky3d30.mtx");
	
  auto result = AGMG::multiple_pairwise_aggregation(T.rows(), T, 8, 2, 4);
  auto pro_matrix = AGMG::get_prolongation_matrix(T, result.first.second);
  writeMatrix("../matrices/CSky3d30promatrix_reduced.mtx", pro_matrix);
  // cout << T.rows() << " " << result.first.first << endl;
  //  auto pro_matrix = AGMG::get_prolongation_matrix(T, result.first.second);
    
  //  for(int i = 0; i < pro_matrix.rows(); i++) {
  //    SparseVector<double> curr = pro_matrix.row(i);
  //    for(SparseVector<double>::InnerIterator j(curr); j; ++j) {
  //      cout << i + 1 << " " << j.index() + 1 << " " << j.value() << endl;
  //    }
  //  }

  // auto pro_matrix = AGMG::get_prolongation_matrix(T, result.first.second);
    
  // for(int i = 0; i < pro_matrix.rows(); i++) {
  //   SparseVector<double> curr = pro_matrix.row(i);
  //   for(SparseVector<double>::InnerIterator j(curr); j; ++j) {
  //     cout << (1 + (i/100)) << " " << (1 + (i % 100)) << " " << j.index() << endl;
  //   }
  // }

  return 0;
}