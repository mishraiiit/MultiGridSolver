#include <iostream>
#include "AGMG.cpp"
#include "MatrixIO.cpp"
#include <typeinfo>
#include <Eigen/Sparse>
#include <bits/stdc++.h>
#include <chrono>
using namespace std;
using namespace Eigen;

typedef SparseMatrix<double, RowMajor> SMatrix;

int main(int argc, char * argv[]) {

  string matrixname;
  double ktg;
  int npass;
  double tou;
  
  if(argc != 5) {
    printf("Invalid arguments.\n");
    printf("First argument should be matrix file in .mtx format.\n");
    printf("Second argument should be the parameter ktg, default value is 8.\n");
    printf("Third argument should be the parameter npass, default value is 2.\n");
    printf("Fourth argument should be the parameter tou, default value is 4.\n");
    exit(1);
  }

  matrixname = argv[1];
  ktg = stod(argv[2]);
  npass = stoi(argv[3]);
  tou = stod(argv[4]);


  SMatrix T = readMatrix(string("../matrices/") + matrixname + string(".mtx"));
  cerr << ktg << " " << npass << " " << tou << endl;


  auto start = std::chrono::system_clock::now();

  auto result = AGMG::multiple_pairwise_aggregation(T.rows(), T, ktg, npass, tou);
  auto pro_matrix = AGMG::get_prolongation_matrix(T, result.first.second);

  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> diff = end-start;
  std::cout << "Time for aggregation: " << diff.count() << " s\n";

  writeMatrix(string("../matrices/") + matrixname + string("promatrix.mtx"), pro_matrix);
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
