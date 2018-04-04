#include <iostream>
#include "AGMG.cpp"
#include <typeinfo>
#include <Eigen/Sparse>
#include <bits/stdc++.h>
using namespace std;
using namespace Eigen;

typedef SparseMatrix<double, RowMajor> SMatrix;

int main() {

	std::string s("../matrices/poisson10000.txt");
	std::ifstream fin(s);
    if(fin.fail()) {
        std::cout << "File " << s << " not found.";
        exit(1);
    }
    int _rows, _cols, _nnz;
    fin >> _rows >> _cols >> _nnz;
    SMatrix T(_rows, _cols);
    
    for(int i = 0; i < _nnz; i++) {
        int x, y;
        double value;
        fin >> x >> y >> value;
        x--; y--;
        T.insert(x, y) = value;
    }

    auto result = AGMG::multiple_pairwise_aggregation(T.rows(), T, 8, 2, 4);
    // cout << T.rows() << " " << result.first.first << endl;
   //  auto pro_matrix = AGMG::get_prolongation_matrix(T, result.first.second);
    
   //  for(int i = 0; i < pro_matrix.rows(); i++) {
   //    SparseVector<double> curr = pro_matrix.row(i);
   //    for(SparseVector<double>::InnerIterator j(curr); j; ++j) {
   //      cout << i + 1 << " " << j.index() + 1 << " " << j.value() << endl;
   //    }
   //  }

    auto pro_matrix = AGMG::get_prolongation_matrix(T, result.first.second);
    
    for(int i = 0; i < pro_matrix.rows(); i++) {
      SparseVector<double> curr = pro_matrix.row(i);
      for(SparseVector<double>::InnerIterator j(curr); j; ++j) {
        cout << (1 + (i/100)) << " " << (1 + (i % 100)) << " " << j.index() << endl;
      }
    }

  	return 0;
}