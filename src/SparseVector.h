#ifndef SPARSE_VECTOR_H
#define SPARSE_VECTOR_H
#include <vector>
#include "DenseVector.h"

class DenseVector;
class SparseVector;
class DenseMatrix;
class SparseMatrix;

class SparseVector {
    private:
        std::vector<std::pair<int, double> > _data; 
        double _abs_sum;
        double _sum;
    public:
        int size;
        std::vector<std::pair<int, double> > & getData();
        SparseVector(int size, std::vector<std::pair<int, double> > data);
        double operator[] (int index);
        SparseVector operator+(SparseVector vec);
        double operator* (SparseVector vec);
        double operator* (DenseVector vec);
        bool operator == (SparseVector vec);
        bool operator != (SparseVector vec);
        DenseVector toDenseVector();
        double abs_sum();
        double sum();
        void print();
};

#endif