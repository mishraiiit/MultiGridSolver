#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "GPUDebug.cu"


// This is used in G0 computation in the research paper.
// To form the set G0, we need to compute the sum of the absolute values of the elements in the row and column of the matrix (except the diagonal elements).
// This stores whether the element is in G0 or not (in the hash table type structure, ising0[id] = 1 means the element is in G0).
__global__ void computeRowColAbsSum(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, int * ising0, float ktg, int iteration) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= matrix_csr->rows)
        return;

    int row_start = matrix_csr->i[id];
    int row_end = matrix_csr->i[id + 1];

    int col_start = matrix_csc->j[id];
    int col_end = matrix_csc->j[id + 1];

    float ans = 0;
    while(row_start < row_end || col_start < col_end) {
        if(row_start < row_end && col_start < col_end) {
            if(matrix_csr->j[row_start] < matrix_csc->i[col_start]) {
                if(matrix_csr->j[row_start] != id)
                    ans += abs(matrix_csr->val[row_start]) / 2;
                row_start++;
            } else if(matrix_csr->j[row_start] > matrix_csc->i[col_start]) {
                if(matrix_csc->i[col_start] != id)
                    ans += abs(matrix_csc->val[col_start]) / 2;
                col_start++;
            } else {
                if(matrix_csr->j[row_start] != id)
                    ans += abs(matrix_csr->val[row_start] + matrix_csc->val[col_start]) / 2;
                row_start++;
                col_start++;
            }
        } 
        else if(row_start < row_end) {
            if(matrix_csr->j[row_start] != id)
                ans += abs(matrix_csr->val[row_start]) / 2;
            row_start++;
        } else {
            if(matrix_csc->i[col_start] != id)
                ans += abs(matrix_csc->val[col_start]) / 2;
            col_start++;
        }
    }
    float aii = getElementMatrixCSR(matrix_csr, id, id);
    float rhs = (ktg / (ktg - 2)) * ans;
    
    if(iteration == 1)
        ising0[id] = aii >= rhs;
    else
        ising0[id] = 0;
}

// This is used in the computation of Si in the research paper.
// Si[i] is the negative sum of the absolute values of the elements in the row and column of the matrix (except the diagonal elements) of the node i.
__global__ void comptueSi(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, float * output) {
int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= matrix_csr->rows)
        return;

    int row_start = matrix_csr->i[id];
    int row_end = matrix_csr->i[id + 1];

    int col_start = matrix_csc->j[id];
    int col_end = matrix_csc->j[id + 1];

    float ans = 0;
    while(row_start < row_end || col_start < col_end) {
        if(row_start < row_end && col_start < col_end) {
            if(matrix_csr->j[row_start] < matrix_csc->i[col_start]) {
                if(matrix_csr->j[row_start] != id)
                    ans += matrix_csr->val[row_start] / 2;
                row_start++;
            } else if(matrix_csr->j[row_start] > matrix_csc->i[col_start]) {
                if(matrix_csc->i[col_start] != id)
                    ans += matrix_csc->val[col_start] / 2;
                col_start++;
            } else {
                if(matrix_csr->j[row_start] != id)
                    ans += (matrix_csr->val[row_start] + matrix_csc->val[col_start]) / 2;
                row_start++;
                col_start++;
            }
        } 
        else if(row_start < row_end) {
            if(matrix_csr->j[row_start] != id)
                ans += matrix_csr->val[row_start] / 2;
            row_start++;
        } else {
            if(matrix_csc->i[col_start] != id)
                ans += matrix_csc->val[col_start] / 2;
            col_start++;
        }
    }

    output[id] = -ans;
}

// This is used in the computation of the allowed pairs in the research paper.
// muij is the function used to compute the allowed pairs.
// muij(i, j) = 2 / (1/aii + 1/ajj) / ((- (aij + aji) / 2) + 1 / ( ( 1 / (aii - Si[i])) + (1 / (ajj - Si[j])) ))
// We are looking for pairs (i, j) such that muij(i, j) <= ktg and such j for which j is minimum.
__host__ __device__ float muij(int i, int j, MatrixCSR * matrix_csr, float * Si) {
    float aii = getElementMatrixCSR(matrix_csr, i, i);
    float ajj = getElementMatrixCSR(matrix_csr, j, j);
    float aij = getElementMatrixCSR(matrix_csr, i, j);
    float aji = getElementMatrixCSR(matrix_csr, j, i);

    float num = 2 * (1 / ((1 / aii) + (1 / ajj)));
    float den = (- (aij + aji) / 2) + 1 / ( ( 1 / (aii - Si[i])) + (1 / (ajj - Si[j])) );
    return num / den;
}

// This is used in the computation of the allowed pairs in the research paper.
// We are looking for pairs (i, j) such that muij(i, j) <= ktg and such j for which j is minimum.
// We sort the neighbour list of i based on the muij value.
// We then mark the pairs (i, j) as allowed if muij(i, j) <= ktg and such j for which j is minimum.
// The CSR matrix is esentially seen as a graph, represented as adjacent list (neighbour_list). We sort the CSR in place, so that
// while aggregating, we can access the neighbour list in a sorted manner.
// Since the graph is sparse, each element doesn't have many neighbours.
// Allowed is an array of size neighbour_list->i[matrix->rows] (so nnz count) which tells whether the pair (i, j) is allowed or not.
__global__ void sortNeighbourList(MatrixCSR * matrix, MatrixCSR * neighbour_list, float * Si, int * allowed, float ktg, int * ising0) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= matrix->rows)
        return;

    int row_start = matrix->i[id];
    int row_end = matrix->i[id + 1];

    #ifdef THRUST_SORT

        auto l = [&id, &matrix, &Si](int a, int b) {
            return muij(id, a, matrix, Si) < muij(id, b, matrix, Si);
        };

        thrust::stable_sort(thrust::seq, neighbour_list->j + row_start, neighbour_list->j + row_end, l);

    #else
        // This section is mostly here for debugging purposes.
        for(int i = row_start; i < row_end; i++) {
            for(int j = i + 1; j < row_end; j++) {
                int id1 = neighbour_list->j[i];
                int id2 = neighbour_list->j[j];
                if(muij(id, id2, matrix, Si) < muij(id, id1, matrix, Si)) {
                    swap_variables(neighbour_list->j[i], neighbour_list->j[j]);
                }
            }
        }
    
    #endif

    for(int i = row_start; i < row_end; i++) {
        int id1 = neighbour_list->j[i];
        float mij = muij(id, id1, matrix, Si);
        allowed[i] = (0 < mij && mij <= ktg);
        if(ising0[id] || ising0[id1]) {
            allowed[i] = false;
        }
    }    
}

// This checks the condition from the research paper: aii - Si + ajj - Sj >= 0.
// We access the matrix elements using the getElementMatrixCSR function (which uses a binary search to access the elements).
__device__ int okay(int i, int j, MatrixCSR * matrix, float * Si) {
    return (getElementMatrixCSR(matrix, i, i) - Si[i] + getElementMatrixCSR(matrix, j, j) - Si[j] >= 0);
}

/*
    Comments:
    This is the main function for initial_std::pairwise_aggregation.
    n is the number of rows in the matrix.
    neighbour_list is the matrix (adjacency matrix) in CSR format.
    allowed marks the positions in the neighbour_list which are not useful
    (the links with which aggregation shouldn't be formed).
    distance tells the distance of the nodes in which aggregation is done on in this kernel. Odd or Even.
    Si is the array containing the Si values in the paper.
    ising0 contains the node which are to be kept out of aggregation.
    bfs_distance tells the distance of a node from node 0.
*/

#ifdef AGGREGATION_WORK_EFFICIENT

    __global__ void aggregation(int n, MatrixCSR * neighbour_list , int * paired_with, int * allowed, MatrixCSR * matrix, float * Si, int distance, int * ising0, int * bfs_distance, MatrixCSR * distance_csr, int offset, int total) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if(i >= total) return;
        i = distance_csr->j[offset + i];
        if(ising0[i]) return;
        int current_distance = bfs_distance[i];
        assert(current_distance == distance);
        if(paired_with[i] != -1) return;
        for(int j = neighbour_list->i[i]; j < neighbour_list->i[i + 1]; j++) {
            if(!allowed[j]) continue;
            int possible_j = neighbour_list->j[j];
            if(current_distance == bfs_distance[possible_j]) continue;
            if(!okay(i, possible_j, matrix, Si)) continue;
            if(atomicCAS(&paired_with[possible_j], -1, i) == -1) {
                paired_with[i] = possible_j;
                return;
            }
        }
        paired_with[i] = i;
    }

#else

    __global__ void aggregation(int n, MatrixCSR * neighbour_list, int * paired_with, int * allowed, MatrixCSR * matrix, float * Si, int distance, int * ising0, int * bfs_distance, int levels) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if(i >= n) return;
        if(ising0[i]) return;
        int current_distance = bfs_distance[i];
        if(current_distance  % levels != distance) return;
        if(paired_with[i] != -1) return;
        for(int j = neighbour_list->i[i]; j < neighbour_list->i[i + 1]; j++) {
            if(!allowed[j]) continue;
            int possible_j = neighbour_list->j[j];
            if(current_distance == bfs_distance[possible_j]) continue;
            if(!okay(i, possible_j, matrix, Si)) continue;
            if(atomicCAS(&paired_with[possible_j], -1, i) == -1) {
                paired_with[i] = possible_j;
                return;
            }
        }
        paired_with[i] = i;
    }

#endif

__global__ void get_useful_pairs(int n, int * paired_with, int * useful_pairs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n)
        return;
    if(paired_with[i] == -1) {
        useful_pairs[i] = 0;
    } else if(paired_with[i] < i) {
        useful_pairs[i] = 0;
    } else {
        useful_pairs[i] = 1;
    }
}

__global__ void mark_aggregations(int n, int * aggregations, int * useful_pairs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;
    int curr = useful_pairs[i];
    int prev = (i == 0) ? 0 : useful_pairs[i - 1];
    if(curr != prev) {
        aggregations[curr - 1] = i;
    }
}

__global__ void get_aggregations_count(int nc, int * aggregations, int * paired_with, int *aggregation_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nc) return;
    aggregation_count[i] = 1 + (aggregations[i] != paired_with[aggregations[i]]);
}

__global__ void create_p_matrix_transpose (int nc, int * aggregations, int * paired_with, int * aggregation_count, int * matrix_i, int * matrix_j, float * matrix_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nc) return;
    matrix_i[i + 1] = aggregation_count[i];

    int first = aggregations[i];
    int second = paired_with[first];

    int prev_sum = (i == 0) ? 0 : aggregation_count[i - 1];

    if(first == second) {
        matrix_j[prev_sum] = first;
        matrix_val[prev_sum] = 1;
    } else {
        matrix_j[prev_sum] = first;
        matrix_j[prev_sum + 1] = second;
        matrix_val[prev_sum] = 1;
        matrix_val[prev_sum + 1] = 1;
    }
}