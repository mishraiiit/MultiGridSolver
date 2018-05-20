#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <fstream>
#include "GPUDebug.cu"

__global__ void computeRowColAbsSum(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, bool * ising0, float ktg, int iteration) {

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

__host__ void comptueSiHost(MatrixCSR * matrix_csr, MatrixCSC * matrix_csc, float * output) {    
    for(int id = 0; id < matrix_csr->rows; id++) {

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
}

__host__ __device__ float muij(int i, int j, MatrixCSR * matrix_csr, float * Si) {
    float aii = getElementMatrixCSR(matrix_csr, i, i);
    float ajj = getElementMatrixCSR(matrix_csr, j, j);
    float aij = getElementMatrixCSR(matrix_csr, i, j);
    float aji = getElementMatrixCSR(matrix_csr, j, i);

    float num = 2 * (1 / ((1 / aii) + (1 / ajj)));
    float den = (- (aij + aji) / 2) + 1 / ( ( 1 / (aii - Si[i])) + (1 / (ajj - Si[j])) );
    return num / den;
}

__global__ void sortNeighbourList(MatrixCSR * matrix, MatrixCSR * neighbour_list, float * Si, bool * allowed, float ktg, bool * ising0) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= matrix->rows)
        return;

    int row_start = matrix->i[id];
    int row_end = matrix->i[id + 1];

    for(int i = row_start; i < row_end; i++) {
        for(int j = i + 1; j < row_end; j++) {
            int id1 = neighbour_list->j[i];
            int id2 = neighbour_list->j[j];
            if(muij(id, id2, matrix, Si) < muij(id, id1, matrix, Si)) {
                swap_variables(neighbour_list->j[i], neighbour_list->j[j]);
                swap_variables(neighbour_list->val[i], neighbour_list->val[j]);
            }
        }
    }

    for(int i = row_start; i < row_end; i++) {
        int id1 = neighbour_list->j[i];
        float mij = muij(id, id1, matrix, Si);
        allowed[i] = (0 < mij && mij <= ktg);
        if(ising0[id] || ising0[id1]) {
            allowed[i] = false;
        }
    }    
}

__global__ void printNeighbourList(MatrixCSR * matrix, MatrixCSR * neighbour_list, float * Si) {

    for(int id = 0; id < neighbour_list->rows; id++) {
        int row_start = neighbour_list->i[id];
        int row_end = neighbour_list->i[id + 1];

        printf("Neighbours for %d\n", id);
        for(int i = row_start; i < row_end; i++) {
            printf("%d %lf\n", neighbour_list->j[i], muij(id, neighbour_list->j[i], matrix, Si));        
        }
    }
    printf("\n");
}

__global__ void mis(MatrixCSR * matrix, int * inmis) {
    for(int i = 0; i < 10000; i++) {
        if(((i / 100) + (i % 100)) % 2 == 0) {
            inmis[i] = true;
        } else {
            inmis[i] = false;
        }
    }
}

__global__ void aggregation_initial(int n, int * paired_with) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < n) {
        paired_with[id] = -1;
    }
}

__device__ bool okay(int i, int j, MatrixCSR * matrix, float * Si) {
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
__global__ void aggregation(int n, MatrixCSR * neighbour_list, int * paired_with, bool * allowed, MatrixCSR * matrix, float * Si, int distance, bool * ising0, int * bfs_distance, int levels) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;
    if(ising0[i]) return;
    if(bfs_distance[i]  % levels != distance) return;
    if(paired_with[i] != -1) return;

    for(int j = neighbour_list->i[i]; j < neighbour_list->i[i + 1]; j++) {
        if(!allowed[j]) continue;
        int possible_j = neighbour_list->j[j];
        if(bfs_distance[i] >= bfs_distance[possible_j]) continue;
        if(!okay(i, possible_j, matrix, Si)) continue;
        if(atomicCAS(&paired_with[possible_j], -1, i) == -1) {
            paired_with[i] = possible_j;
            paired_with[possible_j] = i;
            // printf("%d %d\n", i, possible_j);
            return;
        }
    }

    paired_with[i] = i;
    // printf("%d %d\n", i, paired_with[i]);
}

__global__ void bfs_frontier_kernel(MatrixCSR * matrix, bool * visited, int * distance, bool * frontier, int * new_found) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= matrix->rows) return;
    if(frontier[i]) {
        visited[i] = true;
        frontier[i] = false;
        * new_found = 1;
        for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
            int nj = matrix->j[j];
            if(!visited[nj]) {
                frontier[nj] = true;
                distance[nj] = distance[i] + 1;
            }
        }
    }
}

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

__global__ void gpu_prefix_sum_kernel(int n, int * useful_pairs) {
    for(int i = 1; i < n; i++) {
        useful_pairs[i] += useful_pairs[i - 1];
    }
}

void gpu_prefix_sum(int n, int * useful_pairs) {
    gpu_prefix_sum_kernel <<<1,1>>> (n, useful_pairs);
}

int * bfs(int n, MatrixCSR * matrix_gpu) {
    bool * visited;
    cudaMalloc(&visited, sizeof(bool) * n);

    int * distance;
    cudaMalloc(&distance, sizeof(int) * n);

    bool * frontier;
    cudaMalloc(&frontier, sizeof(int) * n);

    assign<<<1,1>>> (&frontier[0], 1);
    assign<<<1,1>>> (&distance[0], 0);

    int * new_found;
    cudaMallocManaged(&new_found, sizeof(int));

    do {
        * new_found = false;
        bfs_frontier_kernel <<<(n + 1024 - 1)/ 1024, 1024>>>(matrix_gpu, visited, distance, frontier, new_found);
        cudaDeviceSynchronize();
    } while(* new_found);

    return distance;

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