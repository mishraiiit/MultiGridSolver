#ifndef BFS
#define BFS
#include "MatrixIO.cu"
#include "PrefixSum.cu"
#include "GPUDebug.cu"

__global__ void bfs_frontier_kernel(MatrixCSR * matrix, int * visited, int * distance, int * frontier, int * new_found) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= matrix->rows) return;
    int new_found_local = 0;
    if(frontier[i]) {
        visited[i] = true;
        frontier[i] = false;
        for(int j = matrix->i[i]; j < matrix->i[i + 1]; j++) {
            int nj = matrix->j[j];
            if(!visited[nj]) {
		new_found_local = 1;
                frontier[nj] = true;
                distance[nj] = distance[i] + 1;
            }
        }
    }
    if(new_found_local == 1)
    * new_found = new_found_local;
}

std::pair<int *, MatrixCSR *> bfs(int n, MatrixCSR * matrix_gpu, int * max_distance) {
    printInfo("Normal BFS running", 8);
    int * visited;
    cudaMalloc(&visited, sizeof(int) * n);

    int * distance;
    cudaMalloc(&distance, sizeof(int) * n);

    int * frontier;
    cudaMalloc(&frontier, sizeof(int) * n);

    initialize_array(n, visited, false);
    initialize_array(n, frontier, false);

    assign<<<1,1>>> (&frontier[0], 1);
    assign<<<1,1>>> (&distance[0], 0);

    int * new_found;
    cudaMallocManaged(&new_found, sizeof(int));
    * max_distance = 0;
    do {
        * new_found = false;
        * max_distance = * max_distance + 1;
        bfs_frontier_kernel <<<(n + NUMBER_OF_THREADS - 1)/ NUMBER_OF_THREADS, NUMBER_OF_THREADS>>>(matrix_gpu, visited, distance, frontier, new_found);
        cudaDeviceSynchronize();
    } while(* new_found);

    cudaFree(visited);
    cudaFree(frontier);

    return {distance, NULL};

}

__global__ void write_sizes (int total, int * offsets, int * vertices, MatrixCSR * matrix) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= total) return;
    int vertex = vertices[i];
    offsets[i] = matrix->i[vertex + 1] - matrix->i[vertex];
}

__global__ void write_edge_fronteir (int total, int * offsets, MatrixCSR * matrix, int * edge_fronteir, int * vertices, int * visited_by, int * allowed) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= total) return;
    int vertex = vertices[i];
    int offset = (i == 0) ? 0 : offsets[i - 1];
    int count = 0;
    for(int i = matrix->i[vertex]; i < matrix->i[vertex + 1]; i++) {
        int v = matrix->j[i];
        edge_fronteir[offset + count] = v;
        if(visited_by[v] == -1) {
            visited_by[v] = vertex;
        }
        count++;
    }
}

__global__ void culling(int total, int * offsets, int * vertices, int * edges, int * allowed, MatrixCSR * matrix, int * distance, int * visited_by) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= total) return;
    int vertex = vertices[i];
    int dist = distance[vertex];
    int offset = (i == 0) ? 0 : offsets[i - 1];
    int count = 0;
    for(int i = matrix->i[vertex]; i < matrix->i[vertex + 1]; i++) {
        int v = matrix->j[i];
    if(visited_by[v] == vertex) {
            allowed[offset + count] = 1;
            distance[v] = dist + 1;
    } else {
        allowed[offset + count] = 0;
    }
        count++;
    }
}
    
__global__ void write_vertex_fronteir (int edge_fronteir_size, int * vertex_fronteir, int * edge_fronteir, int * allowed) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= edge_fronteir_size) return;
    int prev = (i == 0) ? 0 : allowed[i - 1];
    int curr = allowed[i];
    if(prev != curr) {
        vertex_fronteir[prev] = edge_fronteir[i];
    }
}


std::pair<int *, MatrixCSR *> bfs_work_efficient(int n, MatrixCSR * matrix_gpu, int * max_distance) {
    fprintf(stderr, "Work efficient BFS running\n");

    MatrixCSR * matrix_metadata = shallowCopyMatrixCSRGPUtoCPU(matrix_gpu);

    int * vertex_fronteir;
    cudaMalloc(&vertex_fronteir, sizeof(int) * n);

    int * visited_by;
    cudaMalloc(&visited_by, sizeof(int) * n);
    
    int * edge_fronteir;
    cudaMalloc(&edge_fronteir, sizeof(int) * matrix_metadata->nnz);

    int * offsets;
    cudaMalloc(&offsets, sizeof(int) * n);

    int * allowed;
    cudaMalloc(&allowed, sizeof(int) * n);

    int * distance;
    cudaMalloc(&distance, sizeof(int) * n);

    int vertex_fronteir_size = 1;

    initialize_array(n, visited_by, -1);

    assign<<<1,1>>> (vertex_fronteir, 0);
    assign<<<1,1>>> (distance, 0);
    assign<<<1,1>>> (visited_by, -2);

    int iterations = 1;
    while(vertex_fronteir_size != 0) {

        *max_distance = ++iterations;
        int blocks = (vertex_fronteir_size + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS;
        int threads = NUMBER_OF_THREADS;
        int edge_fronteir_size;
        write_sizes <<<blocks, threads >>> (vertex_fronteir_size, offsets, vertex_fronteir, matrix_gpu);
        prefixSumGPU(offsets, vertex_fronteir_size);
        cudaMemcpy(&edge_fronteir_size, offsets + vertex_fronteir_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
        assert(edge_fronteir_size < matrix_metadata->nnz);
        if(edge_fronteir_size == 0)
              break;
        write_edge_fronteir <<<blocks, threads >>> (vertex_fronteir_size, offsets, matrix_gpu, edge_fronteir, vertex_fronteir, visited_by, allowed);
        culling <<<blocks, threads >>> (vertex_fronteir_size, offsets, vertex_fronteir, edge_fronteir, allowed, matrix_gpu, distance, visited_by);
        prefixSumGPU(allowed, edge_fronteir_size);
        write_vertex_fronteir <<< (edge_fronteir_size + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS, NUMBER_OF_THREADS >>> (edge_fronteir_size, vertex_fronteir, edge_fronteir, allowed);
        cudaMemcpy(&vertex_fronteir_size, allowed + edge_fronteir_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(vertex_fronteir);
    cudaFree(visited_by);
    cudaFree(edge_fronteir);
    cudaFree(offsets);
    cudaFree(allowed);
    free(matrix_metadata);
    
    return {distance, NULL};
}

#endif
