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

#ifdef AGGREGATION_WORK_EFFICIENT

__global__ void set_nodes(int n, int * nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    nodes[i] = i;
}

__global__ void compute_offsets(int n, int * row_ptr, int * col_val, int * distance) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    else if(i == 0) {
        row_ptr[0] = 0;
    } else {
        int prev = distance[col_val[i - 1]];
        int curr = distance[col_val[i]];
        if(curr != prev) {
            row_ptr[curr] = i;
        }
    }
}

std::pair<int *, MatrixCSR *> bfs(int n, MatrixCSR * matrix_gpu, int * max_distance) {
    printInfo("Normal BFS running", 8);
    int * visited;
    cudaMalloc(&visited, sizeof(int) * n);

    int * distance;
    cudaMalloc(&distance, sizeof(int) * n);

    int * frontier;
    cudaMalloc(&frontier, sizeof(int) * n);

    int * nodes;
    cudaMallocManaged(&nodes, sizeof(int) * n);

    set_nodes <<< (n + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS, NUMBER_OF_THREADS >>> (n, nodes);

    initialize_array(n, visited, false);
    initialize_array(n, frontier, false);

    assign<<<1,1>>> (&frontier[0], 1);
    assign<<<1,1>>> (&distance[0], 0);

    int * new_found;
    cudaMallocManaged(&new_found, sizeof(int));
    * max_distance = 0;

    while(true) {

        #ifdef DEBUG
            {
                printf("\nfrontier data:  \n");
                printf("Level :  %d\n", * max_distance);
                int * temp = (int *) malloc(sizeof(int) * n);
                cudaMemcpy(temp, frontier, sizeof(int) * n, cudaMemcpyDeviceToHost);
                for(int i = 0; i < n; i++) {
                    if(temp[i]) {
                        printf("%d ", i);
                    }
                }
                printf("\n");
                free(temp);
            }
        #endif

        * new_found = false;
        bfs_frontier_kernel <<<(n + NUMBER_OF_THREADS - 1)/ NUMBER_OF_THREADS, NUMBER_OF_THREADS>>>(matrix_gpu, visited, distance, frontier, new_found);
        cudaDeviceSynchronize();
        if(*new_found)
            * max_distance = * max_distance + 1;
        else
            break;
    };

    cudaFree(visited);
    cudaFree(frontier);

    thrust::sort(thrust::device, nodes, nodes + n, [distance] __device__ (int u, int v) {
        return distance[u] < distance[v];
    });

    MatrixCSR * toReturn = NULL;
    cudaMallocManaged(&toReturn, sizeof(MatrixCSR));
    cudaMallocManaged(&toReturn->i, sizeof(int) * ((*max_distance) + 2));
    toReturn->rows = *max_distance + 1;
    toReturn->cols = n;
    toReturn->nnz = n;
    toReturn->j = nodes;
    toReturn->i[(*max_distance) + 1] = n;

    compute_offsets <<< (n + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS, NUMBER_OF_THREADS >>> (n, toReturn->i, toReturn->j, distance);

    return {distance, toReturn};

}

#else

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

#endif

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

#ifdef AGGREGATION_WORK_EFFICIENT

__global__ void write_vertex_fronteir (int edge_fronteir_size, int * vertex_fronteir, int * edge_fronteir, int * allowed, int offset, MatrixCSR * level_matrix) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= edge_fronteir_size) return;
    int prev = (i == 0) ? 0 : allowed[i - 1];
    int curr = allowed[i];
    if(prev != curr) {
        int node = edge_fronteir[i];
        level_matrix->j[offset + prev] = node;
        vertex_fronteir[prev] = node;
    }
}

#else
    
__global__ void write_vertex_fronteir (int edge_fronteir_size, int * vertex_fronteir, int * edge_fronteir, int * allowed) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= edge_fronteir_size) return;
    int prev = (i == 0) ? 0 : allowed[i - 1];
    int curr = allowed[i];
    if(prev != curr) {
        vertex_fronteir[prev] = edge_fronteir[i];
    }
}

#endif

#ifdef AGGREGATION_WORK_EFFICIENT

std::pair<int *, MatrixCSR *> bfs_work_efficient(int n, MatrixCSR * matrix_gpu, int * max_distance) {
    printInfo("Work efficient BFS running", 8);
    int * vertex_fronteir;
    cudaMalloc(&vertex_fronteir, sizeof(int) * n);

    int * visited_by;
    cudaMalloc(&visited_by, sizeof(int) * n);
    
    int * edge_fronteir;
    cudaMalloc(&edge_fronteir, sizeof(int) * n);

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


    MatrixCSR * toReturn = NULL;
    #ifdef AGGREGATION_WORK_EFFICIENT
        cudaMallocManaged(&toReturn, sizeof(MatrixCSR));
        cudaMallocManaged(&toReturn->i, sizeof(int) * (n + 1));
        cudaMallocManaged(&toReturn->j, sizeof(int) * n);
        toReturn->nnz = n;
        int offset = 1;
        toReturn->j[0] = 0;
        toReturn->i[0] = 0;
        toReturn->i[1] = offset;
    #endif

    int iterations = 1;
    while(true) {
        int blocks = (vertex_fronteir_size + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS;
        int threads = NUMBER_OF_THREADS;
        int edge_fronteir_size;
        write_sizes <<<blocks, threads >>> (vertex_fronteir_size, offsets, vertex_fronteir, matrix_gpu);
        prefixSumGPU(offsets, vertex_fronteir_size);
        cudaMemcpy(&edge_fronteir_size, offsets + vertex_fronteir_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
        assert(edge_fronteir_size < n);
        assert(edge_fronteir_size != 0);
        write_edge_fronteir <<<blocks, threads >>> (vertex_fronteir_size, offsets, matrix_gpu, edge_fronteir, vertex_fronteir, visited_by, allowed);
        culling <<<blocks, threads >>> (vertex_fronteir_size, offsets, vertex_fronteir, edge_fronteir, allowed, matrix_gpu, distance, visited_by);
        prefixSumGPU(allowed, edge_fronteir_size);
        cudaMemcpy(&vertex_fronteir_size, allowed + edge_fronteir_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
        if(vertex_fronteir_size == 0)
            break;
        #ifdef AGGREGATION_WORK_EFFICIENT
            write_vertex_fronteir <<< (edge_fronteir_size + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS, NUMBER_OF_THREADS >>> (edge_fronteir_size, vertex_fronteir, edge_fronteir, allowed, offset, toReturn);
            offset += vertex_fronteir_size;
            toReturn->i[iterations + 1] = offset;
        #else
            write_vertex_fronteir <<< (edge_fronteir_size + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS, NUMBER_OF_THREADS >>> (edge_fronteir_size, vertex_fronteir, edge_fronteir, allowed);
        #endif
        *max_distance = ++iterations;
    }

    #ifdef AGGREGATION_WORK_EFFICIENT
        toReturn->rows = *max_distance;
        toReturn->cols = n;
    #endif

    cudaFree(vertex_fronteir);
    cudaFree(visited_by);
    cudaFree(edge_fronteir);
    cudaFree(offsets);
    cudaFree(allowed);
    
    return {distance, toReturn};
}

#else

std::pair<int *, MatrixCSR *> bfs_work_efficient(int n, MatrixCSR * matrix_gpu, int * max_distance) {
    fprintf(stderr, "Work efficient BFS running\n");
    int * vertex_fronteir;
    cudaMalloc(&vertex_fronteir, sizeof(int) * n);

    int * visited_by;
    cudaMalloc(&visited_by, sizeof(int) * n);
    
    int * edge_fronteir;
    cudaMalloc(&edge_fronteir, sizeof(int) * n);

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
        int blocks = (vertex_fronteir_size + 1024 - 1) / 1024;
        int threads = 1024;
        int edge_fronteir_size;
        write_sizes <<<blocks, threads >>> (vertex_fronteir_size, offsets, vertex_fronteir, matrix_gpu);
        prefixSumGPU(offsets, vertex_fronteir_size);
        cudaMemcpy(&edge_fronteir_size, offsets + vertex_fronteir_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
        assert(edge_fronteir_size < n);
        if(edge_fronteir_size == 0)
              break;
        write_edge_fronteir <<<blocks, threads >>> (vertex_fronteir_size, offsets, matrix_gpu, edge_fronteir, vertex_fronteir, visited_by, allowed);
        culling <<<blocks, threads >>> (vertex_fronteir_size, offsets, vertex_fronteir, edge_fronteir, allowed, matrix_gpu, distance, visited_by);
        prefixSumGPU(allowed, edge_fronteir_size);
        write_vertex_fronteir <<< (edge_fronteir_size + 1024 - 1) / 1024, 1024 >>> (edge_fronteir_size, vertex_fronteir, edge_fronteir, allowed);
        cudaMemcpy(&vertex_fronteir_size, allowed + edge_fronteir_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(vertex_fronteir);
    cudaFree(visited_by);
    cudaFree(edge_fronteir);
    cudaFree(offsets);
    cudaFree(allowed);
    
    return {distance, NULL};
}

#endif

#endif
