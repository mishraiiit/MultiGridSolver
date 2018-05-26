#include "MatrixIO.cu"
#include "MatrixAccess.cu"
#include "MatrixOperations.cu"
#include "TicToc.cpp"
#include "GPUDebug.cu"
#include "Aggregation.cu"
#include "BFS.cu"
#include "PrefixSum.cu"
#include <cusparse.h>
#include <string>

// #define SKIP_LEVELS 2

int main(int argc, char * argv[]) {

    std::string matrixname;
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
    ktg = std::stod(argv[2]);
    npass = std::stoi(argv[3]);
    tou = std::stod(argv[4]);

    cusparseHandle_t  cudasparse_handle;
    cusparseCreate(&cudasparse_handle);


    TicToc readtime("Read time total");
    readtime.tic();

    MatrixCSR * P_cumm = NULL; // output will be in this.
    std::string filename = "../matrices/" + matrixname + ".mtx";

    auto A_CSRCPU = readMatrixCPUMemoryCSR(filename);
    auto A_CSR = deepCopyMatrixCSRCPUtoGPU(A_CSRCPU);

    readtime.toc();

    int nnz_initial = A_CSRCPU->nnz;

    TicToc cudaalloctime("cudaalloctime");
    cudaalloctime.tic();

    float * Si;
    assert(cudaMalloc(&Si, sizeof(float) * A_CSRCPU->rows) == cudaSuccess);

    bool * ising0;
    assert(cudaMalloc(&ising0, sizeof(bool) * A_CSRCPU->rows) == cudaSuccess);

    bool * allowed;
    assert(cudaMalloc(&allowed, sizeof(bool) * A_CSRCPU->nnz) == cudaSuccess);

    int * paired_with;
    assert(cudaMalloc(&paired_with, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);

    int * useful_pairs;
    assert(cudaMalloc(&useful_pairs, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);
    
    int * aggregations;
    assert(cudaMalloc(&aggregations, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);

    int * aggregation_count;
    assert(cudaMalloc(&aggregation_count, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);

    cudaalloctime.toc();

    TicToc main_timer("Main timer");
    main_timer.tic();

    for(int pass = 1; pass <= npass; pass++) {

        A_CSRCPU = shallowCopyMatrixCSRGPUtoCPU(A_CSR);

        int nnz_now = A_CSRCPU->nnz;
        if(nnz_now <= nnz_initial / tou) break;

        auto neighbour_list = deepCopyMatrixCSRGPUtoGPU(A_CSR);
        auto A_CSC = convertCSRGPU_cudaSparse(A_CSR, cudasparse_handle);
        
        TicToc rowcolsum("Row Col abs sum");
        rowcolsum.tic();
        int number_of_blocks = (A_CSRCPU->rows + 1024 - 1) / 1024;
        int number_of_threads = 1024;
        computeRowColAbsSum <<<number_of_blocks, number_of_threads>>>
        (A_CSR, A_CSC, ising0, ktg, pass);
        cudaDeviceSynchronize();
        rowcolsum.toc();

        TicToc sicomputation("Si computation");
        sicomputation.tic();
        comptueSi<<<number_of_blocks, number_of_threads>>> (A_CSR, A_CSC, Si);
        // debugmuij<<<1,1>>> (A_CSR, Si);
        cudaDeviceSynchronize();
        sicomputation.toc();

        TicToc bfstime("BFS time...");
        bfstime.tic();
        int max_distance;
        int * bfs_distance = bfs(A_CSRCPU->rows, A_CSR, &max_distance);
        cudaDeviceSynchronize();
        bfstime.toc();

        TicToc sortcomputation("Sort computation");
        sortcomputation.tic();
        sortNeighbourList<<<number_of_blocks, number_of_threads>>>
        (A_CSR, neighbour_list, Si, allowed, ktg, ising0);
        cudaDeviceSynchronize();
        sortcomputation.toc();

        TicToc aggregationtime("Aggregation time");
        aggregationtime.tic();

        aggregation_initial<<<number_of_blocks, number_of_threads>>>
        (A_CSRCPU->rows, paired_with);
        cudaDeviceSynchronize();

        #ifdef SKIP_LEVELS
            int skip_levels = SKIP_LEVELS;
        #else
            int skip_levels = max_distance + 1;
        #endif 

        for(int i = 0; i < skip_levels; i++) {
            aggregation<<<number_of_blocks, number_of_threads>>>
            (A_CSRCPU->rows, neighbour_list, paired_with, allowed, A_CSR, Si, i,
             ising0, bfs_distance, skip_levels);
            cudaDeviceSynchronize();
        }

        aggregationtime.toc();

        TicToc get_usefule_pairs_time("Get useful_pairs time");
        get_usefule_pairs_time.tic();
        get_useful_pairs<<<number_of_blocks, number_of_threads>>>
        (A_CSRCPU->rows, paired_with, useful_pairs);
        cudaDeviceSynchronize();
        get_usefule_pairs_time.toc();


        TicToc prefix_sum("Sum kernel");
        prefix_sum.tic();
        gpu_prefix_sum(A_CSRCPU->rows, useful_pairs);
        // cudaMemcpy(useful_pairs_cpu_prefix, useful_pairs,
        //     sizeof(int) * A_CSRCPU->rows, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        prefix_sum.toc();

        TicToc P_matrix_creation_time("Time to P matrix");
        P_matrix_creation_time.tic();

        int nc;
        assert(cudaMemcpy(&nc, useful_pairs + A_CSRCPU->rows - 1,
            sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

        mark_aggregations <<<number_of_blocks, number_of_threads>>>
        (A_CSRCPU->rows, aggregations, useful_pairs);
        cudaDeviceSynchronize();

        get_aggregations_count <<< (nc + 1024 - 1) / 1024, 1024 >>>
        (nc, aggregations, paired_with, aggregation_count);
        cudaDeviceSynchronize();

        gpu_prefix_sum(nc, aggregation_count);
        int nnz_in_p_matrix;
        assert(cudaMemcpy(&nnz_in_p_matrix, aggregation_count + nc - 1,
            sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
        

        MatrixCSR * P_transpose_shallow_cpu = 
            (MatrixCSR *) malloc(sizeof(MatrixCSR));

        P_transpose_shallow_cpu->rows = nc;
        P_transpose_shallow_cpu->cols = A_CSRCPU->rows;
        P_transpose_shallow_cpu->nnz = nnz_in_p_matrix;

        assert(cudaMalloc(&P_transpose_shallow_cpu->i,
            sizeof(int) * (P_transpose_shallow_cpu->rows + 1)) == cudaSuccess);
        assert(cudaMalloc(&P_transpose_shallow_cpu->j,
            sizeof(int) * (P_transpose_shallow_cpu->nnz)) == cudaSuccess);
        assert(cudaMalloc(&P_transpose_shallow_cpu->val,
            sizeof(float) * (P_transpose_shallow_cpu->nnz)) == cudaSuccess);

        create_p_matrix_transpose <<< (nc + 1024 - 1) / 1024, 1024>>>
        (nc, aggregations, paired_with, aggregation_count,
            P_transpose_shallow_cpu->i, P_transpose_shallow_cpu->j,
            P_transpose_shallow_cpu->val);
        cudaDeviceSynchronize();

        MatrixCSR * P_transpose_gpu;
        assert(cudaMalloc(&P_transpose_gpu, sizeof(MatrixCSR)) == cudaSuccess);
        assert(cudaMemcpy(P_transpose_gpu, P_transpose_shallow_cpu, sizeof(MatrixCSR),
            cudaMemcpyHostToDevice) == cudaSuccess);
        cudaDeviceSynchronize();
        P_matrix_creation_time.toc();

        TicToc time_transpose("Time taken csr2csc");
        time_transpose.tic();

        MatrixCSR * P_gpu = transposeCSRGPU_cudaSparse(P_transpose_gpu, cudasparse_handle);
        
        MatrixCSR * newA_gpu = spmatrixmult_cudaSparse(P_transpose_gpu,
            spmatrixmult_cudaSparse(A_CSR, P_gpu, cudasparse_handle),
            cudasparse_handle);

        if(P_cumm == NULL)
            P_cumm = deepCopyMatrixCSRGPUtoGPU(P_gpu);
        else
            P_cumm = spmatrixmult_cudaSparse(P_cumm, P_gpu, cudasparse_handle);

        
        cudaFree(bfs_distance);
        freeMatrixCSRGPU(P_gpu);
        freeMatrixCSRGPU(P_transpose_gpu);
        freeMatrixCSRGPU(A_CSR);
        freeMatrixCSCGPU(A_CSC);

        A_CSR = newA_gpu;
    }

    main_timer.toc();

    assert(cudaFree(Si) == cudaSuccess);
    assert(cudaFree(ising0) == cudaSuccess);
    assert(cudaFree(allowed) == cudaSuccess);
    assert(cudaFree(paired_with) == cudaSuccess);
    assert(cudaFree(useful_pairs) == cudaSuccess);
    assert(cudaFree(aggregations) == cudaSuccess);
    assert(cudaFree(aggregation_count) == cudaSuccess);


    writeMatrixCSRCPU(std::string("../matrices/") + matrixname + \
        std::string("promatrix.mtx"), deepCopyMatrixCSRGPUtoCPU(P_cumm));
 
    freeMatrixCSRGPU(A_CSR);

    return 0;
}
