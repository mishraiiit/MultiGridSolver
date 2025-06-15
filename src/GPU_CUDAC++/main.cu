#define BLELLOCH
#define BFS_WORK_EFFICIENT
#define NUMBER_OF_THREADS 1024
// #define THRUST_SORT
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
#include <vector>
#include <stdlib.h>

int main(int argc, char * argv[]) {

    std::string matrixname;
    double ktg;
    int npass;
    double tou;

    int indent = 4;
  
    if(argc != 5) {
        printf("Invalid arguments.\n");
        printf("First argument should be matrix file in .mtx format.\n");
        printf("Second argument should be the parameter ktg, default value is 10.\n");
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

    printLines();
    printConfig();
    printLines();


    TicToc readtime("Read time total", indent);
    readtime.tic();

    MatrixCSR * P_cumm = NULL; // output will be in this.
    std::string filename = "../../matrices/" + matrixname + ".mtx";

    auto A_CSRCPU = readMatrixCPUMemoryCSR(filename); // CSR, CPU
    auto A_CSR = deepCopyMatrixCSRCPUtoGPU(A_CSRCPU); // CSR, GPU

    readtime.toc();

    int nnz_initial = A_CSRCPU->nnz;

    TicToc cudaalloctime("cudaalloctime", indent);
    cudaalloctime.tic();

    float * Si;
    assert(cudaMalloc(&Si, sizeof(float) * A_CSRCPU->rows) == cudaSuccess);

    int * ising0;
    assert(cudaMalloc(&ising0, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);

    int * allowed;
    assert(cudaMalloc(&allowed, sizeof(int) * A_CSRCPU->nnz) == cudaSuccess);

    int * paired_with;
    assert(cudaMalloc(&paired_with, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);

    int * useful_pairs;
    assert(cudaMalloc(&useful_pairs, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);
    
    int * aggregations;
    assert(cudaMalloc(&aggregations, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);

    int * aggregation_count;
    assert(cudaMalloc(&aggregation_count, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);

    int * nodes;
    assert(cudaMalloc(&nodes, sizeof(int) * A_CSRCPU->rows) == cudaSuccess);

    cudaalloctime.toc();

    TicToc main_timer("AGMG Core Algorithm Time", indent);
    main_timer.tic();

    indent += 4;
    for(int pass = 1; pass <= npass; pass++) {

        A_CSRCPU = shallowCopyMatrixCSRGPUtoCPU(A_CSR);

        int nnz_now = A_CSRCPU->nnz;
        if(nnz_now <= nnz_initial / tou) break;

        printLines();
        printInfo(("PASS " + itoa(pass)).c_str(), indent - 4);

        auto neighbour_list = deepCopyMatrixCSRGPUtoGPU(A_CSR);
        auto A_CSC = convertCSRGPU_cudaSparse(A_CSR, cudasparse_handle);
        
        TicToc rowcolsum("Row Col abs sum", indent);
        rowcolsum.tic();
        int number_of_blocks = (A_CSRCPU->rows + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS;
        int number_of_threads = NUMBER_OF_THREADS;
        computeRowColAbsSum <<<number_of_blocks, number_of_threads>>>
        (A_CSR, A_CSC, ising0, ktg, pass);
        cudaDeviceSynchronize();

        rowcolsum.toc();

        TicToc sicomputation("Si computation", indent);
        sicomputation.tic();
        comptueSi<<<number_of_blocks, number_of_threads>>> (A_CSR, A_CSC, Si);

        cudaDeviceSynchronize();

        sicomputation.toc();

        TicToc bfstime("BFS time...", indent);
        bfstime.tic();
        int max_distance;

        std::pair<int *, MatrixCSR * > bfs_result = bfs_work_efficient(A_CSRCPU->rows, A_CSR, &max_distance);;

        int * bfs_distance = bfs_result.first;
        MatrixCSR * distance_csr = bfs_result.second;

        cudaDeviceSynchronize();
        bfstime.toc();

        TicToc sortcomputation("Sort computation", indent);
        sortcomputation.tic();
        sortNeighbourList<<<number_of_blocks, number_of_threads>>>
        (A_CSR, neighbour_list, Si, allowed, ktg, ising0);
        cudaDeviceSynchronize();

        sortcomputation.toc();

        TicToc aggregationtime("Aggregation time", indent);
        aggregationtime.tic();

        // Initialize the paired_with array to -1.
        initialize_array(A_CSRCPU->rows,  paired_with, -1);

        // Iterate over all the levels.
        // We find aggreagtes level by level.
        for(int i = 0; i <= max_distance; i++) {
            aggregation<<<number_of_blocks, number_of_threads>>>
            (A_CSRCPU->rows, neighbour_list, paired_with, allowed, A_CSR, Si, i,
                ising0, bfs_distance);
            cudaDeviceSynchronize();
        }



        aggregationtime.toc();

        TicToc get_usefule_pairs_time("Get useful_pairs time", indent);
        get_usefule_pairs_time.tic();
        // Counts one for a pair.
        // useful_pairs[i] = 1 if i is the leader of an aggregate.
        // Leader is defined as the node which is smaller.
        get_useful_pairs<<<number_of_blocks, number_of_threads>>>
        (A_CSRCPU->rows, paired_with, useful_pairs);
        cudaDeviceSynchronize();

        get_usefule_pairs_time.toc();


        // Computes prefix sum of useful_pairs.
        // In the end on the prefix sum array, we can number the pair by looking at useful_pairs[i] - useful_pairs[i - 1].
        TicToc prefix_sum("Sum kernel", indent);
        prefix_sum.tic();
        prefixSumGPU(useful_pairs, A_CSRCPU->rows);

        cudaDeviceSynchronize();
        prefix_sum.toc();

        TicToc P_matrix_creation_time("Time to P matrix", indent);
        P_matrix_creation_time.tic();

        int nc;
        // nc is the total number of aggregates.
        // Last element of useful_pairs is the total number of aggregates.
        assert(cudaMemcpy(&nc, useful_pairs + A_CSRCPU->rows - 1,
            sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

        // Aggregations array would be of size nc, and it will contain the leader of every aggregate.
        mark_aggregations <<<number_of_blocks, number_of_threads>>>
        (A_CSRCPU->rows, aggregations, useful_pairs);

        cudaDeviceSynchronize();

        // Calculates the size of every aggregate in aggregation_count array.
        get_aggregations_count <<< (nc + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS, NUMBER_OF_THREADS >>>
        (nc, aggregations, paired_with, aggregation_count);
        cudaDeviceSynchronize();

        // Computes prefix sum of aggregation_count.
        // The last element should sum to be n.
        prefixSumGPU(aggregation_count, nc);

        // nnz_in_p_matrix is the total number of non-zero elements in the P matrix.
        // Last element of aggregation_count is the total number of non-zero elements in the P matrix.
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

        assign<<<1,1>>> (&P_transpose_shallow_cpu->i[0], 0);
        create_p_matrix_transpose <<< (nc + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS, NUMBER_OF_THREADS>>>
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

        TicToc time_transpose("Time taken csr2csc", indent);
        time_transpose.tic();

        MatrixCSR * P_gpu = transposeCSRGPU_cudaSparse(P_transpose_gpu, cudasparse_handle);
        
        MatrixCSR * newA_gpu = spmatrixmult_cudaSparse(P_transpose_gpu,
            spmatrixmult_cudaSparse(A_CSR, P_gpu, cudasparse_handle),
            cudasparse_handle);

        if(P_cumm == NULL)
            P_cumm = deepCopyMatrixCSRGPUtoGPU(P_gpu);
        else {
            MatrixCSR * new_P_cumm = spmatrixmult_cudaSparse(P_cumm, P_gpu, cudasparse_handle);
            cudaFree(P_cumm);
            P_cumm = new_P_cumm;
        }

        
        cudaFree(bfs_distance);
        if (distance_csr != NULL) {
            freeMatrixCSRGPU(distance_csr);
            distance_csr = NULL;
        }
        freeMatrixCSRGPU(P_gpu);
        freeMatrixCSRGPU(P_transpose_gpu);
        freeMatrixCSRGPU(A_CSR);
        freeMatrixCSCGPU(A_CSC);

        printInfo(("Matrix size reduced to : " + itoa(nc)).c_str(), indent);

        A_CSR = newA_gpu;
    }
    printLines();

    main_timer.toc();

    assert(cudaFree(Si) == cudaSuccess);
    assert(cudaFree(ising0) == cudaSuccess);
    assert(cudaFree(allowed) == cudaSuccess);
    assert(cudaFree(paired_with) == cudaSuccess);
    assert(cudaFree(useful_pairs) == cudaSuccess);
    assert(cudaFree(aggregations) == cudaSuccess);
    assert(cudaFree(aggregation_count) == cudaSuccess);


    writeMatrixCSRCPU(std::string("../../matrices/") + matrixname + \
        std::string("promatrix_gpu.mtx"), deepCopyMatrixCSRGPUtoCPU(P_cumm));
 
    freeMatrixCSRGPU(A_CSR);

    return 0;
}
