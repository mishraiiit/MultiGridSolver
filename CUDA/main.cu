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


#define BLELLOCH
// #define THRUST_SORT
// #define DEBUG
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

    cudaalloctime.toc();

    TicToc main_timer("Main timer");
    main_timer.tic();

    for(int pass = 1; pass <= npass; pass++) {

        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("CSR CPU\n");
            printCSRCPU(deepCopyMatrixCSRGPUtoCPU(A_CSR));
        #endif

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
        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("Is in G0\n");
            int * temp_g0 = (int * ) malloc(sizeof(int) * A_CSRCPU->rows);
            cudaMemcpy(temp_g0, ising0, sizeof(int) * A_CSRCPU->rows, cudaMemcpyDeviceToHost);
            for(int i = 0; i < A_CSRCPU->rows; i++) {
                printf("%d ", temp_g0[i]);
            }
            printf("\n");
            free(temp_g0);
        #endif
        rowcolsum.toc();

        TicToc sicomputation("Si computation");
        sicomputation.tic();
        comptueSi<<<number_of_blocks, number_of_threads>>> (A_CSR, A_CSC, Si);
        // debugmuij<<<1,1>>> (A_CSR, Si);
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("Si computation\n");
            float * temp_si = (float * ) malloc(sizeof(float) * A_CSRCPU->rows);
            cudaMemcpy(temp_si, Si, sizeof(float) * A_CSRCPU->rows, cudaMemcpyDeviceToHost);
            for(int i = 0; i < A_CSRCPU->rows; i++) {
                printf("%f ", temp_si[i]);
            }
            printf("\n");
            free(temp_si);
        #endif
        sicomputation.toc();

        TicToc bfstime("BFS time...");
        bfstime.tic();
        int max_distance;
        int * bfs_distance = bfs_work_efficient(A_CSRCPU->rows, A_CSR, &max_distance);
        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("BFS\n");
            int * temp_bfs = (int * ) malloc(sizeof(int) * A_CSRCPU->rows);
            cudaMemcpy(temp_bfs, bfs_distance, sizeof(int) * A_CSRCPU->rows, cudaMemcpyDeviceToHost);
            for(int i = 0; i < A_CSRCPU->rows; i++) {
                printf("%d ", temp_bfs[i]);
            }
            printf("\n");
            free(temp_bfs);
        #endif
        cudaDeviceSynchronize();
        bfstime.toc();

        TicToc sortcomputation("Sort computation");
        sortcomputation.tic();
        sortNeighbourList<<<number_of_blocks, number_of_threads>>>
        (A_CSR, neighbour_list, Si, allowed, ktg, ising0);
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("\n");
            printf("PASS %d\n", pass);
            printf("neighbour_list\n");
            printCSRCPU(deepCopyMatrixCSRGPUtoCPU(neighbour_list));
        #endif
        sortcomputation.toc();

        TicToc aggregationtime("Aggregation time");
        aggregationtime.tic();

        initialize_array(A_CSRCPU->rows,  paired_with, -1);

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

        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("Paired with\n");
            int * temp_paired = (int * ) malloc(sizeof(int) * A_CSRCPU->rows);
            cudaMemcpy(temp_paired, paired_with, sizeof(int) * A_CSRCPU->rows, cudaMemcpyDeviceToHost);
            for(int i = 0; i < A_CSRCPU->rows; i++) {
                printf("%d ", temp_paired[i]);
            }
            printf("\n");
            free(temp_paired);
        #endif

        aggregationtime.toc();

        TicToc get_usefule_pairs_time("Get useful_pairs time");
        get_usefule_pairs_time.tic();
        get_useful_pairs<<<number_of_blocks, number_of_threads>>>
        (A_CSRCPU->rows, paired_with, useful_pairs);
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("Useful pairs\n");
            int * temp_useful= (int * ) malloc(sizeof(int) * A_CSRCPU->rows);
            cudaMemcpy(temp_useful, useful_pairs, sizeof(int) * A_CSRCPU->rows, cudaMemcpyDeviceToHost);
            for(int i = 0; i < A_CSRCPU->rows; i++) {
                printf("%d ", temp_useful[i]);
            }
            printf("\n");
            free(temp_useful);
        #endif
        get_usefule_pairs_time.toc();


        TicToc prefix_sum("Sum kernel");
        prefix_sum.tic();
        prefixSumGPU(useful_pairs, A_CSRCPU->rows);
        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("Prefix sum useful pairs\n");
            int * temp_sum = (int * ) malloc(sizeof(int) * A_CSRCPU->rows);
            cudaMemcpy(temp_sum, useful_pairs, sizeof(int) * A_CSRCPU->rows, cudaMemcpyDeviceToHost);
            for(int i = 0; i < A_CSRCPU->rows; i++) {
                printf("%d ", temp_sum[i]);
            }
            printf("\n");
            free(temp_sum);
        #endif
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
        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("Aggregations\n");
            int * temp_aggregations = (int * ) malloc(sizeof(int) * nc);
            cudaMemcpy(temp_aggregations, aggregations, sizeof(int) * nc, cudaMemcpyDeviceToHost);
            for(int i = 0; i < nc; i++) {
                printf("%d ", temp_aggregations[i]);
            }
            printf("\n");
            free(temp_aggregations);
        #endif
        cudaDeviceSynchronize();

        get_aggregations_count <<< (nc + 1024 - 1) / 1024, 1024 >>>
        (nc, aggregations, paired_with, aggregation_count);
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("Aggregations count\n");
            int * temp_aggregations_count = (int * ) malloc(sizeof(int) * nc);
            cudaMemcpy(temp_aggregations_count, aggregation_count, sizeof(int) * nc, cudaMemcpyDeviceToHost);
            for(int i = 0; i < nc; i++) {
                printf("%d ", temp_aggregations_count[i]);
            }
            printf("\n");
            free(temp_aggregations_count);
        #endif

        prefixSumGPU(aggregation_count, nc);

        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("Aggregations count sum\n");
            int * temp_aggregations_sum = (int * ) malloc(sizeof(int) * nc);
            cudaMemcpy(temp_aggregations_sum, aggregation_count, sizeof(int) * nc, cudaMemcpyDeviceToHost);
            for(int i = 0; i < nc; i++) {
                printf("%d ", temp_aggregations_sum[i]);
            }
            printf("\n");
            free(temp_aggregations_sum);
        #endif

        int nnz_in_p_matrix;
        assert(cudaMemcpy(&nnz_in_p_matrix, aggregation_count + nc - 1,
            sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
        

        MatrixCSR * P_transpose_shallow_cpu = 
            (MatrixCSR *) malloc(sizeof(MatrixCSR));

        P_transpose_shallow_cpu->rows = nc;
        P_transpose_shallow_cpu->cols = A_CSRCPU->rows;
        P_transpose_shallow_cpu->nnz = nnz_in_p_matrix;

        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("P transpose shallow CPU\n");
            printf("rows: %d, cols: %d, nnz %d\n", P_transpose_shallow_cpu->rows, P_transpose_shallow_cpu->cols, P_transpose_shallow_cpu->nnz);
        #endif

        assert(cudaMalloc(&P_transpose_shallow_cpu->i,
            sizeof(int) * (P_transpose_shallow_cpu->rows + 1)) == cudaSuccess);
        assert(cudaMalloc(&P_transpose_shallow_cpu->j,
            sizeof(int) * (P_transpose_shallow_cpu->nnz)) == cudaSuccess);
        assert(cudaMalloc(&P_transpose_shallow_cpu->val,
            sizeof(float) * (P_transpose_shallow_cpu->nnz)) == cudaSuccess);

        assign<<<1,1>>> (&P_transpose_shallow_cpu->i[0], 0);
        create_p_matrix_transpose <<< (nc + 1024 - 1) / 1024, 1024>>>
        (nc, aggregations, paired_with, aggregation_count,
            P_transpose_shallow_cpu->i, P_transpose_shallow_cpu->j,
            P_transpose_shallow_cpu->val);
        cudaDeviceSynchronize();

        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("P transpose row ptr\n");
            int sz = P_transpose_shallow_cpu->rows + 1;
            int * row_ptr = (int * ) malloc(sizeof(int) * sz);
            cudaMemcpy(row_ptr, P_transpose_shallow_cpu->i, sizeof(int) * sz, cudaMemcpyDeviceToHost);
            for(int i = 0; i < sz; i++) {
                printf("%d ", row_ptr[i]);
            }
            printf("\n");
            free(row_ptr);
        #endif

         #ifdef DEBUG
            {
                printf("PASS %d\n", pass);
                printf("P transpose j\n");
                int sz = P_transpose_shallow_cpu->nnz;
                int * temp = (int * ) malloc(sizeof(int) * sz);
                cudaMemcpy(temp, P_transpose_shallow_cpu->j, sizeof(int) * sz, cudaMemcpyDeviceToHost);
                for(int i = 0; i < sz; i++) {
                    printf("%d ", temp[i]);
                }
                printf("\n");
                free(temp);
            }
        #endif

        #ifdef DEBUG
            {
                printf("PASS %d\n", pass);
                printf("P transpose val\n");
                int sz = P_transpose_shallow_cpu->nnz;
                float * temp = (float * ) malloc(sizeof(float) * sz);
                cudaMemcpy(temp, P_transpose_shallow_cpu->val, sizeof(float) * sz, cudaMemcpyDeviceToHost);
                for(int i = 0; i < sz; i++) {
                    printf("%f ", temp[i]);
                }
                printf("\n");
                free(temp);
            }
        #endif

        MatrixCSR * P_transpose_gpu;
        assert(cudaMalloc(&P_transpose_gpu, sizeof(MatrixCSR)) == cudaSuccess);
        assert(cudaMemcpy(P_transpose_gpu, P_transpose_shallow_cpu, sizeof(MatrixCSR),
            cudaMemcpyHostToDevice) == cudaSuccess);

        #ifdef DEBUG
            printf("PASS %d\n", pass);
            printf("P_transpose\n");
            printCSRCPU(deepCopyMatrixCSRGPUtoCPU(P_transpose_gpu));
        #endif

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
