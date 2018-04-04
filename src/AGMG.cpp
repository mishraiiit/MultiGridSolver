#include "SparseMatrix.h"
#include "Utility.h"
#include <assert.h>
#include <vector>

// This implements the algorithm describe in 
// "AGGREGATION-BASED ALGEBRAIC MULTI-GRID FOR CONVECTION DIFFUSION EQUATIONS".
// By Yvan Notay

namespace AGMG {


    SparseMatrix get_prolongation_matrix(SparseMatrix & A,
        std::vector<std::vector<int> > & g_vec) {
        SparseMatrix P(std::vector<SparseVector> (A.row_size(), 
            SparseVector(g_vec.size(), {})), A.row_size(), g_vec.size());

        for(int j = 0; j < g_vec.size(); j++) {
            for(int i : g_vec[j]) {
                assert(i < A.row_size());
                assert(j < g_vec.size());
                P[i].size = g_vec.size();
                P[i].getData().push_back({j, 1});
            }
        }
        return P;
    }

    /*
        compress_matrix
        Input:
            SparseMatrix A
            It's the matrix we want to compress.

            std::vector<std::vector<int> > g_vec
            It contains the sets G1 to Gnc.
            It's required to create the matrices P transpose (restriction matrix)
            and P (prolongation matrix), which is required to return
            P(transpose) * A * P
        Output:
            SparseMatrix result
            It's the resultant matrix returned via P(transpose) * A * P
    */

    SparseMatrix compress_matrix(SparseMatrix A,
        std::vector<std::vector<int> > g_vec) {

        SparseMatrix P = get_prolongation_matrix(A, g_vec);
        P.changed();
        return P.transpose() * A * P;
    }

    /*
        initial_pairwise_aggregation
        Input:
            int n
            The dimension of the matrix.

            SparseMatrix A
            The n x n matrix A.

            double ktg
            Some tuning parameter given in the research paper.
        Output:
            int result.first 
            nc. Size of the compressed coarse matrix.

            std::vector<int> result.second.first
            g0. It contains the set g0. Refer to paper for more details.

            std::vector<std::vector<int> result.second.second
            g_vec. It contains the set g1 to gnc.
            Note that I am returning g0 seperately.
    */

    std::pair<int, std::pair<std::vector<int>, std::vector<std::vector<int> > > >
    initial_pairwise_aggregation
    (int n, SparseMatrix & A, double ktg) {
        std::vector<std::vector<int> > g_vec;

        assert(n == A.row_size());

        bool * in_u = new bool[n];
        for(int i = 0; i < n; i++) {
            in_u[i] = true;
        }

        /* Initialization part of this routine.  */

        /*
            Compute set G.
        */
        std::vector<int> g0;
        for(int i = 0; i < n; i++) {
            if(A[i][i] >= (ktg / (ktg - 2)) * A.getRowColAbsSum(i)) {
                g0.push_back(i);
                in_u[i] = false;
            }
        }


        /* 
            Initialized nc to 0.
        */
        int nc = 0;

        /*
            Compute si vector.
        */
        std::vector<double> s(n);
        for(int i = 0; i < n; i++) {
            s[i] = - A.getRowColSum(i);
        }

        /* Iteration part of this routine now. */

        for(int i = 0; i < n; i++) {
            if(!in_u[i]) continue;
            /* The value of j for which mu ({i, j}) is minimized.
               best_mu_ij would store the minimum mu ({i, j}) value.
               best_j would store the j for which it is minimum.
               Initially best_j is -1. If in the end also it's -1
               then it would mean that there's no suitable j.
            */
            int best_j = -1;
            double best_mu_ij;

            // mu({i, j})
            // Lambda function takes inp {i, j} and returns mu({i, j}).
            auto mu = [&A, &s] (int i, int j) {
                double si = s[i];
                double sj = s[j];
                double num = 2 / (1 / A[i][i] + 1 / A[j][j]);   
                double den = (- (A[i][j] + A[j][i]) / 2) + 1 / (1 / \
                (A[i][i] - si) + 1 / (A[j][j] - sj));
                return num / den;
            };

            for(int j = i + 1; j < n; j++) {
                if(!in_u[j]) continue;
                if((j != i) && (A[i][j] != 0)) {
                    assert(i < j);
                    double si = s[i];
                    double sj = s[j];
                    if(A[i][i] - si + A[j][j] - sj >= 0) {
                        // Finding the best j.
                        double current_mu_ij = mu(i, j);
                        if((best_j == -1) || (current_mu_ij < best_mu_ij)) {
                            best_j = j;
                            best_mu_ij = current_mu_ij;
                        }
                    }
                }
            }
            nc = nc + 1;
            if((best_j != -1) && (best_mu_ij <= ktg)) {
                g_vec.push_back({i, best_j});
                in_u[i] = false;
                in_u[best_j] = false;
            } else {
                g_vec.push_back({i});
                in_u[i] = false;
            }
        }
        assert(nc == g_vec.size());
        delete [] in_u;
        return {nc, {g0, g_vec}};
    }

    /*
        further_pairwise_aggregation
        Input:
            int n
            The dimension of the square SparseMatrix which is the next input.

            SparseMatrix A
            The input n x n matrix A.

            double ktg
            Tuning parameter ktg, refer to the paper for more details.

            int nc_bar
            The tentative coarse matrix size.

            std::vector<std::vector<int> > gk_bar
            The tentative groupings g1 to g_nc_bar

            SparseMatrix A_bar
            The tentative coarse grid matrix.
        Output:
            int result.first
            nc. Size of the coarse grid matrix.

            std::vector<std::vector<int> > result.second
            g_vec. Groups g_1 to g_nc.
    */

    std::pair<int, std::vector<std::vector<int> > >
    further_pairwise_aggregation 
    (int n, SparseMatrix & A, double ktg, int nc_bar,
      std::vector<std::vector<int> > gk_bar, SparseMatrix & A_bar) {
        std::vector<std::vector<int> > g_vec;
        assert(gk_bar.size() == nc_bar);

        bool * in_u = new bool[nc_bar];

        /* Initialization part of this routine.  */
        for(int i = 0; i < nc_bar; i++) {
            in_u[i] = true;
        }
        int nc = 0;

        // i varies from 0 to nc_bar - 1. (in the paper it varies from 1 to 
        // nc_bar, but we follow 0 based indexing. )

        std::vector<double> si_bar(nc_bar);
        for(int i = 0; i < nc_bar; i++) {
            si_bar[i] = - A_bar.getRowColSum(i);
        }

        for(int i = 0; i < nc_bar; i++) {

            if(!in_u[i]) continue;

            int best_j = -1;
            double best_mu_ij;

            auto mu_barij = [&A_bar, &si_bar](int i, int j) {
                double num = 2 / ((1 / A_bar[i][i]) + (1 / A_bar[j][j]));
                double den = (-((A_bar[i][j] + A_bar[j][i]) / 2)) + 1 / ((1 /\
                 (A_bar[i][i] - si_bar[i]))  + (1 / (A_bar[j][j] - si_bar[j])));
                return num / den;
            };

            for(int j = i + 1; j < nc_bar; j++) {
                if(!in_u[j]) continue;
                if((j != i) && (A_bar[i][j] != 0)) {
                    double si = si_bar[i];
                    double sj = si_bar[j];
                    if(A_bar[i][i] - si + A_bar[j][j] - sj >= 0) {
                        // Finding the best j.
                        double current_mu_ij = mu_barij(i, j);
                        if((best_j == -1) || (current_mu_ij < best_mu_ij)) {
                            best_j = j;
                            best_mu_ij = current_mu_ij;
                        }
                    }
                }
            }

            nc = nc + 1;

            if(0 <= best_mu_ij && best_mu_ij <= ktg) {
                g_vec.push_back(merge_sets(gk_bar[i], gk_bar[best_j]));
                in_u[i] = false;
                in_u[best_j] = false;
            } else {
                g_vec.push_back(gk_bar[i]);
                in_u[i] = false;
            }
        }
        
        assert(nc == g_vec.size());
        delete [] in_u;
        return {nc, g_vec};
    }

    /*
        multiple_pairwise_aggregation
        Input:
            int n
            Size of the input square sparse matrix A.

            SparseMatrix A
            The n x n input square sparse matrix A.

            double ktg
            Some turning parameter given in the paper. Refer to paper.

            int npass.
            The number of iterations we want to run on the matrix.
            The first pass applies the initial_pairwise_aggregation.
            The remaining npass - 1 passes applies the further pairwise
            aggregation on the output of the previous pass.

            double tou
            Some tuning paramter given in the paper. The coarsening factor.
            Refer to paper for more details.
        Output:
            int result.first.first
            nc. Size of the coarse grid matrix.

            std::vector<std::set<int> > result.first.second
            g_vec. Aggregates/Groups of the last iteration g_1 to g_nc.

            SparseMatrix> result.second
            Ac. The coarse grid matrix of size nc x nc.
    */

    std::pair<std::pair<int, std::vector<std::vector<int> > >, SparseMatrix> 
    multiple_pairwise_aggregation 
    (int n, SparseMatrix & A, double ktg, int npass , double tou) {
        std::pair<int, std::pair<std::vector<int>, std::vector<std::vector<int> > > > 
        first_result = initial_pairwise_aggregation(n, A, ktg);
        
        std::pair<int, std::vector<std::vector<int> > > last_result = 
            {first_result.first, first_result.second.second};

        SparseMatrix last_A = compress_matrix(A, last_result.second);
        std::cerr << "Round 1 completed. Size: " << last_result.first << std::endl;
        int non_zero_in_A = A.nnz();

        for(int s = 2; s <= npass; s++) {
             last_result = further_pairwise_aggregation(
                n, A, ktg, last_result.first, last_result.second, last_A);
            last_A = compress_matrix(A, last_result.second);
            std::cerr << "Round " << s << " completed. Size: " << last_result.first << std::endl;
            if(last_A.nnz() <= (non_zero_in_A / tou)) break;
        }
        return { { last_result.first, last_result.second}, last_A};
    }
};