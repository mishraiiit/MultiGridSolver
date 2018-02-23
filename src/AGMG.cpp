#include "SparseMatrix.h"
#include "Utility.h"
#include <assert.h>
#include <set>

// This implements the algorithm describe in 
// "AGGREGATION-BASED ALGEBRAIC MULTI-GRID FOR CONVECTION DIFFUSION EQUATIONS".
// By Yvan Notay

namespace AGMG {

    /*
        compress_matrix
        Input:
            SparseMatrix A
            It's the matrix we want to compress.

            std::vector<std::set<int> > g_vec
            It contains the sets G1 to Gnc.
            It's required to create the matrices P transpose (restriction matrix)
            and P (prolongation matrix), which is required to return
            P(transpose) * A * P
        Output:
            SparseMatrix result
            It's the resultant matrix returned via P(transpose) * A * P
    */

    SparseMatrix compress_matrix(SparseMatrix A,
        std::vector<std::set<int> > g_vec) {

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
        P.changed();
        SparseMatrix X = P.transpose();
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

            std::set<int> result.second.first
            g0. It contains the set g0. Refer to paper for more details.

            std::vector<std::set<int> result.second.second
            g_vec. It contains the set g1 to gnc.
            Note that I am returning g0 seperately.
    */

    std::pair<int, std::pair<std::set<int>, std::vector<std::set<int> > > >
    initial_pairwise_aggregation
    (int n, SparseMatrix & A, double ktg) {
        std::vector<std::set<int> > g_vec;

        assert(n == A.row_size());

        /* Initialization part of this routine.  */

        /*
            Compute set G.
        */
        std::set<int> g0;
        for(int i = 0; i < n; i++) {
            if(A[i][i] >= (ktg / (ktg - 2)) * A.getRowColAbsSum(i)) {
                g0.insert(i);
            }
        }

        /* 
            Compute set U.
        */
        std::set<int> u;
        for(int i = 0; i < n; i++) {
            if(!g0.count(i)) {
                u.insert(i);
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

        while(!u.empty()) {
            /* Selecting an i */
            int i = * u.begin();

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

            for(int j : u) {
                if((j != i) && (A[i][j] != 0)) {
                    double si = - A.getRowColSum(i);
                    double sj = - A.getRowColSum(j);
                    if(A[i][i] - si + A[j][j] - sj >= 0) {
                        // Finding the best j.
                        double current_mu_ij = mu(i, j);
                        if((best_j == -1) || (current_mu_ij < best_mu_ij)) {
                            best_j = j;
                            best_mu_ij = mu(i, j);
                        }
                    }
                }
            }
            nc = nc + 1;
            if((best_j != -1) && (best_mu_ij <= ktg)) {
                g_vec.push_back({i, best_j});
                u.erase(i);
                u.erase(best_j);
            } else {
                g_vec.push_back({i});
                u.erase(i);
            }
        }

        assert(nc == g_vec.size());
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

            std::vector<std::set<int> > gk_bar
            The tentative groupings g1 to g_nc_bar

            SparseMatrix A_bar
            The tentative coarse grid matrix.
        Output:
            int result.first
            nc. Size of the coarse grid matrix.

            std::vector<std::set<int> > result.second
            g_vec. Groups g_1 to g_nc.
    */

    std::pair<int, std::vector<std::set<int> > >
    further_pairwise_aggregation 
    (int n, SparseMatrix & A, double ktg, int nc_bar,
      std::vector<std::set<int> > gk_bar, SparseMatrix & A_bar) {
        std::vector<std::set<int> > g_vec;
        assert(gk_bar.size() == nc_bar);

        /* Initialization part of this routine.  */
        std::set<int> u;
        for(int i = 0; i < nc_bar; i++) {
            u.insert(i);
        }
        int nc = 0;

        // i varies from 0 to nc_bar - 1. (in the paper it varies from 1 to 
        // nc_bar, but we follow 0 based indexing. )

        std::vector<double> si_bar(nc_bar);
        for(int i = 0; i < nc_bar; i++) {
            for(int k : gk_bar[i]) {
                for(int j = 0; j < nc_bar; j++) {
                    if(gk_bar[i].count(j) == 0) {
                        // DOUBT
                        si_bar[i] += (A[k][j] + A[j][k]) / 2;
                    }
                }
            }
            si_bar[i] = - si_bar[i];
        }

        auto mu_barij = [&A_bar, &si_bar](int i, int j) {
            double num = 2 / ((1 / A_bar[i][i]) + (1 / A_bar[j][j]));
            double den = (-((A_bar[i][j] + A_bar[j][i]) / 2)) + 1 / ((1 /\
             (A_bar[i][i] - si_bar[i]))  + (1 / (A_bar[j][j] - si_bar[j])));
            return num / den;
        };

        while(!u.empty()) {
            int i = * u.begin();
            std::set<int> T;
            for(int j = 0; j < nc_bar; j++) {
                double mu_ij_val = mu_barij(i, j);
                if(A_bar[i][j] != 0 && ((A_bar[i][i] - si_bar[i] + A_bar[j][j] -\
                 si_bar[j]) >= 0) && (0 < mu_ij_val && mu_ij_val <= ktg)) {
                    T.insert(j);
                }
            }
            nc = nc + 1;
            if(!T.empty()) {
                int best_j = * T.begin();
                double best_mu_ij = mu_barij(i, best_j);
                for(int j : T) {
                    double curr_mu_ij = mu_barij(i, j);
                    if(curr_mu_ij < best_mu_ij) {
                        best_mu_ij = curr_mu_ij;
                        best_j = j;
                    }
                }
                std::set<int> f = gk_bar[i];
                std::set<int> s = gk_bar[best_j];
                g_vec.push_back(merge_sets(f, s));
                u.erase(i);
                u.erase(best_j);
            } else {
                g_vec.push_back(gk_bar[i]);
                u.erase(i);
            }
        }
        assert(nc == g_vec.size());
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

    std::pair<std::pair<int, std::vector<std::set<int> > >, SparseMatrix> 
    multiple_pairwise_aggregation 
    (int n, SparseMatrix & A, double ktg, int npass , double tou) {
        std::pair<int, std::pair<std::set<int>, std::vector<std::set<int> > > > 
        first_result = initial_pairwise_aggregation(n, A, ktg);
        
        std::pair<int, std::vector<std::set<int> > > last_result = 
            {first_result.first, first_result.second.second};

        SparseMatrix last_A = compress_matrix(A, last_result.second);

        int non_zero_in_A = A.nnz();

        for(int s = 2; s <= npass; s++) {
             last_result = further_pairwise_aggregation(
                n, A, ktg, last_result.first, last_result.second, last_A);
            last_A = compress_matrix(A, last_result.second);
            if(last_A.nnz() <= (non_zero_in_A / tou)) break;
        }
        return { { last_result.first, last_result.second}, last_A};
    }
};