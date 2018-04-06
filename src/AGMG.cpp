#include <iostream>
#include <typeinfo>
#include <Eigen/Sparse>
#include <bits/stdc++.h>
#include <deque>
using namespace std;
using namespace Eigen;

typedef SparseMatrix<double, RowMajor> SMatrix;


namespace AGMG {

    vector<int> getNeighbours(int u, const SMatrix & adj) {
        vector<int> result;
        SparseVector<double> rowvec = adj.row(u);
        for(SparseVector<double>::InnerIterator i(rowvec); i; ++i) {
            if(i.value() != 0) {
                result.push_back(i.index());
            }
        }
        return result;
    }

    vector<int> getCMKOrdering(int n, const SMatrix & adj) {
        vector<int> order;
        vector<bool> visited(n, false);
        deque<int> q;
        int start = -1;

        vector<int> mini;

        for(int i = 0; i < n; i++) {
            if(start == -1 || adj.row(i).nonZeros() < adj.row(start).nonZeros()) {
                start = i;
            }
        }

        visited[start] = true;
        q.push_back(start);
        order.push_back(start);

        while(!q.empty()) {
            int u = q.front();
            q.pop_front();
            vector<int> ne = getNeighbours(u, adj);
            for(int v : ne) {
                if(!visited[v]) {
                    visited[v] = true;
                    q.push_back(v);
                    order.push_back(v);
                }
            }
        }

        assert(order.size() == n);
        return order;
    }

    std::vector<int> merge_sets(std::vector<int> arg1, std::vector<int> arg2) {
        std::vector<int> result(arg1.size() + arg2.size());
        std::merge(arg1.begin(), arg1.end(), arg2.begin(), arg2.end(), result.begin());
        return result;
    }

    SMatrix get_prolongation_matrix(SMatrix & A,
        std::vector<std::vector<int> > & g_vec) {
        int n = A.rows();
        SMatrix S(n, g_vec.size());
        vector<int> group_id(n, -1);
        for(int j = 0; j < g_vec.size(); j++) {
            for(int i : g_vec[j]) {
                group_id[i] = j;
            }
        }
        for(int i = 0; i < n; i++) {
            if(group_id[i] != -1)
                S.insert(i, group_id[i]) = 1;
        }
        return S;
    }

    SMatrix compress_matrix(SMatrix A,
        std::vector<std::vector<int> > g_vec) {
        SMatrix P = get_prolongation_matrix(A, g_vec);
        return P.transpose() * A * P;
    }

    double abs_row_col_sum(const SMatrix & A, const SMatrix & A_trans, int i) {
        SparseVector<double> row = A.row(i);
        SparseVector<double> col = A_trans.row(i);

        SparseVector<double>::InnerIterator row_i(row);
        SparseVector<double>::InnerIterator col_i(col);

        double ans = 0.0;
        while((row_i) && (col_i)) {
            if(row_i.index() == col_i.index()) {
                ans = ans + abs((row_i.value() + col_i.value()) / 2);
                ++row_i;
                ++col_i;
            } else if(row_i.index() < col_i.index()) {
                ans = ans + abs((row_i.value()) / 2);
                ++row_i;
            } else {
                ans = ans + abs((col_i.value()) / 2);
                ++col_i;
            }
        }

        while(row_i) {
            ans = ans + abs((row_i.value()) / 2);
            ++row_i;
        }

        while(col_i) {
            ans = ans + abs((col_i.value()) / 2);
            ++col_i;
        }

        ans = ans - abs(A.coeff(i, i));

        // double other = 0.0;
        // for(int j = 0; j < A.rows(); j++) {
        //     if(i == j) continue;
        //     other += abs((A.coeff(i, j) + A.coeff(j, i)) / 2.0);
        // }

        // assert(ans == other);
        return ans;
    }

    double row_col_sum(const SMatrix & A, const SMatrix & A_trans, int i) {
        SparseVector<double> row = A.row(i);
        SparseVector<double> col = A_trans.row(i);

        // double other = 0.0;
        // for(int j = 0; j < A.rows(); j++) {
        //     if(i == j) continue;
        //     other += A.coeff(i, j) + A.coeff(j, i);
        // }
        // other = -other / 2;

        double ans = - (row.sum() + col.sum()) / 2;
        ans += A.coeff(i, i);
        // assert(other == ans);
        return ans;
    }

    std::pair<int, std::pair<std::vector<int>, std::vector<std::vector<int> > > >
    initial_pairwise_aggregation
    (int n, const SMatrix & A, double ktg) {

        vector<int> cmk = getCMKOrdering(n, A);

        SMatrix A_trans = A.transpose();

        vector<vector<int> > reverse_adj(n);
        for(int i = 0; i < n; i++) {
            reverse_adj[i] = getNeighbours(i, A_trans);
        }

        vector<vector<int> > cmk_adj(n);
        for(int i : cmk) {
            for(int j : reverse_adj[i]) {
                cmk_adj[j].push_back(i);
            }
        }


        std::vector<std::vector<int> > g_vec;

        assert(n == A.rows());

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
            if(A.coeff(i, i) >= (ktg / (ktg - 2)) * abs_row_col_sum(A, A_trans, i)) {
                g0.push_back(i);
                // std::cerr << "g0 contains " << i << std::endl;
                in_u[i] = false;
            }
        }

        // std::cerr << "Size of g0 " << g0.size() << std::endl;

        /* 
            Initialized nc to 0.
        */
        int nc = 0;

        /*
            Compute si vector.
        */
        // DOUBT
        std::vector<double> s(n);
        for(int i = 0; i < n; i++) {
            s[i] = row_col_sum(A, A_trans, i);
        }

        /* Iteration part of this routine now. */

        for(int i_index = 0; i_index < n; i_index++) {
            int i = cmk[i_index];
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
                double num = 2 / (1 / A.coeff(i, i) + 1 / A.coeff(j, j));   
                double den = (- (A.coeff(i, j) + A.coeff(j, i)) / 2) + 1 / (1 / \
                (A.coeff(i, i) - si) + 1 / (A.coeff(j, j) - sj));
                return num / den;
            };

            for(int j : cmk_adj[i]) {
                if(!in_u[j]) continue;
                if((j != i) && (A.coeff(i, j) != 0)) {
                    assert(i < j);
                    double si = s[i];
                    double sj = s[j];
                    if(A.coeff(i, i) - si + A.coeff(j, j) - sj >= 0) {
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
                assert(A.coeff(i, best_j) != 0);
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

     std::pair<int, std::vector<std::vector<int> > >
    further_pairwise_aggregation 
    (int n, const SMatrix & A, double ktg, int nc_bar,
      std::vector<std::vector<int> > gk_bar, const SMatrix & A_bar) {
        std::vector<std::vector<int> > g_vec;
        assert(gk_bar.size() == nc_bar);

        vector<int> cmk = getCMKOrdering(nc_bar, A_bar);

        SMatrix A_bar_trans = A_bar.transpose();

        vector<vector<int> > reverse_adj(nc_bar);
        for(int i = 0; i < nc_bar; i++) {
            reverse_adj[i] = getNeighbours(i, A_bar_trans);
        }

        vector<vector<int> > cmk_adj(nc_bar);
        for(int i : cmk) {
            for(int j : reverse_adj[i]) {
                cmk_adj[j].push_back(i);
            }
        }

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
            si_bar[i] = row_col_sum(A_bar, A_bar_trans, i);
        }

        for(int i_index = 0; i_index < nc_bar; i_index++) {
            int i = cmk[i_index];
            if(!in_u[i]) continue;

            int best_j = -1;
            double best_mu_ij;

            auto mu_barij = [&A_bar, &si_bar](int i, int j) {
                double num = 2 / ((1 / A_bar.coeff(i, i)) + (1 / A_bar.coeff(j, j)));
                double den = (-((A_bar.coeff(i, j) + A_bar.coeff(j, i)) / 2)) + 1 / ((1 /\
                 (A_bar.coeff(i, i) - si_bar[i]))  + (1 / (A_bar.coeff(j, j) - si_bar[j])));
                return num / den;
            };

            for(int j : cmk_adj[i]) {
                if(!in_u[j]) continue;
                if((j != i) && (A_bar.coeff(i, j) != 0)) {
                    double si = si_bar[i];
                    double sj = si_bar[j];
                    if(A_bar.coeff(i, i) - si + A_bar.coeff(j, j) - sj >= 0) {
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

            if((best_j != -1) && (0 <= best_mu_ij && best_mu_ij <= ktg)) {
                g_vec.push_back(merge_sets(gk_bar[i], gk_bar[best_j]));
                in_u[i] = false;
                in_u[best_j] = false;
                assert(A_bar.coeff(i, best_j) != 0);
            } else {
                g_vec.push_back(gk_bar[i]);
                in_u[i] = false;
            }

        }
        
        assert(nc == g_vec.size());
        delete [] in_u;
        return {nc, g_vec};
    }

    std::pair<std::pair<int, std::vector<std::vector<int> > >, SMatrix> 
    multiple_pairwise_aggregation 
    (int n, const SMatrix & A, double ktg, int npass , double tou) {
        std::pair<int, std::pair<std::vector<int>, std::vector<std::vector<int> > > > 
        first_result = initial_pairwise_aggregation(n, A, ktg);
        
        std::pair<int, std::vector<std::vector<int> > > last_result = 
            {first_result.first, first_result.second.second};

        SMatrix last_A = compress_matrix(A, last_result.second);
        std::cerr << "Round 1 completed. Size: " << last_result.first << std::endl;
        int non_zero_in_A = A.nonZeros();

        for(int s = 2; s <= npass; s++) {
             last_result = further_pairwise_aggregation(
                n, A, ktg, last_result.first, last_result.second, last_A);
            last_A = compress_matrix(A, last_result.second);
            std::cerr << "Round " << s << " completed. Size: " << last_result.first << std::endl;
            if(last_A.nonZeros() <= (non_zero_in_A / tou)) break;
        }
        return { { last_result.first, last_result.second}, last_A};
    }

}