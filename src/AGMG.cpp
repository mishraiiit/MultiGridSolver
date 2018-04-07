#include <iostream>
#include <typeinfo>
#include <Eigen/Sparse>
#include <bits/stdc++.h>
#include <deque>
using namespace std;
using namespace Eigen;

typedef SparseMatrix<double, RowMajor> SMatrix;

struct vectorint {
    int * data;
    int size;
};

struct vectorvectorint {
    vectorint * data;
    int size;
};

namespace AGMG {

    int * getCMKOrdering(int n, const SMatrix & adj) {
        int * order = new int[n];
        bool * visited = new bool[n];
        for(int i = 0; i < n; i++) {
            visited[i] = 0;
        }

        int start = 0;
        visited[start] = true;
        int added = 0;
        int used = 0;

        order[added++] = start;

        while(used != added) {
            int u = order[used];
            used++;

            SparseVector<double> rowvec = adj.row(u);
            for(SparseVector<double>::InnerIterator i(rowvec); i; ++i) {
                int v = i.index();
                if(!visited[v]) {
                    visited[v] = true;
                    order[added++] = v;
                }
            }
        }

        assert(used == n);
        assert(added == n);
        delete [] visited;
        return order;
    }

    vectorint merge_sets(vectorint arg1, vectorint arg2) {
        vectorint result;
        result.size = arg1.size + arg2.size;
        result.data = new int[result.size];
        std::merge(arg1.data, arg1.data + arg1.size, arg2.data, arg2.data + arg2.size, result.data);
        return result;
    }

    SMatrix get_prolongation_matrix(const SMatrix & A,
        const std::vector<vectorint> & g_vec) {
        int n = A.rows();
        SMatrix S(n, g_vec.size());
        int * group_id = new int[n];
        fill(group_id, group_id + n, -1);
        for(int j = 0; j < g_vec.size(); j++) {
            for(int id = 0; id < g_vec[j].size; id++) {
                int i = g_vec[j].data[id];
                group_id[i] = j;
            }
        }
        for(int i = 0; i < n; i++) {
            if(group_id[i] != -1)
                S.insert(i, group_id[i]) = 1;
        }
        delete [] group_id;
        return S;
    }

    SMatrix compress_matrix(const SMatrix & A,
        const std::vector<vectorint> & g_vec) {
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

    std::pair<int, std::vector<vectorint> > 
    initial_pairwise_aggregation
    (int n, const SMatrix & A, double ktg) {
        int * cmk = getCMKOrdering(n, A);

        SMatrix A_trans = A.transpose();

        std::vector<vectorint> g_vec;

        assert(n == A.rows());

        bool * in_u = new bool[n];
        for(int i = 0; i < n; i++) {
            in_u[i] = true;
        }

        /* Initialization part of this routine.  */

        /*
            Compute set G.
        */

        for(int i = 0; i < n; i++) {
            if(A.coeff(i, i) >= (ktg / (ktg - 2)) * abs_row_col_sum(A, A_trans, i)) {
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
        double * s = new double[n];
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

            SparseVector<double> ne_row = A.row(i);
            for(SparseVector<double>::InnerIterator j_it(ne_row); j_it; ++j_it) {
                int j = j_it.index();
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
                vectorint aggregate;
                aggregate.size = 2;
                aggregate.data = new int[2];
                aggregate.data[0] = i;
                aggregate.data[1] = best_j;
                assert(A.coeff(i, best_j) != 0);
                in_u[i] = false;
                in_u[best_j] = false;
                g_vec.push_back(aggregate);
            } else {
                vectorint aggregate;
                aggregate.size = 1;
                aggregate.data = new int[1];
                aggregate.data[0] = i;
                in_u[i] = false;
                g_vec.push_back(aggregate);
            }
        }
        assert(nc == g_vec.size());
        delete [] in_u;
        delete [] cmk;
        delete [] s;
        return {nc, g_vec};
    }

     std::pair<int, std::vector<vectorint > >
    further_pairwise_aggregation 
    (int n, const SMatrix & A, double ktg, int nc_bar,
      std::vector<vectorint> gk_bar, const SMatrix & A_bar) {
        std::vector<vectorint> g_vec;
        assert(gk_bar.size() == nc_bar);

        SMatrix A_bar_trans = A_bar.transpose();

        bool * in_u = new bool[nc_bar];

        /* Initialization part of this routine.  */
        for(int i = 0; i < nc_bar; i++) {
            in_u[i] = true;
        }
        int nc = 0;

        // i varies from 0 to nc_bar - 1. (in the paper it varies from 1 to 
        // nc_bar, but we follow 0 based indexing. )

        double * si_bar = new double[nc_bar];
        for(int i = 0; i < nc_bar; i++) {
            si_bar[i] = row_col_sum(A_bar, A_bar_trans, i);
        }

        for(int i = 0; i < nc_bar; i++) {
            if(!in_u[i]) continue;

            int best_j = -1;
            double best_mu_ij;

            auto mu_barij = [&A_bar, &si_bar](int i, int j) {
                double num = 2 / ((1 / A_bar.coeff(i, i)) + (1 / A_bar.coeff(j, j)));
                double den = (-((A_bar.coeff(i, j) + A_bar.coeff(j, i)) / 2)) + 1 / ((1 /\
                 (A_bar.coeff(i, i) - si_bar[i]))  + (1 / (A_bar.coeff(j, j) - si_bar[j])));
                return num / den;
            };

            SparseVector<double> ne_row = A_bar.row(i);
            for(SparseVector<double>::InnerIterator j_it(ne_row); j_it; ++j_it) {
                int j = j_it.index();
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
        delete [] si_bar;
        return {nc, g_vec};
    }

    std::pair<std::pair<int, std::vector<vectorint> >, SMatrix> 
    multiple_pairwise_aggregation 
    (int n, const SMatrix & A, double ktg, int npass , double tou) {
        std::pair<int, std::vector<vectorint> > 
        first_result = initial_pairwise_aggregation(n, A, ktg);
        
        std::pair<int, std::vector<vectorint> > last_result = 
            {first_result.first, first_result.second};

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