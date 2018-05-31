#include <iostream>
#include <typeinfo>
#include <Eigen/Sparse>
#include <bits/stdc++.h>
#include <deque>
using namespace std;
using namespace Eigen;

typedef SparseMatrix<double, RowMajor> SMatrix;

namespace AGMG {

    const int * const getCMKOrdering(const int n, const SMatrix & adj) {
        int * const order = new int[n];
        bool * const visited = new bool[n];
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

            const SparseVector<double> rowvec = adj.row(u);
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

    inline const double abs_row_col_sum(const SMatrix & A, const SMatrix & A_trans, const int i) {
        const SparseVector<double> row = A.row(i);
        const SparseVector<double> col = A_trans.row(i);

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
        return ans;
    }

    inline const double row_col_sum(const SMatrix & A, const SMatrix & A_trans, const int i) {
        const SparseVector<double> row = A.row(i);
        const SparseVector<double> col = A_trans.row(i);
        double ans = - (row.sum() + col.sum()) / 2;
        ans += A.coeff(i, i);
        return ans;
    }

    inline const double mu(const SMatrix & A, const double * const s, const int i, const int j) {
        const double si = s[i];
        const double sj = s[j];
        const double num = 2 / (1 / A.coeff(i, i) + 1 / A.coeff(j, j));   
        const double den = (- (A.coeff(i, j) + A.coeff(j, i)) / 2) + 1 / (1 / \
        (A.coeff(i, i) - si) + 1 / (A.coeff(j, j) - sj));
        return num / den;
    }

    const SMatrix initial_pairwise_aggregation
    (const SMatrix & A, const double ktg) {
        const int n = A.rows();
        const int * const cmk = getCMKOrdering(n, A);

        SMatrix A_trans = A.transpose();

        int * const groups = new int[n];

        assert(n == A.rows());

        bool * const in_u = new bool[n];
        for(int i = 0; i < n; i++) {
            in_u[i] = true;
        }


        for(int i = 0; i < n; i++) {
            if(A.coeff(i, i) >= (ktg / (ktg - 2)) * abs_row_col_sum(A, A_trans, i)) {
                // std::cerr << "g0 contains " << i << std::endl;
                in_u[i] = false;
            }
        }

        int nc = 0;

        /*
            Compute si vector.
        */
        double * const s = new double[n];
        for(int i = 0; i < n; i++) {
            groups[i] = -1;
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


            const SparseVector<double> ne_row = A.row(i);
            for(SparseVector<double>::InnerIterator j_it(ne_row); j_it; ++j_it) {
                int j = j_it.index();
                if(!in_u[j]) continue;
                if((j != i) && (A.coeff(i, j) != 0)) {
                    assert(i < j);
                    double si = s[i];
                    double sj = s[j];
                    if(A.coeff(i, i) - si + A.coeff(j, j) - sj >= 0) {
                        // Finding the best j.
                        double current_mu_ij = mu(A, s, i, j);
                        if((best_j == -1 && current_mu_ij > 0) || (current_mu_ij > 0 && current_mu_ij < best_mu_ij)) {
                            best_j = j;
                            best_mu_ij = current_mu_ij;
                        }
                    }
                }
            }
            nc = nc + 1;
            if((best_j != -1) && (best_mu_ij <= ktg)) {
                groups[i] = nc - 1;
                groups[best_j] = nc - 1;
                in_u[i] = false;
                in_u[best_j] = false;
            } else {
                groups[i] = nc - 1;
                in_u[i] = false;
            }
        }

        SMatrix P(n, nc);
        for(int i = 0; i < n; i++) {
            if(groups[i] != -1) {
                P.insert(i, groups[i]) = 1;
            }
        }

        delete [] in_u;
        delete [] cmk;
        delete [] s;
        delete [] groups;

        return P;
    }

    const SMatrix further_pairwise_aggregation 
    (const SMatrix & A, const double ktg, const SMatrix & P_bar, const SMatrix & P_bar_trans, const SMatrix & A_bar) {
        const int n = A.rows();
        const int nc_bar = P_bar.cols();
        const SMatrix A_bar_trans = A_bar.transpose();

        int * const groups = new int[n];
        bool * const in_u = new bool[nc_bar];

        for(int i = 0; i < n; i++) {
            groups[i] = -1;
        }

        /* Initialization part of this routine.  */
        for(int i = 0; i < nc_bar; i++) {
            in_u[i] = true;
        }
        int nc = 0;

        double * const si_bar = new double[nc_bar];
        for(int i = 0; i < nc_bar; i++) {
            si_bar[i] = row_col_sum(A_bar, A_bar_trans, i);
        }

        for(int i = 0; i < nc_bar; i++) {
            if(!in_u[i]) continue;

            int best_j = -1;
            double best_mu_ij;

            const SparseVector<double> ne_row = A_bar.row(i);
            for(SparseVector<double>::InnerIterator j_it(ne_row); j_it; ++j_it) {
                int j = j_it.index();
                if(!in_u[j]) continue;
                if((j != i) && (A_bar.coeff(i, j) != 0)) {
                    double si = si_bar[i];
                    double sj = si_bar[j];
                    if(A_bar.coeff(i, i) - si + A_bar.coeff(j, j) - sj >= 0) {
                        // Finding the best j.
                        double current_mu_ij = mu(A_bar, si_bar, i, j);
                        if((best_j == -1 && current_mu_ij > 0) || (current_mu_ij > 0 && current_mu_ij < best_mu_ij)) {
                            best_j = j;
                            best_mu_ij = current_mu_ij;
                        }
                    }
                }
            }

            nc = nc + 1;

            if((best_j != -1) && (best_mu_ij <= ktg)) {
                SparseVector<double> row_i = P_bar_trans.row(i);
                SparseVector<double> row_j = P_bar_trans.row(best_j);

                for(SparseVector<double>::InnerIterator row_i_it(row_i); row_i_it; ++row_i_it) {
                    groups[row_i_it.index()] = nc - 1;
                }

                for(SparseVector<double>::InnerIterator row_j_it(row_j); row_j_it; ++row_j_it) {
                    groups[row_j_it.index()] = nc - 1;
                }
                in_u[i] = false;
                in_u[best_j] = false;
            } else {
                const SparseVector<double> row_i = P_bar_trans.row(i);
                for(SparseVector<double>::InnerIterator row_i_it(row_i); row_i_it; ++row_i_it) {
                    groups[row_i_it.index()] = nc - 1;
                }
                in_u[i] = false;
            }

        }
        
        SMatrix P(n, nc);
        for(int i = 0; i < n; i++) {
            if(groups[i] != -1) {
                P.insert(i, groups[i]) = 1;
            }
        }
        
        delete [] in_u;
        delete [] si_bar;
        delete [] groups;
        return P;
    }

    const SMatrix multiple_pairwise_aggregation 
    (const SMatrix & A, double ktg, int npass , double tou) {
        const int n = A.rows();   
        const int non_zero_in_A = A.nonZeros();
        SMatrix P_bar = initial_pairwise_aggregation(A, ktg);
        std::cerr << "Round 1 completed. Size: " << P_bar.cols() << std::endl;

        for(int s = 2; s <= npass; s++) {
            const SMatrix & P_bar_trans = P_bar.transpose();
            const SMatrix & A_bar = P_bar_trans * A * P_bar;
            if(A_bar.nonZeros() <= non_zero_in_A / tou) break;
            P_bar = further_pairwise_aggregation(A, ktg, P_bar, P_bar_trans, A_bar);
            std::cerr << "Round " << s << " completed. Size: " << P_bar.cols() << std::endl;
        }
        return P_bar;
    }

}