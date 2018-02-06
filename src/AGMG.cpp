#include "SparseMatrix.h"
#include <assert.h>
#include <set>

namespace AGMG {
	std::pair<int, std::vector<std::set<int> > > initial_pairwise_aggregation \
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
		g_vec.push_back(g0);

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
			auto mu = [&A] (int i, int j) {
				double si = - A.getRowColSum(i);
				double sj = - A.getRowColSum(j);
				double num = 2 / (1 / A[i][i] + 1 / A[j][j]);	
				double den = (- (A[i][j] + A[j][i]) / 2) + 1 / (1 / (A[i][i] - si) + 1 / (A[j][j] - sj));
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
		assert(nc == g_vec.size() + 1);
		return {nc, g_vec};
	}
};