#include "bits/stdc++.h"
using namespace std;

int n;
vector<pair<pair<int, int>, double> > A;

int main() {

	scanf("%d", &n);
	printf("%MatrixMarket matrix coordinate real general\n");
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			int elem = i * n + j + 1;
			
			if(i > 0) {
				A.push_back(make_pair(make_pair(elem, elem - n), -1));
			}
		
			if(j > 0) {
				A.push_back(make_pair(make_pair(elem, elem - 1), -1));
			}

			A.push_back(make_pair(make_pair(elem, elem), 4));

			if(j < n - 1) {
				A.push_back(make_pair(make_pair(elem, elem + 1), -1));
			}

			if(i < n - 1) {
				A.push_back(make_pair(make_pair(elem, elem + n), -1));
			}
		}
	}

	printf("%d %d %d\n", n * n, n * n, (int) A.size());
	for(int i = 0; i < A.size(); i++) {
		printf("%d %d %d\n", A[i].first.first, A[i].first.second, (int)A[i].second);
	}

	return 0;
}
