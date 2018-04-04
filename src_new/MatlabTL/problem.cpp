#include "bits/stdc++.h"
using namespace std;

vector<double> A[3];

int main() {
	int n = 148800;
	n = n / 3;
	for(int i = 0; i < n; i++) {
		double x, y, val;
		cin >> x >> y >> val;
		A[0].push_back(x);
		A[1].push_back(y);
		A[2].push_back(val);
	}
	cout << 10000 << " " << 10000 << " " << n << endl;
	for(int i = 0; i < n; i++) {
		cout << A[0][i] << " " << A[1][i] << " " << A[2][i] << endl;
	}
	return 0;
}
