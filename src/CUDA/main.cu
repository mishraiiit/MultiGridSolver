#include "bits/stdc++.h"
#include "Matrix.cu"

using namespace std;

int main() {
	auto temp = readMatrix("../../matrices/poisson10000.mtx");
	for(int i = 0; i < temp->nnz; i++) {
		cout << temp->i[i] << " " << temp->j[i] << " " << temp->val[i] << endl;
	}
}
