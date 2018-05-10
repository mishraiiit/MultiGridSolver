#include "bits/stdc++.h"
using namespace std;

int main() {

	for(int i = 0; i < 4096; i++) {
		int u, v;
		cin >> u >> v;
		u--; v--;
		cout << u << " " << i + 1 << " " << 1 << endl;
		if(u != v) {
			cout << v << " " << i + 1 << " " << 1 << endl;
		}
	}

	return 0;
}