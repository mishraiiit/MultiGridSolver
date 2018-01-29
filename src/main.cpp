#include "bits/stdc++.h"
#include "DenseMatrix.h"
#define ll long long int
using namespace std;
struct ${$(){
    ios_base::sync_with_stdio(0);cin.tie(0);
    cout << fixed << setprecision(9);
}}$;

#ifdef TRACE
  #define trace(...) __f(#__VA_ARGS__, __VA_ARGS__)
  template <typename Arg1>
  void __f(const char* name, Arg1&& arg1){
    cerr << name << " : " << arg1 << std::endl;
  }
  template <typename Arg1, typename... Args>
  void __f(const char* names, Arg1&& arg1, Args&&... args){
    const char* comma = strchr(names + 1, ',');cerr.write(names, comma - names) << " : " << arg1<<" | ";__f(comma+1, args...);
  }
#else
  #define trace(...)
#endif

string operator+(string a, int b) {
  return a + (char)(b + '0');
}

string operator+(int a, string b) {
  return (char)(a + '0') + b;
}

int main() {

  DenseMatrix D(5, 5);
  D[0][0] = 1;
  for(int i = 0; i < 5; i++) {
    D[i][0] = i + 5;
  }
  D[4][4] = 100;
  D = D * D * 2;
  for(int i = 0; i < 5; i++) {
    for(int j = 0; j < 5; j++) {
      cout << D[i][j] << " ";
    }
    cout << endl;
  }
}