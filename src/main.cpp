#include "bits/stdc++.h"
#include "AGMG.cpp"
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

int main() {

  SparseMatrix S = DenseMatrix({
    {1, 0, 0, 2},
    {0, 1, 0, 0},
    {2, 0, 1, 0},
    {0, 2, 0, 1},
  }).toSparseMatrix();

  auto result = AGMG::multiple_pairwise_aggregation(S.row_size(), S, 3, 0, 1);
  cout << result.first.first << endl;
  result.second.print();
  return 0;
}