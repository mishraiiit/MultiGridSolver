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

  
  std::string inp_file("../matrices/poisson10000.txt");
  SparseMatrix T(inp_file);
  
  assert(T.row_size() == 10000);
  assert(T.col_size() == 10000);

  auto result = AGMG::multiple_pairwise_aggregation(T.row_size(), T, 8, 6, 1000000);
  cout << T.row_size() << " " << result.first.first << endl;
  auto pro_matrix = AGMG::get_prolongation_matrix(T, result.first.second);
  for(int  i = 0; i < pro_matrix.row_size(); i++) {
    auto & p = pro_matrix[i].getData();
    for(auto g : p) {
      cout << i+1 << " " << g.first + 1 << " " << g.second << endl;
    }
  }
  return 0;
}