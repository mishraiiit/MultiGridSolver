#include <iostream>
#include "MatrixIO.cpp"
#include <typeinfo>
#include <Eigen/Sparse>
#include <bits/stdc++.h>
#include <chrono>
#include <string>
using namespace std;
using namespace Eigen;

vector < vector < pair <int, int> > > cluster_contains;
vector < vector <int> > adj;
vector <int> colors;

bool adjacent(pair<int, int> a, pair<int, int> b) {
    return (abs(a.first - b.first) + abs(a.second - b.second) == 1);
}

bool connected(int u, int v) {
    auto & pointsu = cluster_contains[u];
    auto & pointsv = cluster_contains[v];
    for(auto pointu : pointsu) {
        for(auto pointv : pointsv) {
            if(adjacent(pointu, pointv))
                return true;
        }
    }
    return false;
}

void dfs(int start) {
    set<int> adjcolors;
    for(int i : adj[start]) {
        adjcolors.insert(colors[i]);
    }
    for(int i = 0; ;i++) {
        if(adjcolors.count(i) == 0) {
            colors[start] = i;
            break;
        }
    }
    for(int v : adj[start]) {
        if(colors[v] == -1) {
            dfs(v);
        }
    }
}

int main(int argc, char ** argv) {

  if(argc != 2) {
    printf("Invalid arguments.\n");
    printf("First argument should be the name of the prolongation matrix without the suffix promatrix.mtx.\n");
    exit(1);
  }

  string matrixname = argv[1];
  SMatrix T = readMatrix(string("../matrices/") + matrixname + string("promatrix.mtx"));

  int elements = T.rows();
  int clusters = T.cols();
  int root = sqrt(elements);

  cluster_contains.resize(clusters);
  colors = vector<int> (clusters, -1);
  adj.resize(clusters);

  if(root * root != elements) {
    printf("Not a valid prolongation matrix of a 2D matrix.\n");
    exit(1);
  }

  for(int i = 0; i < elements; i++) {
    SparseVector<double> row = T.row(i);
    SparseVector<double>::InnerIterator it(row);
    if(it) {
        int cluster = it.index();
        cluster_contains[cluster].push_back({i/root, i %root});
    }
  }

  for(int i = 0; i < clusters; i++) {
    for(int j = 0; j < clusters; j++) {
        if(connected(i, j)) {
            adj[i].push_back(j);
            adj[j].push_back(j);
        }
    }
  }

  dfs(0);
  vector < pair < pair <int, int>, int > > points;
  for(int i = 0; i < clusters; i++) {
    for(auto p : cluster_contains[i]) {
        points.push_back({p, colors[i]});
    }
  }

  sort(points.begin(), points.end());
  SMatrix result(root, root);
  for(auto p : points) {
    result.insert(p.first.first, p.first.second) = p.second;
  }
  writeMatrix(string("../matrices/") + matrixname + string("grid.mtx"), result);
  return 0;
}
