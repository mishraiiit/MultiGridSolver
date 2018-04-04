#ifndef UTILITY_H
#define UTILITY_H
#include <vector>
#include <set>

std::vector<std::pair<int, std::pair<double, double> > > union_list(std::vector<std::pair<int, double> > arg1, std::vector<std::pair<int, double> > arg2);
std::vector<std::pair<int, std::pair<double, double> > > intersection_list(std::vector<std::pair<int, double> > arg1, std::vector<std::pair<int, double> > arg2);
std::vector<int> merge_sets(std::vector<int> arg1, std::vector<int> arg2);

#endif