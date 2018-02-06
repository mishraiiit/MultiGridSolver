#ifndef UTILITY_H
#define UTILITY_H
#include <vector>

std::vector<std::pair<int, std::pair<double, double> > > union_list(std::vector<std::pair<int, double> > arg1, std::vector<std::pair<int, double> > arg2);
std::vector<std::pair<int, std::pair<double, double> > > intersection_list(std::vector<std::pair<int, double> > arg1, std::vector<std::pair<int, double> > arg2);

#endif