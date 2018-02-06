#include "Utility.h"

std::vector<std::pair<int, std::pair<double, double> > > union_list(std::vector<std::pair<int, double> > arg1, std::vector<std::pair<int, double> > arg2) {
    std::vector< std::pair<int, std::pair<double, double> > > result; 
    int i = 0, j = 0;
    while(i < arg1.size() && j < arg2.size()) {
        if(arg1[i].first < arg2[j].first) {
            result.push_back({arg1[i].first, {arg1[i].second, 0}});
            i = i + 1;
        } else if(arg1[i].first > arg2[j].first) {
            result.push_back({arg2[j].first, {0, arg2[j].second}});
            j = j + 1;
        } else {
            result.push_back({arg1[i].first, {arg1[i].second, arg2[j].second}});
            i = i + 1;
            j = j + 1;
        }
    }
    while(i < arg1.size()) {
        result.push_back({arg1[i].first, {arg1[i].second, 0}});
        i = i + 1;
    }
    while(j < arg2.size()) {
        result.push_back({arg2[j].first, {0, arg2[j].second}});
        j = j + 1;
    }
    return result;
}

std::vector<std::pair<int, std::pair<double, double> > > intersection_list(std::vector<std::pair<int, double> > arg1, std::vector<std::pair<int, double> > arg2) {
    std::vector< std::pair<int, std::pair<double, double> > > result; 
    int i = 0, j = 0;
    while(i < arg1.size() && j < arg2.size()) {
        if(arg1[i].first < arg2[j].first) {
            i = i + 1;
        } else if(arg1[i].first > arg2[j].first) {
            j = j + 1;
        } else {
            result.push_back({arg1[i].first, {arg1[i].second, arg2[j].second}});
            i = i + 1;
            j = j + 1;
        }
    }
    return result;
}