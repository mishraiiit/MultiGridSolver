#include "../Utility.h"
#include <assert.h>

void testUnionList() {
    // Test 1.
    {
        std::vector<std::pair<int, double> > arg1 = {{0, 2}, {1, 5}, {3, 4}, {8, 8}};
        std::vector<std::pair<int, double> > arg2 = {{1, 2}, {3, 5}, {5, 4}, {8, 8}};
        std::vector<std::pair<int, std::pair<double, double> > > expected = \
        {{0, {2, 0}}, {1, {5, 2}}, {3, {4, 5}}, {5, {0, 4}}, {8, {8, 8}}};
        assert(union_list(arg1, arg2) == expected);
    }
}

void testIntersectionList() {
    // Test 1.
    {
        std::vector<std::pair<int, double> > arg1 = {{0, 2}, {1, 5}, {3, 4}, {8, 8}};
        std::vector<std::pair<int, double> > arg2 = {{1, 2}, {3, 5}, {5, 4}, {8, 8}};
        std::vector<std::pair<int, std::pair<double, double> > > expected = \
        {{1, {5, 2}}, {3, {4, 5}}, {8, {8, 8}}};
        assert(intersection_list(arg1, arg2) == expected);
    }
}

int main() {
    testUnionList();
    testIntersectionList();
}