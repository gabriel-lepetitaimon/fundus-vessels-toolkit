#ifndef DISJOINT_SET_H
#define DISJOINT_SET_H

#include <unordered_map>
#include <vector>

class ConstantDisjointSet {
   public:
    ConstantDisjointSet(std::size_t n);
    int find(int u);
    int merge(int u, int v);

    int size() const;
    std::unordered_map<int, std::vector<int>> get_sets();

   private:
    int n;
    std::vector<int> parent;
};

#endif