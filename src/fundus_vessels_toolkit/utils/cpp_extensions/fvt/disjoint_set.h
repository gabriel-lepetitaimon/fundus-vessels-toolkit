#ifndef DISJOINT_SET_H
#define DISJOINT_SET_H

#include <unordered_map>
#include <vector>

#include "common.h"

bool has_cycle_tensor(torch::Tensor parent_list);
std::vector<std::vector<int>> find_cycles_tensor(torch::Tensor parent_list);
std::vector<std::vector<int>> find_cycles(const std::vector<int>& parents);
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