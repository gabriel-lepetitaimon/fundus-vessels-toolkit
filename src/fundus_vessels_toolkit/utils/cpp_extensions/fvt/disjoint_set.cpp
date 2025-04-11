#include "disjoint_set.h"

/*********************************************************************************************
 *             === Cycle Detection ===
 ********************************************************************************************/
bool has_cycle(torch::Tensor parent_list) {
    TORCH_CHECK(parent_list.dim() == 1, "Input tensor must be 1D.");
    auto sets = ConstantDisjointSet(parent_list.size(0));
    auto parent = parent_list.accessor<int, 1>();

    for (int i = 0; i < parent_list.size(0); i++) {
        int p = parent[i];
        if (p < 0) continue;
        if (sets.merge(p, i) == i) return true;
    }

    return false;
}

std::list<std::list<int>> find_cycles(torch::Tensor parent_list) {
    TORCH_CHECK(parent_list.dim() == 1, "Input tensor must be 1D.");
    auto sets = ConstantDisjointSet(parent_list.size(0));
    auto parent = parent_list.accessor<int, 1>();

    std::list<int> cyclesRoot;
    for (int i = 0; i < parent_list.size(0); i++) {
        int p = parent[i];
        if (p < 0) continue;
        if (sets.merge(p, i) == i) cyclesRoot.push_back(i);
    }

    std::list<std::list<int>> cycles;
    for (int root : cyclesRoot) {
        cycles.push_back({root});
        std::list<int>& cycle = cycles.back();
        int node = parent[root];
        while (node != root) {
            cycle.push_back(node);
            node = parent[node];
        }
    }

    return cycles;
}

/*********************************************************************************************
 *             === Disjoint Set ===
 ********************************************************************************************/
ConstantDisjointSet::ConstantDisjointSet(std::size_t n) : n((int)n) {
    parent.reserve(n);
    for (int i = 0; i < (int)n; i++) parent.push_back(i);
}

int ConstantDisjointSet::find(int u) {
    if (parent[u] == u) return u;
    return parent[u] = find(parent[u]);
}

int ConstantDisjointSet::merge(int u, int v) {
    int pu = find(u);
    int pv = find(v);

    if (pu == pv) return v;

    parent[pv] = pu;
    return pu;
}

int ConstantDisjointSet::size() const { return n; }

std::unordered_map<int, std::vector<int>> ConstantDisjointSet::get_sets() {
    std::unordered_map<int, std::vector<int>> sets;
    for (int i = 0; i < n; i++) sets[find(i)].push_back(i);
    return sets;
}