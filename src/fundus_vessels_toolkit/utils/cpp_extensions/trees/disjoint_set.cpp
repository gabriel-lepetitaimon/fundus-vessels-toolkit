#include "disjoint_set.h"

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