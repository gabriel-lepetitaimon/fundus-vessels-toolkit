#include <pybind11/stl.h>

#include "common.h"
#include "disjoint_set.h"

/*********************************************************************************************
 *             === SKELETON PARSING ===
 *********************************************************************************************/
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

/**************************************************************************************
 *             === PYBIND11 BINDINGS ===
 **************************************************************************************/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("has_cycle", &has_cycle, "Find cycles in a list of parent."); }
