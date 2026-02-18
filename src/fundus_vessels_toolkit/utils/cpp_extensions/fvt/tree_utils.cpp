#include "tree.h"

Tree::Tree(const Tensor1DAcc<long>& tree_list) : nodes(tree_list.size(0)) {
    for (long i = 0; i < tree_list.size(0); ++i) {
        nodes[i].id = i;
        nodes[i].parent = tree_list[i];
        if (tree_list[i] >= 0)
            nodes[tree_list[i]].children.push_back(i);
        else
            root_nodes.push_back(i);
    }
}

torch::Tensor tree_distance(const torch::Tensor& tree_tensor) {
    const Tree tree(tree_tensor.accessor<long, 1>());
    long N = tree_tensor.size(0);

    auto out = torch::full({2, N, N}, std::nanf(""), torch::kFloat);
    auto out_acc = out.accessor<float, 3>();
    auto path_dist = out_acc[0], common_ancestor_dist = out_acc[1];

    for (const auto& root_id : tree.root_nodes) {
        std::vector<long> stack{root_id};
        std::list<long> related_nodes{root_id};
        path_dist[root_id][root_id] = common_ancestor_dist[root_id][root_id] = 0;

        while (!stack.empty()) {
            long node_id = stack.back();
            stack.pop_back();
            const auto& node = tree.nodes[node_id];

            auto ca_dist_node_ = common_ancestor_dist[node_id];
            auto path_dist_node_ = path_dist[node_id];

            for (const auto& child_id : node.children) {
                auto ca_dist_child_ = common_ancestor_dist[child_id];
                auto path_dist_child_ = path_dist[child_id];

                path_dist_child_[child_id] = ca_dist_child_[child_id] = 0;
                path_dist_node_[child_id] = path_dist_child_[node_id] = 1;
                ca_dist_node_[child_id] = -1;
                ca_dist_child_[node_id] = 1;
                for (const auto& i : related_nodes) {
                    if (i == node_id) continue;

                    path_dist_child_[i] = path_dist[i][child_id] = path_dist_node_[i] + 1;
                    if (common_ancestor_dist[i][node_id] < 0) {
                        // If i is a ancestor of node_id (and therefore of child id) derive ca_dist from path_dist
                        ca_dist_child_[i] = path_dist_child_[i];
                        common_ancestor_dist[i][child_id] = -path_dist_child_[i];
                    } else {
                        ca_dist_child_[i] = std::max(ca_dist_node_[i], 0.0f) + 1;
                        common_ancestor_dist[i][child_id] = common_ancestor_dist[i][node_id];
                    }
                }

                stack.push_back(child_id);
                related_nodes.push_back(child_id);
            }
        }
    }
    return out;
}