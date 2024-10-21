#include "edit_distance.h"

/*************************************************************************************************
 *             === NODE SIMILARITY ===
 ************************************************************************************************/

std::tuple<torch::Tensor, IncidentBranchesMatch, torch::Tensor> nodes_similarity(
    torch::Tensor matchable_nodes, torch::Tensor angle_features_1, torch::Tensor angle_features_2,
    torch::Tensor scalar_features_1, torch::Tensor scalar_features_2, torch::Tensor scalar_features_std,
    torch::Tensor n_branches_1, torch::Tensor n_branches_2, torch::Tensor branch_uvector_1,
    torch::Tensor branch_uvector_2, bool rotation_invariant) {
    bool branch_uvector_available = branch_uvector_1.ndimension() == 3 && branch_uvector_2.ndimension() == 3;

    // Check input shapes
    TORCH_CHECK_VALUE(matchable_nodes.ndimension() == 2, "matchable_nodes must be a 2D tensor: (N1, N2), instead of ",
                      matchable_nodes.sizes());
    TORCH_CHECK_VALUE(angle_features_1.ndimension() == 4, "angle_features_1 must be a 4D tensor: (N1, B1, F_cos, 2),",
                      " instead of ", angle_features_1.sizes());
    TORCH_CHECK_VALUE(angle_features_2.ndimension() == 4, "angle_features_2 must be a 4D tensor: (N2, B2, F_cos, 2),",
                      " instead of ", angle_features_2.sizes());
    TORCH_CHECK_VALUE(scalar_features_1.ndimension() == 3, "scalar_features_1 must be a 3D tensor: (N1, B1, F_l2), ",
                      "instead of ", scalar_features_1.sizes());
    TORCH_CHECK_VALUE(scalar_features_2.ndimension() == 3, "scalar_features_2 must be a 3D tensor: (N2, B2, F_l2), ",
                      "instead of ", scalar_features_2.sizes());
    TORCH_CHECK_VALUE(scalar_features_std.ndimension() == 1, "scalar_features_std must be a 1D tensor of length F_l2.",
                      "instead of ", scalar_features_std.sizes());
    TORCH_CHECK_VALUE(n_branches_1.ndimension() == 1, "n_branches_1 must be a 1D tensor of length N1, instead of ",
                      n_branches_1.sizes());
    TORCH_CHECK_VALUE(n_branches_2.ndimension() == 1, "n_branches_2 must be a 1D tensor of length N2, instead of ",
                      n_branches_2.sizes());
    if (rotation_invariant && !branch_uvector_available) {
        TORCH_CHECK_VALUE(branch_uvector_1.ndimension() == 3, "branch_uvector_1 must be a 3D tensor: (N1, B1, 2), ",
                          "instead of ", branch_uvector_1.sizes());
        TORCH_CHECK_VALUE(branch_uvector_2.ndimension() == 3, "branch_uvector_2 must be a 3D tensor: (N2, B2, 2), ",
                          "instead of ", branch_uvector_2.sizes());
    }

    TORCH_CHECK_VALUE(angle_features_1.size(3) == 2, "angle_features_1 4th dimension must have a length of 2.",
                      " But provided shape is: ", angle_features_1.sizes(), ".");
    TORCH_CHECK_VALUE(angle_features_2.size(3) == 2, "angle_features_2 4th dimension must have a length of 2.",
                      " But provided shape is: ", angle_features_2.sizes(), ".");
    if (branch_uvector_available) {
        TORCH_CHECK_VALUE(branch_uvector_1.size(2) == 2, "branch_uvector_1 3rd dimension must have a length of 2.",
                          " But provided shape is: ", branch_uvector_1.sizes(), ".");
        TORCH_CHECK_VALUE(branch_uvector_2.size(2) == 2, "branch_uvector_2 3rd dimension must have a length of 2.",
                          " But provided shape is: ", branch_uvector_2.sizes(), ".");
    }

    const uint N1 = matchable_nodes.size(0), N2 = matchable_nodes.size(1);
    TORCH_CHECK_VALUE(angle_features_1.size(0) == N1, "Invalid number of nodes for angle_features_1 1st dimension.",
                      " Expected ", N1, " but got ", angle_features_1.size(0), ".");
    TORCH_CHECK_VALUE(angle_features_2.size(0) == N2, "Invalid number of nodes for angle_features_2 1st dimension.",
                      " Expected ", N2, " but got ", angle_features_2.size(0), ".");
    TORCH_CHECK_VALUE(scalar_features_1.size(0) == N1, "Invalid number of nodes for scalar_features_1 2nd dimension.",
                      " Expected ", N1, " but got ", scalar_features_1.size(0), ".");
    TORCH_CHECK_VALUE(scalar_features_2.size(0) == N2, "Invalid number of nodes for scalar_features_2 2nd dimension.",
                      " Expected ", N2, " but got ", scalar_features_2.size(0), ".");
    TORCH_CHECK_VALUE(n_branches_1.size(0) == N1, "Invalid number of nodes for n_branches_1. Expected ", N1,
                      " but got ", n_branches_1.size(0), ".");
    TORCH_CHECK_VALUE(n_branches_2.size(0) == N2, "Invalid number of nodes for n_branches_2. Expected ", N2,
                      " but got ", n_branches_2.size(0), ".");
    if (branch_uvector_available) {
        TORCH_CHECK_VALUE(branch_uvector_1.size(0) == N1,
                          "Incoherent number of nodes in cos_features_1 and branch_uvector_1");
        TORCH_CHECK_VALUE(branch_uvector_2.size(0) == N2,
                          "Incoherent number of nodes in cos_features_2 and branch_uvector_2");
    }

    const long F_cos = angle_features_1.size(2), F_l2 = scalar_features_1.size(2);
    TORCH_CHECK_VALUE(F_cos == angle_features_2.size(2),
                      "Incoherent number of features in angle_features_1: ", angle_features_1.sizes(),
                      " and angle_features_2: ", angle_features_2.sizes());
    TORCH_CHECK_VALUE(F_l2 == scalar_features_2.size(2),
                      "Incoherent number of features in scalar_features_1: ", scalar_features_1.sizes(),
                      " and scalar_features_2: ", scalar_features_2.sizes());
    TORCH_CHECK_VALUE(F_l2 == scalar_features_std.size(0),
                      "Incoherent number of features in scalar_features: ", scalar_features_1.sizes(),
                      " and scalar_features_std: ", scalar_features_std.sizes());

    const long maxB = angle_features_1.size(1);
    TORCH_CHECK_VALUE(maxB == angle_features_2.size(1),
                      "Incoherent number of branches in angle_features_1: ", angle_features_1.sizes(),
                      " and cos_features_2: ", angle_features_2.sizes());
    TORCH_CHECK_VALUE(maxB == scalar_features_1.size(1),
                      "Incoherent number of branches in angle_features_1: ", angle_features_1.sizes(),
                      " and scalar_features_1: ", scalar_features_1.sizes());
    TORCH_CHECK_VALUE(maxB == scalar_features_2.size(1),
                      "Incoherent number of branches in angle_features_2: ", angle_features_2.sizes(),
                      " and scalar_features_2: ", scalar_features_2.sizes());
    if (branch_uvector_available) {
        TORCH_CHECK_VALUE(maxB == branch_uvector_1.size(1),
                          "Incoherent number of branches in angle_features_1 and branch_uvector_1");
        TORCH_CHECK_VALUE(maxB == branch_uvector_2.size(1),
                          "Incoherent number of branches in angle_features_2 and branch_uvector_2");
    }

    // Define accessors to the input tensors
    torch::Tensor empty_u = torch::empty({0, 0, 2}, torch::kFloat32);
    auto u1_acc = empty_u.accessor<float, 3>();
    auto u2_acc = empty_u.accessor<float, 3>();
    if (branch_uvector_available) {
        u1_acc = branch_uvector_1.accessor<float, 3>();
        u2_acc = branch_uvector_2.accessor<float, 3>();
    }

    auto B1 = n_branches_1.accessor<int, 1>();
    auto B2 = n_branches_2.accessor<int, 1>();
    IncidentBranchesData data = {angle_features_1.accessor<float, 4>(),
                                 angle_features_2.accessor<float, 4>(),
                                 scalar_features_1.accessor<float, 3>(),
                                 scalar_features_2.accessor<float, 3>(),
                                 scalar_features_std.accessor<float, 1>(),
                                 u1_acc,
                                 u2_acc,
                                 branch_uvector_available};

    //
    auto matchable_nodes_acc = matchable_nodes.accessor<bool, 2>();

    // Instantiate the output tensors
    torch::Tensor similarities = torch::zeros({N1, N2}, torch::kFloat32);
    auto sim = similarities.accessor<float, 2>();
    torch::Tensor n_iterations = torch::zeros({N1, N2}, torch::kInt32);
    auto n_iters = n_iterations.accessor<int, 2>();
    IncidentBranchesMatch matches(N1, std::vector<torch::Tensor>(N2));

#pragma omp parallel for collapse(2)
    for (uint n1 = 0; n1 < N1; n1++) {
        for (uint n2 = 0; n2 < N2; n2++) {
            if (!matchable_nodes_acc[n1][n2]) continue;
            uint node_B1 = B1[n1], node_B2 = B2[n2];
            auto const& [similarity, branch_match, n_iter] =
                node_similarity(n1, n2, node_B1, node_B2, data, rotation_invariant);
            sim[n1][n2] = similarity;
            matches[n1][n2] = vector_to_tensor(branch_match);
            n_iters[n1][n2] = n_iter;
        }
    }

    return {similarities, matches, n_iterations};
}

std::tuple<float, std::vector<UIntPair>, uint> node_similarity(uint n1, uint n2, uint B1, uint B2,
                                                               const IncidentBranchesData& data,
                                                               bool rotation_invariant) {
    const uint F_cos = data.angle_features_1.size(2), F_l2 = data.scalar_features_1.size(2);

    // L2 similarity
    auto norm_similarity = [&](uint b1, uint b2) {
        auto const &f1 = data.scalar_features_1[n1][b1], &f2 = data.scalar_features_2[n2][b2];
        float totalSim = 0;
        for (uint f = 0; f < F_l2; f++) {
            const float sim = gaussian(f1[f] - f2[f], data.scalar_features_std[f]);
            totalSim += sim;
        }
        return totalSim;
    };

    // ROTATION INVARIANT == FALSE
    auto node_match_weight = [&](uint b1, uint b2, const UIntPair& iniMatch) {
        // Compute the cosine similarities
        auto const &f1 = data.angle_features_1[n1][b1], &f2 = data.angle_features_2[n2][b2];
        float cos_sim = 0;
        for (uint f = 0; f < F_cos; f++) {
            const float sim = Point(f1[f]).dot(f2[f]);
            cos_sim += sim;
        }

        // Compute the match weight: 1 - AVERAGE(cosine similarities + L2 similarities)
        const float norm_sim = norm_similarity(b1, b2);
        float sim = 1 - (cos_sim + norm_sim) / (F_cos + F_l2);
        return sim < 1 ? sim : 1;
    };

    // ROTATION INVARIANT == TRUE
    auto node_match_weight_invar = [&](uint b1, uint b2, const UIntPair& iniMatch) {
        // Compute the rotation correction u_rot12
        auto const u_bRef1 = Point(data.branch_uvector_1[n1][iniMatch[0]]),
                   u_bRef2 = Point(data.branch_uvector_2[n2][iniMatch[1]]);
        auto const& u_rot12 = u_bRef2.rotate_neg(u_bRef1);

        // Compute the cosine similarities
        auto const &f1 = data.angle_features_1[n1][b1], &f2 = data.angle_features_2[n2][b2];
        float cos_sim = 0;
        for (uint f = 0; f < F_cos; f++) cos_sim += Point(f1[f]).rotate(u_rot12).dot(f2[f]);

        // Compute the match weight: 1 - AVERAGE(cosine similarities + L2 similarities)
        float sim = 1 - (cos_sim + norm_similarity(b1, b2)) / (F_cos + F_l2);
        return sim < 1 ? sim : 1;
    };

    // Compute the optimal branch matching
    std::vector<UIntPair> initial_pairs;
    if (rotation_invariant) {
        for (uint b1 = 0; b1 < B1; b1++)
            for (uint b2 = 0; b2 < B2; b2++) initial_pairs.push_back({b1, b2});
    } else {
        bool has_uncertain_branches = true;
        if (data.uvectors_available) {
            has_uncertain_branches = false;
            for (uint b1 = 0; b1 < B1; b1++) {
                const auto u = Point(data.branch_uvector_1[n1][b1]).normalize();
                if (u.x < 0 && abs(u.y) < sin(10 * M_PI / 180)) {
                    has_uncertain_branches = true;
                    break;
                }
            }
            for (uint b2 = has_uncertain_branches ? B2 : 0; b2 < B2; b2++) {
                const auto u = Point(data.branch_uvector_2[n2][b2]).normalize();
                if (u.x < 0 && abs(u.y) < sin(10)) {
                    has_uncertain_branches = true;
                    break;
                }
            }
        }
        if (has_uncertain_branches)
            initial_pairs = {{0, 0}, {B1 - 1, 0}, {0, B2 - 1}};
        else
            initial_pairs = {{0, 0}};
    }
    if (rotation_invariant)
        return djikstra_optimal_branch_matching(B1, B2, node_match_weight_invar, initial_pairs, false);
    else
        return djikstra_optimal_branch_matching(B1, B2, node_match_weight, initial_pairs, true);
}

std::tuple<float, std::vector<UIntPair>, uint> djikstra_optimal_branch_matching(
    uint B1, uint B2, std::function<float(uint, uint, const UIntPair&)> eval_match_cost,
    std::vector<UIntPair> initial_pairs, const bool are_match_cost_independent_of_ini_pairs) {
    using namespace BranchMatching;

    // === INITIALIZE ===
    const std::size_t Npairs = initial_pairs.size();
    const std::array<std::size_t, 3> Ns = {Npairs, B1 + 1, B2 + 1};

    // Initialize the memoization of the match costs
    std::vector<float> match_costs;
    if (are_match_cost_independent_of_ini_pairs) match_costs = std::vector<float>(B1 * B2, -1);
    auto memoized_eval_match_cost = [&](uint b1, uint b2) {
        const std::size_t idx = b1 * B2 + b2;
        if (match_costs[idx] == -1) {
            match_costs[idx] = eval_match_cost(b1, b2, {0, 0});
            return match_costs[idx];
        } else
            return match_costs[idx];
    };

    // Prepare index mapping utility functions
    auto true_branches_indexes = [&](DjikstraStep match) {
        // Shift the indexes of the match to account for the initial branch index offset
        auto const& [iniB1, iniB2] = initial_pairs[match.iniPairId];
        return UIntPair{(iniB1 + match.b1) % B1, (iniB2 + match.b2) % B2};
    };

    // Instantiate the priority queue, the distance and parent matrix
    std::priority_queue<DjikstraPath, std::vector<DjikstraPath>, LowerPriorityPath> Q(LowerPriorityPath(B1, B2));
    std::vector<float> pathCosts(Npairs * (B1 + 1) * (B2 + 1), std::numeric_limits<float>::infinity());
    std::vector<no_init<uint>> previousSteps;
    previousSteps.resize(Npairs * (B1 + 1) * (B2 + 1) * 2);
    auto getPreviousStep = [&](std::size_t stepId) {
        return std::pair<uint, uint>{previousSteps[stepId * 2], previousSteps[stepId * 2 + 1]};
    };
    auto setPreviousStep = [&](std::size_t stepId, uint previousB1, uint previousB2) {
        previousSteps[stepId * 2] = previousB1;
        previousSteps[stepId * 2 + 1] = previousB2;
    };

    // === ENQUEUE INITIAL STEPS ===
    for (uint ini = 0; ini < Npairs; ini++) {
        const DjikstraStep iniStep = {ini, 0, 0};
        const std::size_t iniStepId = ini * Ns[1] * Ns[2];
        Q.push({astar_lower_bound(iniStep, B1, B2), iniStep});
        pathCosts[iniStepId] = 0;
        setPreviousStep(iniStepId, 0, 0);
    }

    // === RUN DIJKSTRA SEARCH ===
    uint n_iteration = 0;
    float match_cost, drop_cost;
    DjikstraPath path;
    do {
        TORCH_CHECK(!Q.empty(), "Djikstra algorithm terminated preemptively.");

        // Pop the path with the fewer cost this far (see LowerPriorityPath implementation below for more details)
        path = Q.top();
        const auto& step = path.step;
        Q.pop();

        // Stop the search if the current path reached a point where all branches of B1 or B2 have been evaluated
        if (step.b1 == B1 || step.b2 == B2) break;

        float pathCost = pathCosts[step.index(Ns)];

        // Skip this step if its using a stale path (i.e. a shorter path to this step has already been found)
        if (path.cost != pathCost + astar_lower_bound(step, B1, B2)) continue;

        // Prepare update utility functions
        auto update_path_if_shorter = [&](float nextPathCost, uint nextB1, uint nextB2) {
            const DjikstraStep nextStep = {step.iniPairId, nextB1, nextB2};
            const std::size_t nextStepId = nextStep.index(Ns);
            // If the path to nextStep is shorter from this current step...
            if (nextPathCost < pathCosts[nextStepId]) {
                // ... update the path cost and parent ...
                pathCosts[nextStepId] = nextPathCost;
                setPreviousStep(nextStepId, step.b1, step.b2);
                // ... and enqueue the next step with its new cost.
                Q.push({nextPathCost + astar_lower_bound(nextStep, B1, B2), nextStep});
                // Any other path to nextStep previously enqueued will be ignored by the above skip check.
            }
        };

        // Investigate the three possible steps from the current step:
        // => MATCHING b1 with b2
        auto const& [true_b1, true_b2] = true_branches_indexes(step);
        if (are_match_cost_independent_of_ini_pairs) {
            match_cost = memoized_eval_match_cost(true_b1, true_b2);
        } else {
            match_cost = eval_match_cost(true_b1, true_b2, initial_pairs[step.iniPairId]);
        }
        update_path_if_shorter(pathCost + match_cost, step.b1 + 1, step.b2 + 1);

        // => DROPPING b1
        drop_cost = DROP_COST;
        // If there is no more B1 branches left, add the cost of dropping all the remaining B2 branches
        if (step.b1 + 1 == B1) drop_cost += (B2 - step.b2) * DROP_COST;
        update_path_if_shorter(pathCost + drop_cost, step.b1 + 1, step.b2);

        // => DROPPING b2
        drop_cost = DROP_COST;
        // If there is no more B2 branches left, add the cost of dropping all the remaining B1 branches
        if (step.b2 + 1 == B2) drop_cost += B1 - step.b1 * DROP_COST;
        update_path_if_shorter(pathCost + drop_cost, step.b1, step.b2 + 1);

        n_iteration++;
    } while (true);

    // === BACKTRACK THE PATH ===
    float similarity = 0;
    std::vector<UIntPair> matches;
    matches.reserve(std::max(B1, B2));
    DjikstraStep step = path.step, parentStep = step;
    do {
        // Read the parent step from the parent matrix
        const std::size_t stepId = step.index(Ns);
        std::tie(parentStep.b1, parentStep.b2) = getPreviousStep(stepId);
        // If the parent step was diagonal (i.e. a match), save the matches
        if (parentStep.b1 == step.b1 - 1 && parentStep.b2 == step.b2 - 1) {
            matches.push_back(true_branches_indexes(parentStep));
            const float match_cost = pathCosts[stepId] - pathCosts[parentStep.index(Ns)];
            similarity += (2 * DROP_COST - match_cost) / (2 * DROP_COST);
        }
        // Continue to the parent step
        step.b1 = parentStep.b1;
        step.b2 = parentStep.b2;
    } while (!(step.b1 == 0 && step.b2 == 0));  // Stop when step is the initial step

    return {similarity / std::min(B1, B2), matches, n_iteration};
}

BranchMatching::LowerPriorityPath::LowerPriorityPath(uint B1, uint B2) : B1(B1), B2(B2) {}
/// @brief Return true if p1 has a lower priority than p2
bool BranchMatching::LowerPriorityPath::operator()(const DjikstraPath& p1, const DjikstraPath& p2) const {
    // The path with the lower cost has the higher priority
    if (p1.cost != p2.cost) return p1.cost > p2.cost;

    const auto &s1 = p1.step, s2 = p2.step;
    // Then the path closer to the end has the higher priority
    if (s1.b1 + s1.b2 != s2.b1 + s2.b2) return s1.b1 + s1.b2 < s2.b1 + s2.b2;
    // Then the path with the lower initial pair ID (then b1, then b2) has the higher priority (see below)
    return s1 > s2;
}

bool BranchMatching::DjikstraStep::operator>(const DjikstraStep& other) const {
    if (iniPairId != other.iniPairId) return iniPairId > other.iniPairId;
    if (b1 != other.b1) return b1 > other.b1;
    return b2 > other.b2;
}

/*************************************************************************************************
 *             === SHORTEST PATH ===
 ************************************************************************************************/

std::array<torch::Tensor, 2> shortest_secondary_path(const torch::Tensor& edge_list, const torch::Tensor& primary_nodes,
                                                     const torch::Tensor& secondary_nodes) {
    const std::size_t n_primary = primary_nodes.size(0), n_secondary = secondary_nodes.size(0);
    const std::size_t n_nodes = n_primary + n_secondary;

    auto edge_list_acc = edge_list.accessor<int, 2>();
    auto primary_acc = primary_nodes.accessor<int, 1>();
    // auto secondary_acc = secondary_nodes.accessor<int, 1>();

    /*
    // Initialize node lookup table
    std::vector<int> node_lookup(n_nodes, -1);
    for (std::size_t i = 0; i < n_primary; i++) node_lookup[primary_acc[i]] = i;
    for (std::size_t i = 0; i < n_secondary; i++) node_lookup[secondary_acc[i]] = i + n_primary;
    */

    // Initialize the distance matrix
    torch::Tensor distance_tensor = torch::full({(long)n_primary, (long)n_nodes}, -1, torch::kInt32);
    auto distance_acc = distance_tensor.accessor<int, 2>();
    for (std::size_t i = 0; i < n_nodes; i++) distance_acc[primary_acc[i]][i] = 0;

    // Initialize the backtrack matrix
    torch::Tensor backtrack_edge_node = torch::full({(long)n_primary, (long)n_nodes, 2}, -1, torch::kInt32);
    auto backtrack_edge_node_acc = backtrack_edge_node.accessor<int, 3>();

    // Initialize adjacency list
    auto const& adjacency_list = edge_list_to_adjlist(edge_list_acc, n_nodes);

    // Compute distance from primary nodes to all other nodes
    for (std::size_t p_primary_id = 0; p_primary_id < n_primary; p_primary_id++) {
        std::queue<int> to_visit, next_to_visit;
        const int p = primary_acc[p_primary_id];
        for (auto const& edge : adjacency_list[p]) {
            int neighbor = edge.other(p);
            to_visit.push(neighbor);
            backtrack_edge_node_acc[p][neighbor][0] = edge.id;
            backtrack_edge_node_acc[p][neighbor][1] = p;
        }

        int d = 1;
        while (!to_visit.empty()) {
            // Iterate over all nodes marked to visit
            while (!to_visit.empty()) {
                int node = to_visit.front();
                to_visit.pop();

                // Update the distance of the neighbor
                distance_acc[p_primary_id][node] = d;

                // If n is a primary node don't follow edges
                if (node < (int)n_primary) continue;

                // Otherwise oterate over all neighbors of the current node
                for (auto const& edge : adjacency_list[node]) {
                    // Skip if already visited or if the edge is the one that led to the current node
                    if (distance_acc[p_primary_id][edge.other(node)] != -1) continue;
                    // Update the backtrack matrix
                    backtrack_edge_node_acc[p_primary_id][edge.other(node)][0] = edge.id;
                    backtrack_edge_node_acc[p_primary_id][edge.other(node)][1] = node;
                    // Add the neighbor to the list of nodes to visit
                    next_to_visit.push(edge.other(node));
                }
            }

            // When all nodes have been visited, swap the queues ...
            std::swap(to_visit, next_to_visit);
            std::queue<int> empty;
            std::swap(next_to_visit, empty);

            // ... and increment the distance
            d++;

            // The loop stops when no more nodes were marked in next_to_visit
        }
    }
    return {distance_tensor, backtrack_edge_node};
}
