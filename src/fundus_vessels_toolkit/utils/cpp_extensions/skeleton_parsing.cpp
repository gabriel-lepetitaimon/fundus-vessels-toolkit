#include "neighbors.h"

/********************************************************************************************
 *                                 DETECT SKELETON NODES                                    *
 *******************************************************************************************/  

enum SkeletonRank {
    NONE = 0,
    ENDPOINT = 1,
    BRANCH = 2,
    JUNCTION3 = 3,
    JUNCTION4 = 4,
    HOLLOW_CROSS = 5,
};

/**
 * @brief Detect junction and endpoints base on neighborhood.
 * 
 * 
 * @param neighbors Value of the 8 neighbors of the pixel. The top-left neighbor is the most significant bit.
 * @return int 1 if it is an endpoint, 3 or 4 if it is a junction, 2 otherwise.
*/
SkeletonRank detect_skeleton_rank(const bool pixelValue, const uint8_t neighbors) {
    if(!pixelValue){
        return hit(neighbors, 0b01010101) ? SkeletonRank::HOLLOW_CROSS : SkeletonRank::NONE;
    }

    // -- Single endpoints --
    if(neighbors == 0)
        return SkeletonRank::ENDPOINT;
    
    for (int s = 0; s < 8; s++){
        const uint8_t neighbors_rolled = roll_neighbors(neighbors, s);

        // -- Endpoints --
        if(hit_and_miss(neighbors_rolled, 0b01000000, 0b00011111))
            return SkeletonRank::ENDPOINT;

        // -- 3 branches junctions --
        // Y shape
        if (hit(neighbors_rolled, 0b10100100)) {
            if(miss(neighbors_rolled, 0b01010001) || miss(neighbors_rolled, 0b01001010))
                return SkeletonRank::JUNCTION3;
        }
        
        // T shape
        if (hit_and_miss(neighbors_rolled, 0b00010101, 0b01001010))
            return SkeletonRank::JUNCTION3;
    }

    // -- 4 branches junctions --
    if(hit_and_miss(neighbors, 0b10101010, 0b01010101) 
    || hit_and_miss(neighbors, 0b01010101, 0b10101010)
    || hit(neighbors, 0b00011100))
        return SkeletonRank::JUNCTION4;

    return SkeletonRank::BRANCH;    
}


/**
 * @brief Fix the skeleton of a binary image.
 * 
 * @param skeleton Binary 2d image representing the skeleton.
 * @param fix_hollow If true, fill the hollow crosses of the skeleton.
 * @param remove_single_endpoints If true, remove terminal branches made of a single point.
 * @return torch::Tensor The fix skeleton were endpoints are set to 1, junctions are set to 2 or 3 and the rest of the skeleton is set to 2.
 * 
*/
torch::Tensor detect_skeleton_nodes(torch::Tensor skeleton, bool fix_hollow=true, bool remove_single_endpoints=true) {
    skeleton = torch::constant_pad_nd(skeleton, {1, 1, 1, 1}, 0);
    auto skeleton_accessor = skeleton.accessor<bool, 2>();
    int H = skeleton.size(0), W = skeleton.size(1);

    torch::Tensor parsed_skeleton = torch::zeros_like(skeleton, torch::kInt);
    auto parsed_accessor = parsed_skeleton.accessor<int, 2>();


    std::vector<Point> hollow_crosses;
    std::vector<Point> endpoints;

    #pragma omp parallel for collapse(2) reduction(merge: endpoints, hollow_crosses)
    for (int y = 0; y < H-1; y++) {
        for (int x = 1; x < W-1; x++) {
            const uint8_t neighbors = get_neighborhood(skeleton_accessor, y, x);
            const SkeletonRank point = detect_skeleton_rank(skeleton_accessor[y][x], neighbors);
            if (point == SkeletonRank::HOLLOW_CROSS){
                if (fix_hollow)
                    hollow_crosses.push_back({y, x});
            } else if (remove_single_endpoints && point == SkeletonRank::ENDPOINT){
                endpoints.push_back({y, x});
            }
            parsed_accessor[y][x] = point;
        }
    }

    // Fill Hollow Crosses and decrement their neighbors
    for (const Point& p : hollow_crosses){
        parsed_accessor[p.y][p.x] = SkeletonRank::BRANCH;
        for(const PointWithID& n : NEIGHBORHOOD){
            if (parsed_accessor[p.y + n.y][p.x + n.x] > 0)
                parsed_accessor[p.y + n.y][p.x + n.x]--;
        }
    }

    // Detect skeleton point again on hollow crosses
    for (const Point& p : hollow_crosses){
        const uint8_t neighbors = get_neighborhood(parsed_accessor, p.y, p.x);
        parsed_accessor[p.y][p.x] = detect_skeleton_rank(parsed_accessor[p.y][p.x], neighbors);
    }

    // Remove single endpoints
    #pragma omp parallel for
    for (const Point& p : endpoints) {
        bool singleEndpoint = true;
        for (const PointWithID& n : NEIGHBORHOOD){
            int n_rank = parsed_accessor[p.y + n.y][p.x + n.x];
            if(n_rank >= SkeletonRank::JUNCTION3){
                parsed_accessor[p.y + n.y][p.x + n.x] = n_rank-1;
                singleEndpoint = true;
                break;
            } else if (n_rank > SkeletonRank::ENDPOINT){
                singleEndpoint = false;
                break;
            }
        }
        if(singleEndpoint)
                parsed_accessor[p.y][p.x] = SkeletonRank::NONE;
    }

    return parsed_skeleton.index({torch::indexing::Slice(1,-1), torch::indexing::Slice(1,-1)});
}

/********************************************************************************************
 *                                      SKELETON PARSING                                    *
 *******************************************************************************************/  

enum PixelRole {
    BACKGROUND = 0,
    BRANCH_ROLE = 1,
    NODE = 2,
};

inline bool is_node(int pixelRole){
    return pixelRole >= PixelRole::NODE;
}


/**
 * @brief Track the next neighbors of a pixel in a skeleton.
 * 
 * @param skeleton 2d image representing the skeleton with identified junctions and branches.
 * @param current Coordinates of the pixel.
 * @param ignore_neighbor Id of the neighbor to ignore.
 * @param ignore_branches_close_to_node If true, ignore branches close to a node.
 * @return std::pair<std::vector<PointWithID>, std::vector<PointWithID>> 
 *     - The coordinates of neighbor nodes.
 *     - The coordinates of neighbor branches.
*/
std::pair<std::vector<PointWithID>, std::vector<PointWithID>> track_next_neighbors(
    Tensor2DAccessor<int> skeleton,  
    Point current,
    int ignore_neighbor=-1,
    bool ignore_branches_close_to_node=false
){
    std::vector<PointWithID> nodes;
    std::vector<PointWithID> branches;
    std::array<int, 8> neighbors;
    for (const PointWithID& n : NEIGHBORHOOD){
        neighbors[n.id] = skeleton[current.y + n.y][current.x + n.x];
    }

    for (const PointWithID& n : NEIGHBORHOOD){       
        if (n.id == ignore_neighbor)
            continue;

        const int ny = current.y + n.y, nx = current.x + n.x, n_id = n.id;
        int pixelRole = neighbors[n.id];
        if (pixelRole == PixelRole::BACKGROUND)
            continue;
        else if(pixelRole == PixelRole::BRANCH_ROLE){
            if(ignore_branches_close_to_node){
                const int next_n_id = n_id < 7 ? n_id+1 : 0;
                const int prev_n_id = n_id > 0 ? n_id-1 : 7;
                if (is_node(neighbors[next_n_id]) || is_node(neighbors[prev_n_id]))
                    continue;
            }
            branches.push_back({ny, nx, n.id});
        } else
            nodes.push_back({ny, nx, n.id});
    }
    return {nodes, branches};
}

/**
 * @brief Parse a skeleton image into a graph of branches and nodes.
 * 
 * @param skeleton 2d image representing the skeleton with identified junctions and branches.
 * 
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
 *     - The labeled map of branches and nodes
 *     - The coordinates of the nodes
 *     - The edge list. Each edge is composed of the id of the two nodes and the id of the branch connecting them.
*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int> parse_skeleton(torch::Tensor skeleton) {
    int H = skeleton.size(0), W = skeleton.size(1);
    auto skel_acc = skeleton.accessor<int, 2>();

    torch::Tensor labeled_skeleton = torch::zeros_like(skeleton, torch::kInt);
    auto label_acc = labeled_skeleton.accessor<int, 2>();
    std::vector<PointWithID> nodes;
    int nNodes = 0;


    // -- Search junctions and endpoints --
    #pragma omp parallel for collapse(2) reduction(merge: nodes)
    for (int y = 1; y < H-1; y++) {
        for (int x = 1; x < W-1; x++) {
            const int pixelRole = skel_acc[y][x];
            if(pixelRole != PixelRole::NODE)
                continue;
            
            int node_id;
            #pragma omp atomic capture
            {
                node_id = nNodes;
                nNodes++;
            }
            node_id++;
            // Junctions
            nodes.push_back({y, x, node_id});
            label_acc[y][x] = -node_id;
        }
    }

    // -- Search branches --
    int nBranches = 0;
    std::vector<std::array<int32_t, 3>> edge_list;
    edge_list.reserve(nNodes);
    torch::Tensor nodes_coordinates = torch::zeros({nNodes, 2}, torch::kInt);
    auto nodes_coord_acc = nodes_coordinates.accessor<int, 2>();
    
    for (const PointWithID& node : nodes){
        
        #ifdef DEBUG
        std::cout << "====================================================" << std::endl;
        std::cout << "Node " << node.id << "(" << node.y << ", " << node.x << ")" << std::endl;
        #endif
        
        
        // Save node coordinates
        nodes_coord_acc[node.id-1][0] = node.y;
        nodes_coord_acc[node.id-1][1] = node.x;

        // -- Track next nodes and branches --
        auto[next_nodes, next_branches] = track_next_neighbors(skel_acc, {node.y, node.x});
        
        // Remember connection with nodes next to the current node
        #ifdef DEBUG
        std::cout << "\tNeighbor nodes: " << next_nodes.size() << std::endl;
        #endif
        for (const PointWithID& n : next_nodes){
            int neighbor_node_id = -label_acc[n.y][n.x];
            if (node.id < neighbor_node_id){
                edge_list.push_back({node.id, neighbor_node_id, 0});
            }
            #ifdef DEBUG
            std::cout << "\t\t Node "<< neighbor_node_id << "(" << n.y << ", " << n.x << ")" << std::endl;
            #endif
        }

        // Track branches not yet labelled
        #ifdef DEBUG
        std::cout << "\tNeighbor branches: " << next_branches.size() << std::endl;
        #endif
        for (const PointWithID& n : next_branches){
            #ifdef DEBUG
            std::cout << "\t\t == Branch " ;
            #endif

            if(label_acc[n.y][n.x] == 0) {
                nBranches++;
                const int branch_id = nBranches;

                PointWithID tracker = n;
                std::vector<PointWithID> tracked_next_nodes, tracked_next_branches;
                
                #ifdef DEBUG
                std::cout << branch_id << " ==" << std::endl << "\t\t\t";
                #endif
                while(label_acc[tracker.y][tracker.x]==0){
                    #ifdef DEBUG
                    std::cout << "(" << tracker.y << ", " << tracker.x << ") -> ";
                    #endif
                    
                    // Label the branch
                    label_acc[tracker.y][tracker.x] = branch_id;

                    // Track next nodes and branches
                    std::tie(tracked_next_nodes, tracked_next_branches) = 
                        track_next_neighbors(skel_acc, {tracker.y, tracker.x}, opposite_neighbor_id(tracker.id), true);

                    if(tracked_next_nodes.size()==0 && tracked_next_branches.size() >= 0)
                        tracker = tracked_next_branches[0];  
                    else
                        break; // If node is found stop the tracking
                }
                #ifdef DEBUG
                std::cout << std::endl;
                #endif
                
                if (tracked_next_nodes.size() == 1){
                    // Save connectivity
                    const auto& other_node = tracked_next_nodes[0];
                    const int other_node_id = -label_acc[other_node.y][other_node.x];
                    if(node.id < other_node_id)
                        edge_list.push_back({node.id, other_node_id, branch_id});
                    else
                        edge_list.push_back({other_node_id, node.id, branch_id});
                    #ifdef DEBUG
                    std::cout << "\t\t\t -> Node " << other_node_id<< " (" << other_node.y << ", " << other_node.x << ")" << std::endl;
                    #endif
                    
                } else {
                    // Todo: Handle invalid skeleton
                    #ifdef DEBUG
                    if(tracked_next_nodes.size() > 1){
                        std::cout << "\t\t!! Nodes found: ";
                        for (const Neighbor& n : tracked_next_nodes){
                            std::cout << "(" << n.y << ", " << n.x << ") ";
                        }
                        std::cout << " !!" << std::endl;
                    } else if (tracked_next_branches.size() > 1){
                        std::cout << "\t\t!! Multiple branches candidates: ";
                        for (const Neighbor& n : tracked_next_branches){
                            std::cout << "(" << n.y << ", " << n.x << ") ";
                        }
                        std::cout << "!!" << std::endl;
                    } else if (label_acc[tracker.y][tracker.x] > 0) {
                        std::cout << "\t\t!! Branch (" << tracker.y << ", " << tracker.x << ") already tracked: " << label_acc[tracker.y][tracker.x] << " !!" << std::endl;
                    }
                    #endif
                }

            } 
            #ifdef DEBUG
            else { std::cout << label_acc[n.y][n.x] << " == (already tracked)" << std::endl; }
            #endif
        }
    }   

    // -- Create edge list tensor --

    // const torch::Tensor edge_list_tensor = torch::from_blob(edge_list.data(), {(long)edge_list.size(), 3}, torch::kInt32);
    
    torch::Tensor edge_list_tensor = torch::zeros({(int)edge_list.size(), 2}, torch::kInt32);
    auto edge_list_acc = edge_list_tensor.accessor<int32_t, 2>();
    int iEdgeNoBranch = 0;
    for (const auto& v : edge_list){
        const int& edge_id = v[2];
        if(edge_id>0){
            edge_list_acc[edge_id-1][0] = v[0]-1;
            edge_list_acc[edge_id-1][1] = v[1]-1;
        } else {
            edge_list_acc[nBranches+iEdgeNoBranch][0] = v[0]-1;
            edge_list_acc[nBranches+iEdgeNoBranch][1] = v[1]-1;
            iEdgeNoBranch++;
        }
    }
    return {labeled_skeleton, edge_list_tensor, nodes_coordinates, nBranches};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("detect_skeleton_nodes", &detect_skeleton_nodes, "Detect junctions and endpoints in a skeleton.");
    m.def("parse_skeleton", &parse_skeleton, "Parse a skeleton image into a graph of branches and nodes.");
}
