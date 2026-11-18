#ifndef KD_TREE_CUH
#define KD_TREE_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <memory>

#include "DArray.hpp"
#include "GranularParticles.hpp"

// =========================================================================
// Device Helper & Kernel Declarations
// =========================================================================

__device__ bool checkMoreThan10(float empty_space, float total);
__device__ bool isLeafNode(int count);

__device__ void copyBoundingBox(
    float *__restrict__ _d_bbox_min_x,
    float *__restrict__ _d_bbox_min_y,
    float *__restrict__ _d_bbox_min_z,
    float *__restrict__ _d_bbox_max_x,
    float *__restrict__ _d_bbox_max_y,
    float *__restrict__ _d_bbox_max_z,
    int child_node_idx,
    int parent_node_idx);

// Initializes node 0 (head, count, children, bounding box).
// If use_custom_bounds is false, the bounding box is taken from the first/last entries of the
// axis-sorted lists (tight particle bounds) and padded by root_padding * (largest extent).
__global__ void initRootNode(
    const float3 *__restrict__ d_pos,
    const int *__restrict__ list_x,
    const int *__restrict__ list_y,
    const int *__restrict__ list_z,
    int *__restrict__ split_axis,
    int *__restrict__ left_child,
    int *__restrict__ right_child,
    float *__restrict__ bbox_min_x,
    float *__restrict__ bbox_min_y,
    float *__restrict__ bbox_min_z,
    float *__restrict__ bbox_max_x,
    float *__restrict__ bbox_max_y,
    float *__restrict__ bbox_max_z,
    int *__restrict__ prim_head,
    int *__restrict__ prim_count,
    const int total_particles,
    const bool use_custom_bounds,
    const float3 custom_min,
    const float3 custom_max,
    const float root_padding);

__global__ void findSplitPlane(
    const float3 *__restrict__ d_pos,
    const int *__restrict__ list_x,
    const int *__restrict__ list_y,
    const int *__restrict__ list_z,
    int *__restrict__ split_axis,
    float *__restrict__ split_pos,
    float *__restrict__ bbox_min_x,
    float *__restrict__ bbox_min_y,
    float *__restrict__ bbox_min_z,
    float *__restrict__ bbox_max_x,
    float *__restrict__ bbox_max_y,
    float *__restrict__ bbox_max_z,
    const int *__restrict__ prim_head,
    const int *__restrict__ prim_count,
    int *__restrict__ children_generated,
    int num_active_nodes,
    int offset,
    const bool force_leaf);

__global__ void generateLeftAndRightChild(
    int *__restrict__ left_child,
    int *__restrict__ right_child,
    int *__restrict__ split_axis,
    float *__restrict__ split_pos,
    int *__restrict__ _d_children_spawn_count,
    float *__restrict__ _d_bbox_min_x,
    float *__restrict__ _d_bbox_min_y,
    float *__restrict__ _d_bbox_min_z,
    float *__restrict__ _d_bbox_max_x,
    float *__restrict__ _d_bbox_max_y,
    float *__restrict__ _d_bbox_max_z,
    int *__restrict__ _d_children_spawn_prefix,
    const int active_nodes,
    const int offset);

__global__ void classifyParticles(
    const float3 *__restrict__ d_pos,
    const int *__restrict__ _particle_to_node,
    const int *__restrict__ split_axis,
    const float *__restrict__ split_pos,
    int *__restrict__ _is_left,
    const int total_particles);

__global__ void scatterList(
    const int *__restrict__ _left_offset,
    const int *__restrict__ _is_left,
    const int *__restrict__ list_dim,
    int *__restrict__ temp_dim,
    const int *__restrict__ _particle_to_node,
    const int *__restrict__ prim_head,
    const int *__restrict__ prim_count,
    const int *__restrict__ _d_children_spawn_count,
    const int total_particles);

__global__ void setChildRanges(
    const int *__restrict__ _left_offset,
    const int *__restrict__ _is_left,
    const int *__restrict__ list_dim,
    const int *__restrict__ left_child,
    const int *__restrict__ right_child,
    const int *__restrict__ _d_children_spawn_count,
    int *__restrict__ prim_head,
    int *__restrict__ prim_count,
    const int active_nodes,
    const int offset);

__global__ void updateParticleToNode(
    int *__restrict__ _particle_to_node,
    const int *__restrict__ _is_left,
    const int *__restrict__ _d_children_spawn_count,
    const int *__restrict__ _d_left_child,
    const int *__restrict__ _d_right_child,
    const int total_particles);

// =========================================================================
// KD-Tree Host Class
// =========================================================================

class KDTree
{
public:
    // initializers are in the same order as the member declarations below
    KDTree(unsigned int size) : num_particles(size), _list_x(size), _list_y(size), _list_z(size),
                                _temp_x(size), _temp_y(size), _temp_z(size),
                                _particle_to_node(size), _is_left(size), _left_offset(size),
                                // Node list (initialize to 0 or 2 * size upper bound)
                                node_capacity(size),
                                _d_split_axis(size), _d_split_pos(size), _d_left_child(size), _d_right_child(size),
                                _d_children_spawn_count(size),
                                _d_bbox_min_x(size), _d_bbox_min_y(size), _d_bbox_min_z(size),
                                _d_bbox_max_x(size), _d_bbox_max_y(size), _d_bbox_max_z(size),
                                _d_prim_head(size), _d_prim_count(size),
                                // UPASS / DOWNPASS
                                _d_subtree_size(0), _d_preorder_address(0),
                                _d_start_levels(0), _d_size_levels(0),
                                // Final preorder tree
                                _d_out_split_axis(0), _d_out_split_position(0), _d_out_right_child(0),
                                _d_out_bbox_minmax(0), _d_out_prim_start(0), _d_out_prim_count(0)
    {
    }
    ~KDTree() = default;

    void constructKDTree(const std::shared_ptr<GranularParticles> &particles);

    // TEMPORARY debug check of the finished tree (slow, copies everything to the host), see KD_Tree.cu.
    // Remove together with its definition once the construction is validated.
    bool validateKDTree(const float3 *d_pos) const;

    // Use an explicit root bounding box instead of the default (tight particle bounds + padding).
    void setRootBounds(float3 min_corner, float3 max_corner)
    {
        _use_custom_root = true;
        _root_min = min_corner;
        _root_max = max_corner;
    }
    // Go back to the default root bounding box.
    void clearRootBounds() { _use_custom_root = false; }
    // Padding for the default root box, as a fraction of the largest particle extent.
    void setRootPadding(float fraction) { _root_padding = fraction; }

private:
    bool _use_custom_root = false;
    float3 _root_min = {0.f, 0.f, 0.f};
    float3 _root_max = {0.f, 0.f, 0.f};
    float _root_padding = 0.01f;

    void constructKDTreeHelper(const float3 *d_pos, int num_particles);
    void scanAndScatter(DArray<int> &list_dim, DArray<int> &temp_dim);

    unsigned int num_particles;

    // Particle List (doesn't change)
    DArray<int> _list_x; // store particle ids, but sorted based on respective axis
    DArray<int> _list_y;
    DArray<int> _list_z;
    DArray<int> _temp_x;
    DArray<int> _temp_y;
    DArray<int> _temp_z;

    DArray<int> _particle_to_node;
    DArray<int> _is_left;
    DArray<int> _left_offset;

    // Node List (changes based on number of nodes in KD-Tree)
    unsigned int node_capacity;
    unsigned int _num_nodes = 0;  // number of nodes in the finished tree
    unsigned int _num_levels = 0; // number of tree levels, _d_start_levels / _d_size_levels have this many entries
    DArray<int> _d_split_axis;
    DArray<float> _d_split_pos;
    DArray<int> _d_left_child; // index of left/right child of current node
    DArray<int> _d_right_child;
    DArray<int> _d_children_spawn_count;

    DArray<float> _d_bbox_min_x;
    DArray<float> _d_bbox_min_y;
    DArray<float> _d_bbox_min_z;
    DArray<float> _d_bbox_max_x;
    DArray<float> _d_bbox_max_y;
    DArray<float> _d_bbox_max_z;

    DArray<int> _d_prim_head;
    DArray<int> _d_prim_count;

    // for UPASS and DOWNPASS when constructing preorder version of KD-TREE
    DArray<int> _d_subtree_size;
    DArray<int> _d_preorder_address;

    // Levels for UPASS and DOWNPASS Algorithm
    DArray<int> _d_start_levels;
    DArray<int> _d_size_levels;

    // Once Tree is fully built, will store in preorder traversal
    DArray<int> _d_out_split_axis;
    DArray<int> _d_out_split_position;
    DArray<int> _d_out_right_child;
    DArray<int> _d_out_bbox_minmax;
    DArray<int> _d_out_prim_start;
    DArray<int> _d_out_prim_count;
};

#endif // KD_TREE_CUH