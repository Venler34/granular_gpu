#ifndef KD_TREE_CUH
#define KD_TREE_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <memory>

#include "DArray.hpp"
#include "GranularParticles.h" // Forward-declare class GranularParticles if header name differs

// =========================================================================
// Device Helper & Kernel Declarations
// =========================================================================

__device__ bool checkMoreThan10(float bb_max, float bb_min, float dim);
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

__device__ void rearrangeList(
    const int *__restrict__ _left_offset,
    const int *__restrict__ _is_left,
    const int *__restrict__ list_dim,
    int *__restrict__ temp_dim,
    int *__restrict__ _particle_to_node,
    const int *__restrict__ prim_head,
    const int *__restrict__ prim_count,
    int idx,
    const int *__restrict__ _d_left_child,
    const int *__restrict__ _d_right_child);

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
    int offset);

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
    int *__restrict__ split_axis,
    float *__restrict__ split_pos,
    int *__restrict__ _is_left);

__global__ void scatterAxisListInPlace(
    const int *__restrict__ _left_offset,
    const int *__restrict__ _is_left,
    const int *__restrict__ list_x,
    const int *__restrict__ list_y,
    const int *__restrict__ list_z,
    int *__restrict__ temp_x,
    int *__restrict__ temp_y,
    int *__restrict__ temp_z,
    int *__restrict__ _particle_to_node,
    const int *__restrict__ prim_head,
    const int *__restrict__ prim_count,
    const int node_offset,
    const int active_nodes,
    const int *__restrict__ _d_left_child,
    const int *__restrict__ _d_right_child,
    const int *__restrict__ _d_children_spawn_count);

// =========================================================================
// KD-Tree Host Class
// =========================================================================

class KDTree
{
public:
    KDTree(unsigned int size) : num_particles(size), _list_x(size), _list_y(size),
                                _list_z(size), _particle_to_node(size), _is_left(size), _left_offset(size),
                                _temp_x(size), _temp_y(size), _temp_z(size),
                                // Node list (initialize to 0 or 2 * size upper bound)
                                _d_split_axis(size), _d_split_pos(size), _d_left_child(size), _d_right_child(size),
                                _d_children_spawn_count(size),
                                _d_bbox_min_x(size), _d_bbox_min_y(size), _d_bbox_min_z(size),
                                _d_bbox_max_x(size), _d_bbox_max_y(size), _d_bbox_max_z(size),
                                _d_prim_head(size), _d_prim_count(size), node_capacity(size),
                                // UPASS / DOWNPASS
                                _d_subtree_size(0), _d_preorder_address(0),
                                _d_start_levels(0), _d_size_levels(0),
                                // Final preorder tree
                                _d_out_split_axis(0), _d_out_split_position(0), _d_out_right_child(0),
                                _d_out_bbox_minmax(0), _d_out_prim_start(0), _d_out_prim_count(0)
    {
    }
    KDTree() = default;
    ~KDTree() = default;

    void constructKDTree(const std::shared_ptr<GranularParticles> &particles);

private:
    void constructKDTreeHelper(const float3 *d_pos, int num_particles);

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