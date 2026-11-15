#pragma once

#include "DArray.hpp"
#include "helper_math.h"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <memory>

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
    ~KDTree() = default;

private:
    unsigned int num_particles;

    // Particle List (doesn't change)
    DArray<int> _list_x; // store particle ids, but sorted based on respective axis
    DArray<int> _list_y;
    DArray<int> _list_z;
    DArray<int> _temp_x;
    DArray<int> _temp_y;
    DArray<int> _temp_z;

    DArray<int> _particle_to_node;
    DArray<bool> _is_left;
    DArray<int> _left_offset;

    // Node List (changes based on number of nodes in KD-Tree)
    unsigned int node_capacity;
    DArray<int> _d_split_axis;
    DArray<int> _d_split_pos;
    DArray<int> _d_left_child; // index of left/right child of current node 
    DArray<int> _d_right_child;
    DArray<int> _d_children_spawn_count;

    DArray<int> _d_bbox_min_x;
    DArray<int> _d_bbox_min_y;
    DArray<int> _d_bbox_min_z;
    DArray<int> _d_bbox_max_x;
    DArray<int> _d_bbox_max_y;
    DArray<int> _d_bbox_max_z;

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