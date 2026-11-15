#include <algorithm>
#include <cfloat> // FLT_MAX for bounding box bounds and distance queries
#include <cmath>
#include <cstdint> // uint32_t, uint64_t for Morton codes or node bitmasks
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>   // Permuting particle properties into tree leaf order
#include <thrust/scan.h>     // Exclusive/inclusive scan for leaf/node offsets
#include <thrust/sequence.h> // Generating 0..N-1 index arrays
#include <thrust/sort.h>     // Radix sort / sort_by_key (Morton codes or split axes)
#include <thrust/reduce.h>   // Min/max reductions for bounding boxes
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>     // float3, int3, etc.
#include <vector_functions.h> // make_float3, make_int3

#include "DArray.hpp"        // GPU device buffer abstraction used across the solver
#include "CUDAFunctions.cuh" // CUDA error checking macros (CUDA_CALL, CHECK_KERNEL) and block size
#include "helper_math.h"     // Vector math operations (dot, length, normalize, float3 operators)

// Custom device comparators reading directly from float3* pos
struct CompareAxisX
{
    const float3 *pos;
    __host__ __device__ explicit CompareAxisX(const float3 *p) : pos(p) {}

    __host__ __device__ bool operator()(const int a, const int b) const
    {
        return pos[a].x < pos[b].x;
    }
};

struct CompareAxisY
{
    const float3 *pos;
    __host__ __device__ explicit CompareAxisY(const float3 *p) : pos(p) {}

    __host__ __device__ bool operator()(const int a, const int b) const
    {
        return pos[a].y < pos[b].y;
    }
};

struct CompareAxisZ
{
    const float3 *pos;
    __host__ __device__ explicit CompareAxisZ(const float3 *p) : pos(p) {}

    __host__ __device__ bool operator()(const int a, const int b) const
    {
        return pos[a].z < pos[b].z;
    }
};

__device__ bool checkMoreThan10(int bb_max, int bb_min, int dim)
{
    int total = bb_max - bb_min;
    int diff = min(abs(dim - bb_min), abs(bb_max - dim)); // double check logic if remove abs

    return diff * 10 >= total;
}

__device__ bool isLeafNode(int count)
{
    return count <= 32;
}

// for each node determines the split plane
__global__ void findSplitPlane(
    const float3 *__restrict__ d_pos,
    const int *__restrict__ list_x,
    const int *__restrict__ list_y,
    const int *__restrict__ list_z,
    // Node list
    int *__restrict__ split_axis, // 0, 1, 2 - 0 for x-axis, 1 for y-axis, 2 for z-axis useful for comparing node[axis] in pos array as well
    int *__restrict__ split_pos,
    int *__restrict__ bbox_min_x,
    int *__restrict__ bbox_min_y,
    int *__restrict__ bbox_min_z,
    int *__restrict__ bbox_max_x,
    int *__restrict__ bbox_max_y,
    int *__restrict__ bbox_max_z,
    const int *__restrict__ prim_head,
    const int *__restrict__ prim_count,
    int *__restrict__ children_generated,
    int num_active_nodes,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // node_idx
    int node_idx = offset + idx;
    if (idx >= num_active_nodes)
    {
        return;
    }
    int head = prim_head[node_idx];
    int count = prim_count[node_idx];

    // Threshold for being leaf node
    if (isLeafNode(count)) {
        children_generated[node_idx] = 0;
        return;
    }
    children_generated[node_idx] = 2;

    // Bounding box for primitives
    int minX = d_pos[list_x[head]].x;
    int minY = d_pos[list_y[head]].y;
    int minZ = d_pos[list_z[head]].z;
    int maxX = d_pos[list_x[head + count - 1]].x;
    int maxY = d_pos[list_y[head + count - 1]].y;
    int maxZ = d_pos[list_z[head + count - 1]].z;

    // Bounding box for node
    int bb_minX = bbox_min_x[node_idx];
    int bb_minY = bbox_min_y[node_idx];
    int bb_minZ = bbox_min_z[node_idx];
    int bb_maxX = bbox_max_x[node_idx];
    int bb_maxY = bbox_max_y[node_idx];
    int bb_maxZ = bbox_max_z[node_idx];

    // See if more than 10% empty space, if so we set the split at the corresponding boundary that contains all particles
    if (checkMoreThan10(bb_maxX, bb_minX, minX))
    {
        split_axis[node_idx] = 0;
        split_pos[node_idx] = minX;
    }
    else if (checkMoreThan10(bb_maxX, bb_minX, maxX))
    {
        split_axis[node_idx] = 0;
        split_pos[node_idx] = maxX;
    }
    else if (checkMoreThan10(bb_maxY, bb_minY, minY))
    {
        split_axis[node_idx] = 1;
        split_pos[node_idx] = minY;
    }
    else if (checkMoreThan10(bb_maxY, bb_minY, maxY))
    {
        split_axis[node_idx] = 1;
        split_pos[node_idx] = maxY;
    }
    else if (checkMoreThan10(bb_maxZ, bb_minZ, minZ))
    {
        split_axis[node_idx] = 2;
        split_pos[node_idx] = minZ;
    }
    else if (checkMoreThan10(bb_maxZ, bb_minZ, maxZ))
    {
        split_axis[node_idx] = 2;
        split_pos[node_idx] = maxZ;
    }
    else
    {
        // take spatial median along longest axis
        int diffX = bb_maxX - bb_minX;
        int diffY = bb_maxY - bb_minY;
        int diffZ = bb_maxZ - bb_minZ;

        if (diffX > diffY && diffX > diffZ)
        {
            split_axis[node_idx] = 0;
            split_pos[node_idx] = bb_minX + diffX / 2;
        }
        else if (diffY > diffZ)
        {
            split_axis[node_idx] = 1;
            split_pos[node_idx] = bb_minY + diffY / 2;
        }
        else
        {
            split_axis[node_idx] = 2;
            split_pos[node_idx] = bb_minZ + diffZ / 2;
        }
    }
}

__device__ void copyBoundingBox(
    int *__restrict__  _d_bbox_min_x,
    int *__restrict__  _d_bbox_min_y,
    int *__restrict__  _d_bbox_min_z,
    int *__restrict__  _d_bbox_max_x,
    int *__restrict__  _d_bbox_max_y,
    int *__restrict__  _d_bbox_max_z,
    int *__restrict__ child_node_idx,
    int *__restrict__ parent_node_idx
) {
    _d_bbox_min_x[child_node_idx] = _d_bbox_min_x[parent_node_idx];
    _d_bbox_max_x[child_node_idx] = _d_bbox_max_x[parent_node_idx];
    _d_bbox_min_y[child_node_idx] = _d_bbox_min_y[parent_node_idx];
    _d_bbox_max_y[child_node_idx] = _d_bbox_max_y[parent_node_idx];
    _d_bbox_min_z[child_node_idx] = _d_bbox_min_z[parent_node_idx];
    _d_bbox_max_z[child_node_idx] = _d_bbox_max_z[parent_node_idx];
}

// Called per node generates left / right child indexes
// Generate the node's bounding boxes for left / right child as well
// Called after split plane is found
__global__ void generateLeftAndRightChild(
    int *__restrict__ left_child, 
    int *__restrict__ right_child,
    int *__restrict__ split_axis, // 0, 1, 2 - 0 for x-axis, 1 for y-axis, 2 for z-axis useful for comparing node[axis] in pos array as well
    int *__restrict__ split_pos,
    bool *__restrict__ _d_children_spawn_count,
    int *__restrict__  _d_bbox_min_x,
    int *__restrict__  _d_bbox_min_y,
    int *__restrict__  _d_bbox_min_z,
    int *__restrict__  _d_bbox_max_x,
    int *__restrict__  _d_bbox_max_y,
    int *__restrict__  _d_bbox_max_z,
    int *__restrict__ _d_children_spawn_prefix, // offset to find children, size is active_nodes
    const int active_nodes,
    const int offset
) {
    int idx = blockIdx.idx * blockIdx.dim + threadIdx.x;
    int node_idx = offset + idx;
    if (idx >= active_nodes) return;
    if (_d_children_spawn_count[node_idx] == 0) return; // no children produced

    int left_node_idx = offset + _d_children_spawn_prefix[idx];
    int right_node_idx = left_node+1;
    
    left_child[node_idx] = left_node_idx;
    right_child[node_idx] = right_node_idx;

    copyBoundingBox(_d_bbox_min_x, _d_bbox_min_y, _d_bbox_min_z,  _d_bbox_max_x,  _d_bbox_max_y,  _d_bbox_max_z, left_node_idx, node_idx);
    copyBoundingBox(_d_bbox_min_x, _d_bbox_min_y, _d_bbox_min_z,  _d_bbox_max_x,  _d_bbox_max_y,  _d_bbox_max_z, right_node_idx, node_idx);

    if (split_axis[node_idx] == 0) {
        _d_bbox_max_x[left_node_idx] = split_pos[node_idx];
        _d_bbox_min_x[right_node_idx] = split_pos[node_idx];
    } else if (split_axis[node_idx] == 1) {
        _d_bbox_max_y[left_node_idx] = split_pos[node_idx];
        _d_bbox_min_y[right_node_idx] = split_pos[node_idx];
    } else {
        _d_bbox_max_z[left_node_idx] = split_pos[node_idx];
        _d_bbox_min_z[right_node_idx] = split_pos[node_idx];
    }
}


// determines if each particle goes to left node or right node
__global__ void classifyParticles(
    const float3 *__restrict__ d_pos,
    const int *__restrict__ _particle_to_node,
    int *__restrict__ split_axis, // 0, 1, 2 - 0 for x-axis, 1 for y-axis, 2 for z-axis useful for comparing node[axis] in pos array as well
    int *__restrict__ split_pos,
    bool *__restrict__ _is_left)
{
    int particle_idx = blockIdx.idx * blockIdx.dim + threadIdx.x;
    int node_idx = _particle_to_node[particle_idx];

    int axis = split_axis[node_idx];
    int pos = split_pos[node_idx];

    // assign left = 1 or right = 0
    is_left[particle_idx] = d_pos[axis] < pos;
}

// arbitrary function that takes a particle and moves it accordingly depending on if it is less than or greater than threshold
__device__ void rearrangeList(
    const int* __restrict__ _left_offset, // doesn't rearrange _left_offset[particleID] = number of left particles before this one during split
    const int* __restrict__ _is_left, // doesn't rearrange _is_left[particleID] = if should go to left or right child
    const int *__restrict__ list_dim,
    int* __restrict__ temp_dim,
    const int* __restrict__ _particle_to_node, // _particle_to_node[particleID] = node
    // Node List (- indexed by node)
    const int* __restrict__ prim_head, 
    const int* __restrict__ prim_count,
    int idx,
    const int* __restrict__ _d_left_child,
    const int* __restrict__ _d_right_child,
) {
    int particleID = list_dim[idx];
    int nodeID = _particle_to_node[particleID];
    if (_is_left[particleID] == 1) 
    {
        int leftIdx = prim_head[nodeID] + _left_offset[particleID];
        temp_dim[leftIdx] = particleID;

        // assign particles to their new node (left child) - this is repeated 3 times per particle so may be inefficient
        _particle_to_node[particleID] = _d_left_child[nodeID];
    } else
    {
        int totalLeft = _left_offset[prim_head[nodeID] + prim_count[nodeID] - 1];
        int rightElementsBefore = idx - prim_head[nodeID] - _left_offset[particleID];
        int rightIdx = prim_head[nodeID] + totalLeft + rightElementsBefore;
        temp_dim[rightIdx] = particleID;

        // assign particles to their new node (right child)
        _particle_to_node[particleID] = _d_right_child[nodeID];
    }
}

// update listX, listY, listZ
__global__ void scatterAxisListInPlace(
    // Particle List
    const int* __restrict__ _left_offset, // doesn't rearrange _left_offset[particleID] = number of left particles before this one during split
    const int* __restrict__ _is_left, // doesn't rearrange _is_left[particleID] = if should go to left or right child
    const int *__restrict__ list_x, // list_x[idx] = particleID
    const int *__restrict__ list_y,
    const int *__restrict__ list_z,
    int* __restrict__ temp_x,
    int* __restrict__ temp_y,
    int* __restrict__ temp_z,
    const int* __restrict__ _particle_to_node, // _particle_to_node[particleID] = node
    // Node List (- indexed by node)
    const int* __restrict__ prim_head, 
    const int* __restrict__ prim_count,
    const int node_offset, // where the level starts for the parent node
    const int active_nodes,

    const int* __restrict__ _d_left_child,
    const int* __restrict__ _d_right_child,
    const int* __restrict__ _d_children_spawn_count
)
{
    int idx = blockIdx.idx * blockIdx.dim + threadIdx.x; // picks up 3 possibly different particles, may want to rearrange this later on

    int particleID = list_x[idx];
    int nodeID = _particle_to_node[particleID];

    if (_d_children_spawn_count[nodeID] == 0) {
        return; // this doesn't generate any children so we don't rearrange the list
    }


    rearrangeList(_left_offset, _is_left, list_x, temp_x, _particle_to_node, prim_head, prim_count, idx, _d_left_child, _d_right_child);
    rearrangeList(_left_offset, _is_left, list_y, temp_y, _particle_to_node, prim_head, prim_count, idx, _d_left_child, _d_right_child);
    rearrangeList(_left_offset, _is_left, list_z, temp_z, _particle_to_node, prim_head, prim_count, idx, _d_left_child, _d_right_child);
}

void constructKDTreeHelper()
{
    // Goes level by level until the tree is complete
    int num_active_nodes = 1;
    int offset = 0; // node offset

    DArray<int> _d_children_spawn_prefix(num_active_nodes);

    while (num_active_nodes > 0)
    {
        int gridSize = (num_active_nodes + blockSize - 1) / blockSize;
        int blockSize = 256;
        findSplitPlane<<<gridSize, blockSize>>>(d_pos, _list_x.addr(), _list_y.addr(), _list_z.addr(), _d_split_axis.addr(), _d_split_pos.addr(), _d_bbox_min_x.addr(), _d_bbox_min_y.addr(), _d_bbox_min_z.addr(), _d_bbox_max_x.addr(), _d_bbox_max_y.addr(), _d_bbox_max_z.addr(), _d_prim_head.addr(), _d_prim_count.addr(), _d_child_spawn_count.addr(), num_active_nodes, offset);

        int childrenProducedLast = _d_child_spawn_count[offset+num_active_nodes-1]; // overwritten at end
        thrust::exclusive_scan(thrust::device, _d_child_spawn_count.addr()+offset, _d_child_spawn_count.addr() + offset + num_active_nodes, _d_children_spawn_prefix.addr()); // exclusive prefix sum

        generateLeftAndRightChild<<<gridSize, blockSize>>>(_d_left_child.addr(), _d_right_child.addr(), _d_split_axis.addr(), _d_children_spawn_count.addr(), _d_bbox_min_x.addr(), _d_bbox_min_y.addr(), _d_bbox_min_z.addr(), _d_bbox_max_x.addr(), _d_bbox_max_y.addr(), _d_bbox_max_z.addr(), _d_children_spawn_prefix.addr(), num_active_nodes, offset);
        
        int particleGridSize = (num_particles - blockSize - 1) / blockSize;
        classifyParticles<<<particleGridSize, blockSize>>>(d_pos, _particle_to_node.addr(), _d_split_axis.addr(), _d_split_pos.addr(), _is_left.addr());

        // compute exlusive prefix scan per node
        thrust::exclusive_scan_by_key(thrust::device, _particle_to_node.addr(), _particle_to_node.addr() + num_particles, _is_left.addr(), _left_offset.addr());

        // Move particles accordingly so left and right are contiguous in memory
        scatterAxisListInPlace<<<particleGridSize, blockSize>>>(_left_offset.addr(), _is_left.addr(), _list_x.addr(), _list_y.addr(), _list_z.addr(), _temp_x.addr(), _temp_y.addr(), _temp_z.addr(), _particle_to_node.addr(), _d_prim_head.addr(), _d_prim_count.addr(), offset, num_active_nodes, _d_left_child.addr(), _d_right_child.addr(), _d_child_spawn_count.addr());
        swap(_list_x, _temp_x);
        swap(_list_y, _temp_y);
        swap(_list_z, _temp_z);

        offset += num_active_nodes;
        num_active_nodes = _d_child_spawn_count[num_active_nodes-1] + childrenProducedLast; // number of children produced

        if (_d_children_spawn_prefix.capacity() < num_active_nodes) {
            _d_children_spawn_prefix.resize(2 * _d_children_spawn_prefix.capacity());
        }
    }
}

// particles is stored in device memory
void constructKDTree(const std::shared_ptr<GranularParticles> &particles)
{
    const int num_particles = particles->size();
    if (num_particles <= 0)
    {
        return;
    }

    if (num_particles >= _list_x.capacity())
    {
        _list_x.resize(num_particles);
        _list_y.resize(num_particles);
        _list_z.resize(num_particles);
        _particle_to_node.resize(num_particles); // should contain onlys 0s initially, index -> node
    }

    const float3 *d_pos = particles->get_pos_ptr();

    // Fills with incrementing numbers representing particle ids
    thrust::sequence(thrust::device, _list_x.addr(), _list_x.addr() + num_particles);
    thrust::sequence(thrust::device, _list_y.addr(), _list_y.addr() + num_particles);
    thrust::sequence(thrust::device, _list_z.addr(), _list_z.addr() + num_particles);
    thrust::fill(thrust::device, _particle_to_node.addr(), _particle_to_node.addr() + num_particles, 0); // all particles are assigned to node 0 initially

    // 2. Sort particle IDs in place along each axis using d_pos, performs merge sort which may be problematic
    thrust::sort(thrust::device, _list_x.addr(), _list_x.addr() + num_particles,
                 CompareAxisX(d_pos));
    thrust::sort(thrust::device, _list_y.addr(), _list_y.addr() + num_particles,
                 CompareAxisY(d_pos));
    thrust::sort(thrust::device, _list_z.addr(), _list_z.addr() + num_particles,
                 CompareAxisZ(d_pos));

    // Now you have to run algorithm - split the work up into kernel, kernel should be working on active list
    constructKDTreeHelper();
}
