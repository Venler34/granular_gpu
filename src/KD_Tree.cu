#include <algorithm>
#include <cfloat> // FLT_MAX for bounding box bounds and distance queries
#include <cmath>
#include <cstdint> // uint32_t, uint64_t for Morton codes or node bitmasks
#include <cstdio>
#include <functional>
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
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>     // float3, int3, etc.
#include <vector_functions.h> // make_float3, make_int3

#include "DArray.hpp"        // GPU device buffer abstraction used across the solver
#include "CUDAFunctions.cuh" // CUDA error checking macros (CUDA_CALL, CHECK_KERNEL) and block size
#include "helper_math.h"     // Vector math operations (dot, length, normalize, float3 operators)
#include "KD_Tree.cuh"

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

// Nodes at this level are forced to be leaves, so coincident particles (or boxes shrunk to float precision) can't split forever
constexpr int MAX_TREE_DEPTH = 64;
static_assert(MAX_TREE_DEPTH < KD_QUERY_STACK_SIZE, "the query stack in KD_Tree.cuh has to hold one entry per tree level");

// Checks for kernel launch errors. Compile with -DKD_TREE_DEBUG_SYNC to also synchronize after every launch,
// so errors raised while the kernel runs (e.g. out of bounds access) are reported at the kernel that caused them
#ifdef KD_TREE_DEBUG_SYNC
#define KD_CHECK_KERNEL()                                                                      \
    do                                                                                         \
    {                                                                                          \
        CHECK_KERNEL();                                                                        \
        cudaError_t kd_sync_err = cudaDeviceSynchronize();                                     \
        if (kd_sync_err)                                                                       \
            printf("CUDA Error at %s:%d (after sync):\t%s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(kd_sync_err));                                           \
    } while (0)
#else
#define KD_CHECK_KERNEL() CHECK_KERNEL()
#endif

// Reads one int from device memory on the host
static int readDeviceInt(const int *d_ptr, int index)
{
    int value;
    CUDA_CALL(cudaMemcpy(&value, d_ptr + index, sizeof(int), cudaMemcpyDeviceToHost));
    return value;
}

// Copies count elements from device memory into a new host vector
template <typename T>
static std::vector<T> copyToHost(const T *d_ptr, size_t count)
{
    std::vector<T> host(count);
    if (count > 0)
    {
        CUDA_CALL(cudaMemcpy(host.data(), d_ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
    }
    return host;
}

// true if the empty space on one side of the node is more than 10% of the node's extent along that axis
__device__ bool checkMoreThan10(float empty_space, float total)
{
    return empty_space * 10.0f > total;
}

__device__ bool isLeafNode(int count)
{
    return count <= 32;
}

// Single-thread kernel that sets up node 0
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
    const float root_padding)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    float3 bb_min, bb_max;
    if (use_custom_bounds)
    {
        bb_min = custom_min;
        bb_max = custom_max;
    }
    else
    {
        // lists are sorted per axis, so the first/last entries give the tight bounds
        bb_min = make_float3(d_pos[list_x[0]].x, d_pos[list_y[0]].y, d_pos[list_z[0]].z);
        bb_max = make_float3(d_pos[list_x[total_particles - 1]].x,
                             d_pos[list_y[total_particles - 1]].y,
                             d_pos[list_z[total_particles - 1]].z);

        float extent = fmaxf(bb_max.x - bb_min.x, fmaxf(bb_max.y - bb_min.y, bb_max.z - bb_min.z));
        float pad = fmaxf(extent * root_padding, FLT_EPSILON); // stays nonzero if all particles coincide
        bb_min -= make_float3(pad);
        bb_max += make_float3(pad);
    }

    bbox_min_x[0] = bb_min.x;
    bbox_min_y[0] = bb_min.y;
    bbox_min_z[0] = bb_min.z;
    bbox_max_x[0] = bb_max.x;
    bbox_max_y[0] = bb_max.y;
    bbox_max_z[0] = bb_max.z;

    prim_head[0] = 0;
    prim_count[0] = total_particles;
    left_child[0] = -1;
    right_child[0] = -1;
    split_axis[0] = -1;
}

// for each node determines the split plane
__global__ void findSplitPlane(
    const float3 *__restrict__ d_pos,
    const int *__restrict__ list_x,
    const int *__restrict__ list_y,
    const int *__restrict__ list_z,
    // Node list
    int *__restrict__ split_axis, // 0, 1, 2 - 0 for x-axis, 1 for y-axis, 2 for z-axis useful for comparing node[axis] in pos array as well
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
    const bool force_leaf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // node_idx
    int node_idx = offset + idx;
    if (idx >= num_active_nodes)
    {
        return;
    }
    int head = prim_head[node_idx];
    int count = prim_count[node_idx];

    // Threshold for being leaf node (or max depth reached)
    if (force_leaf || isLeafNode(count))
    {
        children_generated[node_idx] = 0;
        // no split plane, set to 0 so classifyParticles never reads uninitialized values
        split_axis[node_idx] = 0;
        split_pos[node_idx] = 0.0f;
        return;
    }
    children_generated[node_idx] = 2;

    // Bounding box for primitives
    float minX = d_pos[list_x[head]].x;
    float minY = d_pos[list_y[head]].y;
    float minZ = d_pos[list_z[head]].z;
    float maxX = d_pos[list_x[head + count - 1]].x;
    float maxY = d_pos[list_y[head + count - 1]].y;
    float maxZ = d_pos[list_z[head + count - 1]].z;

    // Bounding box for node
    float bb_minX = bbox_min_x[node_idx];
    float bb_minY = bbox_min_y[node_idx];
    float bb_minZ = bbox_min_z[node_idx];
    float bb_maxX = bbox_max_x[node_idx];
    float bb_maxY = bbox_max_y[node_idx];
    float bb_maxZ = bbox_max_z[node_idx];

    // See if more than 10% empty space, if so we set the split at the corresponding boundary that contains all particles
    // Particles exactly on the plane go to the right child (coord < split_pos goes left), so for the max side the
    // plane is placed just past the largest coordinate to keep that particle in the left child
    float diffX = bb_maxX - bb_minX;
    float diffY = bb_maxY - bb_minY;
    float diffZ = bb_maxZ - bb_minZ;

    if (checkMoreThan10(minX - bb_minX, diffX))
    {
        split_axis[node_idx] = 0;
        split_pos[node_idx] = minX;
    }
    else if (checkMoreThan10(bb_maxX - maxX, diffX))
    {
        split_axis[node_idx] = 0;
        split_pos[node_idx] = nextafterf(maxX, FLT_MAX); // this ensures all particles go to the left including the max particle
    }
    else if (checkMoreThan10(minY - bb_minY, diffY))
    {
        split_axis[node_idx] = 1;
        split_pos[node_idx] = minY;
    }
    else if (checkMoreThan10(bb_maxY - maxY, diffY))
    {
        split_axis[node_idx] = 1;
        split_pos[node_idx] = nextafterf(maxY, FLT_MAX);
    }
    else if (checkMoreThan10(minZ - bb_minZ, diffZ))
    {
        split_axis[node_idx] = 2;
        split_pos[node_idx] = minZ;
    }
    else if (checkMoreThan10(bb_maxZ - maxZ, diffZ))
    {
        split_axis[node_idx] = 2;
        split_pos[node_idx] = nextafterf(maxZ, FLT_MAX);
    }
    else
    {
        // take spatial median along longest axis
        if (diffX > diffY && diffX > diffZ)
        {
            split_axis[node_idx] = 0;
            split_pos[node_idx] = bb_minX + diffX * 0.5f;
        }
        else if (diffY > diffZ)
        {
            split_axis[node_idx] = 1;
            split_pos[node_idx] = bb_minY + diffY * 0.5f;
        }
        else
        {
            split_axis[node_idx] = 2;
            split_pos[node_idx] = bb_minZ + diffZ * 0.5f;
        }
    }
}

__device__ void copyBoundingBox(
    float *__restrict__ _d_bbox_min_x,
    float *__restrict__ _d_bbox_min_y,
    float *__restrict__ _d_bbox_min_z,
    float *__restrict__ _d_bbox_max_x,
    float *__restrict__ _d_bbox_max_y,
    float *__restrict__ _d_bbox_max_z,
    int child_node_idx,
    int parent_node_idx)
{
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
    float *__restrict__ split_pos,
    int *__restrict__ _d_children_spawn_count,
    float *__restrict__ _d_bbox_min_x,
    float *__restrict__ _d_bbox_min_y,
    float *__restrict__ _d_bbox_min_z,
    float *__restrict__ _d_bbox_max_x,
    float *__restrict__ _d_bbox_max_y,
    float *__restrict__ _d_bbox_max_z,
    int *__restrict__ _d_children_spawn_prefix, // offset to find children, size is active_nodes
    const int active_nodes,
    const int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int node_idx = offset + idx;
    if (idx >= active_nodes)
        return;
    if (_d_children_spawn_count[node_idx] == 0)
    {
        // leaf node
        left_child[node_idx] = -1;
        right_child[node_idx] = -1;
        return; // no children produced
    }

    // children of this level are stored right after the current level's nodes
    int left_node_idx = offset + active_nodes + _d_children_spawn_prefix[idx];
    int right_node_idx = left_node_idx + 1;

    left_child[node_idx] = left_node_idx;
    right_child[node_idx] = right_node_idx;

    copyBoundingBox(_d_bbox_min_x, _d_bbox_min_y, _d_bbox_min_z, _d_bbox_max_x, _d_bbox_max_y, _d_bbox_max_z, left_node_idx, node_idx);
    copyBoundingBox(_d_bbox_min_x, _d_bbox_min_y, _d_bbox_min_z, _d_bbox_max_x, _d_bbox_max_y, _d_bbox_max_z, right_node_idx, node_idx);

    if (split_axis[node_idx] == 0)
    {
        _d_bbox_max_x[left_node_idx] = split_pos[node_idx];
        _d_bbox_min_x[right_node_idx] = split_pos[node_idx];
    }
    else if (split_axis[node_idx] == 1)
    {
        _d_bbox_max_y[left_node_idx] = split_pos[node_idx];
        _d_bbox_min_y[right_node_idx] = split_pos[node_idx];
    }
    else
    {
        _d_bbox_max_z[left_node_idx] = split_pos[node_idx];
        _d_bbox_min_z[right_node_idx] = split_pos[node_idx];
    }
}

// determines if each particle goes to left node or right node
__global__ void classifyParticles(
    const float3 *__restrict__ d_pos,
    const int *__restrict__ _particle_to_node,
    const int *__restrict__ split_axis, // 0, 1, 2 - 0 for x-axis, 1 for y-axis, 2 for z-axis useful for comparing node[axis] in pos array as well
    const float *__restrict__ split_pos,
    int *__restrict__ _is_left,
    const int total_particles)
{
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle_idx >= total_particles)
        return;

    int node_idx = _particle_to_node[particle_idx];

    int axis = split_axis[node_idx];
    float pos = split_pos[node_idx];

    float3 p = d_pos[particle_idx];
    float coord = (axis == 0) ? p.x : (axis == 1) ? p.y
                                                  : p.z;

    // assign left = 1 or right = 0
    _is_left[particle_idx] = coord < pos;
}

// Moves the particles of one dimension list (list_x / list_y / list_z) so that, inside every splitting node,
// left-child particles come first and right-child particles second, keeping the sorted order (stable).
// Output goes to temp_dim, the caller swaps list_dim and temp_dim afterwards.
// Doesn't modify _particle_to_node, that is done afterwards in updateParticleToNode.
__global__ void scatterList(
    const int *__restrict__ _left_offset, // _left_offset[i] = left particles before list position i within its node
    const int *__restrict__ _is_left,     // _is_left[particleID]
    const int *__restrict__ list_dim,     // list_dim[i] = particleID
    int *__restrict__ temp_dim,
    const int *__restrict__ _particle_to_node, // _particle_to_node[particleID] = node
    const int *__restrict__ prim_head,
    const int *__restrict__ prim_count,
    const int *__restrict__ _d_children_spawn_count,
    const int total_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // position in list_dim
    if (idx >= total_particles)
        return;

    int particleID = list_dim[idx];
    int nodeID = _particle_to_node[particleID];

    if (_d_children_spawn_count[nodeID] == 0)
    {
        temp_dim[idx] = particleID; // leaf, keeps its slot
        return;
    }

    int head = prim_head[nodeID];
    int last = head + prim_count[nodeID] - 1;

    // exclusive scan leaves out the last element, so add its flag to get the total
    int totalLeft = _left_offset[last] + _is_left[list_dim[last]];
    int leftBefore = _left_offset[idx]; // left particles before this one in the node

    int dst;
    if (_is_left[particleID])
    {
        dst = head + leftBefore;
    }
    else
    {
        int rightBefore = (idx - head) - leftBefore;
        dst = head + totalLeft + rightBefore;
    }
    temp_dim[dst] = particleID;
}

// Called per node, sets the particle range (head, count) of the left / right child of every splitting node
// Called after the particles are scattered, but before the lists are swapped with the temp lists:
// _left_offset must describe the same list that is passed as list_dim
__global__ void setChildRanges(
    const int *__restrict__ _left_offset, // _left_offset[i] = left particles before list position i within its node
    const int *__restrict__ _is_left,     // _is_left[particleID]
    const int *__restrict__ list_dim,     // list_dim[i] = particleID, same list that _left_offset was computed for
    const int *__restrict__ left_child,
    const int *__restrict__ right_child,
    const int *__restrict__ _d_children_spawn_count,
    int *__restrict__ prim_head,
    int *__restrict__ prim_count,
    const int active_nodes,
    const int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int node_idx = offset + idx;
    if (idx >= active_nodes)
        return;
    if (_d_children_spawn_count[node_idx] == 0)
        return; // leaf, no children

    int head = prim_head[node_idx];
    int count = prim_count[node_idx];
    int last = head + count - 1;

    // exclusive scan leaves out the last element, so add its flag to get the total
    int totalLeft = _left_offset[last] + _is_left[list_dim[last]];

    int left_node_idx = left_child[node_idx];
    int right_node_idx = right_child[node_idx];

    prim_head[left_node_idx] = head;
    prim_count[left_node_idx] = totalLeft;
    prim_head[right_node_idx] = head + totalLeft;
    prim_count[right_node_idx] = count - totalLeft;
}

// Assigns every particle that was in a splitting node to its left or right child.
// Must run after all three lists are scattered, since the scans and scatters read _particle_to_node.
__global__ void updateParticleToNode(
    int *__restrict__ _particle_to_node, // _particle_to_node[particleID] = node
    const int *__restrict__ _is_left,    // _is_left[particleID]
    const int *__restrict__ _d_children_spawn_count,
    const int *__restrict__ _d_left_child,
    const int *__restrict__ _d_right_child,
    const int total_particles)
{
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= total_particles)
        return;

    int node_idx = _particle_to_node[particle_idx];
    if (_d_children_spawn_count[node_idx] == 0)
        return; // leaf, particle stays in its node

    _particle_to_node[particle_idx] = _is_left[particle_idx] ? _d_left_child[node_idx] : _d_right_child[node_idx];
}

// Preorder UPPASS for one level, called per node of the level
// subtree_size[node] = number of KDNode slots the subtree of node takes in the preorder tree
// A leaf is a single KDNode (it only points into the particle list), so its size is 1
// Children are always in the level below, so the levels have to be processed from the bottom up
__global__ void upPassLevel(
    const int *__restrict__ left_child,
    const int *__restrict__ right_child,
    int *__restrict__ subtree_size,
    const int level_start,
    const int level_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_nodes)
        return;
    int node_idx = level_start + idx;

    if (left_child[node_idx] == -1)
    {
        subtree_size[node_idx] = 1; // leaf
    }
    else
    {
        subtree_size[node_idx] = subtree_size[left_child[node_idx]] + subtree_size[right_child[node_idx]] + 1;
    }
}

// Preorder DOWNPASS for one level, called per node of the level
// The parent sets the final address of its children, the children read it when their level is processed
// Left child goes right after the node, right child goes after the whole left subtree
// The node itself is written as a KDNode to its own address
// Levels have to be processed from the top down, and the root's address has to be set to 0 beforehand
__global__ void downPassLevel(
    const int *__restrict__ left_child,
    const int *__restrict__ right_child,
    const int *__restrict__ split_axis,
    const float *__restrict__ split_pos,
    const float *__restrict__ bbox_min_x,
    const float *__restrict__ bbox_min_y,
    const float *__restrict__ bbox_min_z,
    const float *__restrict__ bbox_max_x,
    const float *__restrict__ bbox_max_y,
    const float *__restrict__ bbox_max_z,
    const int *__restrict__ prim_head,
    const int *__restrict__ prim_count,
    const int *__restrict__ subtree_size,
    int *__restrict__ preorder_address,
    KDNode *__restrict__ out_nodes,
    const int level_start,
    const int level_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_nodes)
        return;
    int node_idx = level_start + idx;
    int address = preorder_address[node_idx];

    KDNode node;
    node.bb_min = make_float3(bbox_min_x[node_idx], bbox_min_y[node_idx], bbox_min_z[node_idx]);
    node.bb_max = make_float3(bbox_max_x[node_idx], bbox_max_y[node_idx], bbox_max_z[node_idx]);

    if (left_child[node_idx] == -1)
    {
        node.setLeaf(prim_head[node_idx], prim_count[node_idx]);
    }
    else
    {
        int left_node_idx = left_child[node_idx];
        int right_node_idx = right_child[node_idx];

        int right_address = address + 1 + subtree_size[left_node_idx];
        preorder_address[left_node_idx] = address + 1;
        preorder_address[right_node_idx] = right_address;

        node.setInternal(split_axis[node_idx], split_pos[node_idx], right_address);
    }

    out_nodes[address] = node;
}

// Computes _left_offset for one dimension list, then rearranges that list
void KDTree::scanAndScatter(DArray<int> &list_dim, DArray<int> &temp_dim)
{
    // views of per-particle arrays in list order: element i = array[list_dim[i]]
    auto keys = thrust::make_permutation_iterator(thrust::device_pointer_cast(_particle_to_node.addr()),
                                                  thrust::device_pointer_cast(list_dim.addr()));
    auto flags = thrust::make_permutation_iterator(thrust::device_pointer_cast(_is_left.addr()),
                                                   thrust::device_pointer_cast(list_dim.addr()));

    // per-node exclusive prefix sum of left flags, _left_offset is indexed by list position
    thrust::exclusive_scan_by_key(thrust::device, keys, keys + num_particles, flags,
                                  thrust::device_pointer_cast(_left_offset.addr()));

    int blockSize = 256;
    int gridSize = (num_particles + blockSize - 1) / blockSize;
    scatterList<<<gridSize, blockSize>>>(_left_offset.addr(), _is_left.addr(), list_dim.addr(), temp_dim.addr(),
                                         _particle_to_node.addr(), _d_prim_head.addr(), _d_prim_count.addr(),
                                         _d_children_spawn_count.addr(), num_particles);
    KD_CHECK_KERNEL();
}

void KDTree::constructKDTreeHelper(const float3 *d_pos, int num_particles)
{
    // Goes level by level until the tree is complete
    int num_active_nodes = 1;
    int offset = 0; // node offset
    int depth = 0;  // tree level of the active nodes

    // start offset and number of nodes of every level, needed later for the preorder UPPASS / DOWNPASS
    _level_start.clear();
    _level_size.clear();

    DArray<int> _d_children_spawn_prefix(num_active_nodes);

    while (num_active_nodes > 0)
    {
        _level_start.push_back(offset);
        _level_size.push_back(num_active_nodes);

        const bool force_leaf = depth >= MAX_TREE_DEPTH;
        int blockSize = 256;
        int gridSize = (num_active_nodes + blockSize - 1) / blockSize;
        findSplitPlane<<<gridSize, blockSize>>>(d_pos, _list_x.addr(), _list_y.addr(), _list_z.addr(), _d_split_axis.addr(), _d_split_pos.addr(), _d_bbox_min_x.addr(), _d_bbox_min_y.addr(), _d_bbox_min_z.addr(), _d_bbox_max_x.addr(), _d_bbox_max_y.addr(), _d_bbox_max_z.addr(), _d_prim_head.addr(), _d_prim_count.addr(), _d_children_spawn_count.addr(), num_active_nodes, offset, force_leaf);
        KD_CHECK_KERNEL();

        int childrenProducedLast = readDeviceInt(_d_children_spawn_count.addr(), offset + num_active_nodes - 1);                                                                                            // overwritten at end
        thrust::exclusive_scan(thrust::device, _d_children_spawn_count.addr() + offset, _d_children_spawn_count.addr() + offset + num_active_nodes, _d_children_spawn_prefix.addr()); // exclusive prefix sum

        int childrenProduced = readDeviceInt(_d_children_spawn_prefix.addr(), num_active_nodes - 1) + childrenProducedLast; // children produced this level

        // How much memory do we need to allocate every time we work with the KD-Tree?
        // generateLeftAndRightChild writes the children right after the current level, so we need room for
        // the current level's nodes plus every child they spawn (worse case every node generates 2 child nodes)
        int nodes_needed = offset + num_active_nodes + childrenProduced;
        if (node_capacity < nodes_needed)
        {
            node_capacity = 2 * nodes_needed; // try to minimize reallocations if possible
            _d_split_axis.resizeKeepAll(node_capacity);
            _d_split_pos.resizeKeepAll(node_capacity);
            _d_left_child.resizeKeepAll(node_capacity);
            _d_right_child.resizeKeepAll(node_capacity);
            _d_children_spawn_count.resizeKeepAll(node_capacity);
            _d_bbox_min_x.resizeKeepAll(node_capacity);
            _d_bbox_min_y.resizeKeepAll(node_capacity);
            _d_bbox_min_z.resizeKeepAll(node_capacity);
            _d_bbox_max_x.resizeKeepAll(node_capacity);
            _d_bbox_max_y.resizeKeepAll(node_capacity);
            _d_bbox_max_z.resizeKeepAll(node_capacity);
            _d_prim_head.resizeKeepAll(node_capacity);
            _d_prim_count.resizeKeepAll(node_capacity);
        }

        generateLeftAndRightChild<<<gridSize, blockSize>>>(_d_left_child.addr(), _d_right_child.addr(), _d_split_axis.addr(), _d_split_pos.addr(), _d_children_spawn_count.addr(), _d_bbox_min_x.addr(), _d_bbox_min_y.addr(), _d_bbox_min_z.addr(), _d_bbox_max_x.addr(), _d_bbox_max_y.addr(), _d_bbox_max_z.addr(), _d_children_spawn_prefix.addr(), num_active_nodes, offset);
        KD_CHECK_KERNEL();

        int particleGridSize = (num_particles + blockSize - 1) / blockSize;
        classifyParticles<<<particleGridSize, blockSize>>>(d_pos, _particle_to_node.addr(), _d_split_axis.addr(), _d_split_pos.addr(), _is_left.addr(), num_particles);
        KD_CHECK_KERNEL();

        // For each list: per-node exclusive prefix scan of _is_left into _left_offset (reused), then move particles
        // so left and right are contiguous in memory. All three read _particle_to_node, so it is only updated after.
        scanAndScatter(_list_x, _temp_x);
        scanAndScatter(_list_y, _temp_y);
        scanAndScatter(_list_z, _temp_z);

        // _left_offset currently holds the z scan, so it has to be paired with list_z (not swapped yet)
        setChildRanges<<<gridSize, blockSize>>>(_left_offset.addr(), _is_left.addr(), _list_z.addr(), _d_left_child.addr(), _d_right_child.addr(), _d_children_spawn_count.addr(), _d_prim_head.addr(), _d_prim_count.addr(), num_active_nodes, offset);
        KD_CHECK_KERNEL();

        updateParticleToNode<<<particleGridSize, blockSize>>>(_particle_to_node.addr(), _is_left.addr(), _d_children_spawn_count.addr(), _d_left_child.addr(), _d_right_child.addr(), num_particles);
        KD_CHECK_KERNEL();

        std::swap(_list_x, _temp_x);
        std::swap(_list_y, _temp_y);
        std::swap(_list_z, _temp_z);

        offset += num_active_nodes;
        num_active_nodes = childrenProduced; // number of children produced
        depth++;

        if (_d_children_spawn_prefix.capacity() < num_active_nodes)
        {
            // contents are rewritten by the scan every level, so nothing needs to be preserved
            _d_children_spawn_prefix.resize(std::max(2 * _d_children_spawn_prefix.capacity(), (unsigned int)num_active_nodes));
        }
    }

    // offset is now the total number of nodes in the tree
    _num_nodes = offset;
    _num_levels = _level_start.size();

    // One entry per node for the preorder passes, contents are rewritten by those passes so nothing needs to be preserved
    if (_d_subtree_size.capacity() < _num_nodes)
    {
        _d_subtree_size.resize(_num_nodes);
        _d_preorder_address.resize(_num_nodes);
    }
}

// Runs the UPPASS over all levels, bottom-up. Launches on the same stream run in order, so no sync is needed between levels
void KDTree::upPass()
{
    const int blockSize = 256;
    for (int level = (int)_num_levels - 1; level >= 0; level--)
    {
        int level_nodes = _level_size[level];
        int gridSize = (level_nodes + blockSize - 1) / blockSize;
        upPassLevel<<<gridSize, blockSize>>>(_d_left_child.addr(), _d_right_child.addr(), _d_subtree_size.addr(), _level_start[level], level_nodes);
        KD_CHECK_KERNEL();
    }

    // every leaf has size 1, so the root's subtree has to contain every node
    int root_size = readDeviceInt(_d_subtree_size.addr(), 0);
    if (root_size != (int)_num_nodes)
    {
        printf("[KDTree upPass] root subtree size %d does not match the node count %u\n", root_size, _num_nodes);
    }
}

// Runs the DOWNPASS over all levels, top-down. Needs the subtree sizes from upPass()
void KDTree::downPass()
{
    // leaves are a single KDNode, so the preorder tree has exactly one entry per node
    if (_d_out_nodes.capacity() < _num_nodes)
    {
        _d_out_nodes.resize(_num_nodes); // rewritten completely below, nothing needs to be preserved
    }

    // the root goes to address 0
    CUDA_CALL(cudaMemset(_d_preorder_address.addr(), 0, sizeof(int)));

    const int blockSize = 256;
    for (int level = 0; level < (int)_num_levels; level++)
    {
        int level_nodes = _level_size[level];
        int gridSize = (level_nodes + blockSize - 1) / blockSize;
        downPassLevel<<<gridSize, blockSize>>>(_d_left_child.addr(), _d_right_child.addr(), _d_split_axis.addr(), _d_split_pos.addr(),
                                               _d_bbox_min_x.addr(), _d_bbox_min_y.addr(), _d_bbox_min_z.addr(),
                                               _d_bbox_max_x.addr(), _d_bbox_max_y.addr(), _d_bbox_max_z.addr(),
                                               _d_prim_head.addr(), _d_prim_count.addr(), _d_subtree_size.addr(),
                                               _d_preorder_address.addr(), _d_out_nodes.addr(), _level_start[level], level_nodes);
        KD_CHECK_KERNEL();
    }
}

// Organizes the finished tree in preorder (node, left subtree, right subtree) into _d_out_nodes
// UPPASS computes the subtree sizes bottom-up, DOWNPASS uses them to assign the addresses top-down and writes the nodes
void KDTree::buildPreorderTree()
{
    upPass();
    downPass();
}

// particles is stored in device memory
void KDTree::constructKDTree(const std::shared_ptr<GranularParticles> &particles)
{
    constructKDTree(particles->get_pos_ptr(), particles->size());
}

// d_pos is stored in device memory
void KDTree::constructKDTree(const float3 *d_pos, unsigned int count)
{
    if (count == 0)
    {
        _num_nodes = 0; // nothing to query
        return;
    }
    num_particles = count; // class member, used by the scans / kernels in the helper

    if (num_particles > _list_x.capacity())
    {
        _list_x.resize(num_particles);
        _list_y.resize(num_particles);
        _list_z.resize(num_particles);
        _temp_x.resize(num_particles);
        _temp_y.resize(num_particles);
        _temp_z.resize(num_particles);
        _particle_to_node.resize(num_particles); // should contain onlys 0s initially, index -> node
        _is_left.resize(num_particles);
        _left_offset.resize(num_particles);
    }

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

    // Set up the root node (custom bounds if provided, otherwise tight particle bounds + padding)
    initRootNode<<<1, 1>>>(d_pos, _list_x.addr(), _list_y.addr(), _list_z.addr(),
                           _d_split_axis.addr(), _d_left_child.addr(), _d_right_child.addr(),
                           _d_bbox_min_x.addr(), _d_bbox_min_y.addr(), _d_bbox_min_z.addr(),
                           _d_bbox_max_x.addr(), _d_bbox_max_y.addr(), _d_bbox_max_z.addr(),
                           _d_prim_head.addr(), _d_prim_count.addr(), num_particles,
                           _use_custom_root, _root_min, _root_max, _root_padding);
    KD_CHECK_KERNEL();

    // Now you have to run algorithm - split the work up into kernel, kernel should be working on active list
    constructKDTreeHelper(d_pos, num_particles);

    // Reorganize the finished tree in preorder
    buildPreorderTree();
}

// Squared distance from a point to an axis aligned box, 0 if the point is inside the box
static float distSqToBox(const float3 &p, const float3 &bb_min, const float3 &bb_max)
{
    float dx = fmaxf(fmaxf(bb_min.x - p.x, 0.0f), p.x - bb_max.x);
    float dy = fmaxf(fmaxf(bb_min.y - p.y, 0.0f), p.y - bb_max.y);
    float dz = fmaxf(fmaxf(bb_min.z - p.z, 0.0f), p.z - bb_max.z);
    return dx * dx + dy * dy + dz * dz;
}

// Host side radius search on the preorder tree: returns the ids of all particles whose distance to center is
// at most radius. The result is in traversal order (not sorted) and contains every particle at the center
// position, including the one the center was taken from.
// Everything the traversal reads (tree, particle order, positions) is copied to the host on every call, so this
// is meant for checking the tree, not for querying many particles. d_pos has to be the positions the tree was built from.
std::vector<int> KDTree::radiusSearch(const float3 *d_pos, float3 center, float radius) const
{
    std::vector<int> result;
    if (_num_nodes == 0 || num_particles == 0 || radius < 0.0f)
    {
        return result;
    }

    std::vector<KDNode> nodes = copyToHost(_d_out_nodes.addr(), _num_nodes);
    std::vector<int> list = copyToHost(_list_x.addr(), num_particles); // leaf particles are list[prim_start .. prim_start + count)
    std::vector<float3> pos = copyToHost(d_pos, num_particles);

    const float radius_sq = radius * radius;

    // Depth first traversal with an explicit stack of node addresses, starting at the root (address 0)
    std::vector<int> stack;
    stack.push_back(0);
    while (!stack.empty())
    {
        int address = stack.back();
        stack.pop_back();
        const KDNode &node = nodes[address];

        // Nothing inside the node can be within the radius if the node's box is further away than that.
        // For the far child of a split this also covers the split plane test: its box starts at the plane.
        if (distSqToBox(center, node.bb_min, node.bb_max) > radius_sq)
        {
            continue;
        }

        if (node.isLeaf())
        {
            int start = node.prim_start;
            for (int i = start; i < start + node.primCount(); i++)
            {
                int id = list[i];
                float3 d = make_float3(pos[id].x - center.x, pos[id].y - center.y, pos[id].z - center.z);
                if (d.x * d.x + d.y * d.y + d.z * d.z <= radius_sq)
                {
                    result.push_back(id);
                }
            }
        }
        else
        {
            stack.push_back(node.rightChild()); // right subtree
            stack.push_back(address + 1);       // left child follows its parent, popped first
        }
    }
    return result;
}

// Same as radiusSearch, with the center taken from the position of one particle
std::vector<int> KDTree::radiusSearchParticle(const float3 *d_pos, int particle_id, float radius) const
{
    if (particle_id < 0 || particle_id >= (int)num_particles)
    {
        return std::vector<int>();
    }

    float3 center;
    CUDA_CALL(cudaMemcpy(&center, d_pos + particle_id, sizeof(float3), cudaMemcpyDeviceToHost));
    return radiusSearch(d_pos, center, radius);
}

// =====================================================================================================
// TEMPORARY: debug validation of the finished tree. It copies everything to the host and checks the tree
// with plain loops, so it is slow and only meant for development. Call validateKDTree(d_pos) (BFS tree built
// by the construction) and validatePreorderTree(d_pos) (preorder tree in _d_out_nodes) right after
// constructKDTree while checking it. Once the tree is trusted, remove this whole section, the declarations
// of validateKDTree and validatePreorderTree in KD_Tree.cuh, and every call to them.
// =====================================================================================================
bool KDTree::validateKDTree(const float3 *d_pos) const
{
    const int N = (int)num_particles;
    const int M = (int)_num_nodes;
    const int leaf_threshold = 32; // same threshold as isLeafNode
    int errors = 0;
    auto fail = [&errors](const char *fmt, auto... args)
    {
        if (errors++ < 10) // only print the first few
        {
            printf("[KDTree validate] ");
            printf(fmt, args...);
            printf("\n");
        }
    };

    if (N <= 0 || M <= 0)
    {
        fail("empty tree: %d nodes, %d particles", M, N);
        return false;
    }

    // Copy everything to the host
    std::vector<float3> pos = copyToHost(d_pos, N);
    std::vector<int> list_x = copyToHost(_list_x.addr(), N);
    std::vector<int> list_y = copyToHost(_list_y.addr(), N);
    std::vector<int> list_z = copyToHost(_list_z.addr(), N);
    std::vector<int> particle_to_node = copyToHost(_particle_to_node.addr(), N);
    std::vector<int> left = copyToHost(_d_left_child.addr(), M);
    std::vector<int> right = copyToHost(_d_right_child.addr(), M);
    std::vector<int> head = copyToHost(_d_prim_head.addr(), M);
    std::vector<int> count = copyToHost(_d_prim_count.addr(), M);
    std::vector<float> bb_min_x = copyToHost(_d_bbox_min_x.addr(), M);
    std::vector<float> bb_min_y = copyToHost(_d_bbox_min_y.addr(), M);
    std::vector<float> bb_min_z = copyToHost(_d_bbox_min_z.addr(), M);
    std::vector<float> bb_max_x = copyToHost(_d_bbox_max_x.addr(), M);
    std::vector<float> bb_max_y = copyToHost(_d_bbox_max_y.addr(), M);
    std::vector<float> bb_max_z = copyToHost(_d_bbox_max_z.addr(), M);

    if (head[0] != 0 || count[0] != N)
    {
        fail("root should own all particles: head %d, count %d (expected 0, %d)", head[0], count[0], N);
    }

    std::vector<int> parent_refs(M, 0); // how many parents point at each node
    std::vector<int> leaf_of(N, -1);    // leaf that each particle ends up in (found through list_x)
    std::vector<int> leaves;
    int empty_leaves = 0, big_leaves = 0, max_leaf = 0;
    long long leaf_total = 0;

    // 1. Tree structure, and particle ranges of parents / children
    for (int node = 0; node < M; node++)
    {
        if (head[node] < 0 || count[node] < 0 || head[node] + count[node] > N)
        {
            fail("node %d has an invalid range: head %d, count %d", node, head[node], count[node]);
            continue;
        }

        if (left[node] == -1)
        {
            if (right[node] != -1)
            {
                fail("leaf %d has a right child %d", node, right[node]);
            }
            leaves.push_back(node);
            leaf_total += count[node];
            max_leaf = std::max(max_leaf, count[node]);
            if (count[node] == 0)
                empty_leaves++;
            if (count[node] > leaf_threshold)
                big_leaves++; // only happens when max depth forced a leaf
            continue;
        }

        int l = left[node], r = right[node];
        if (l <= node || l >= M || r <= node || r >= M)
        {
            fail("node %d has invalid children %d, %d (%d nodes)", node, l, r, M);
            continue;
        }
        parent_refs[l]++;
        parent_refs[r]++;

        if (count[l] + count[r] != count[node])
            fail("node %d: children hold %d + %d particles, parent holds %d", node, count[l], count[r], count[node]);
        if (head[l] != head[node])
            fail("node %d: left child head %d, expected %d", node, head[l], head[node]);
        if (head[r] != head[node] + count[l])
            fail("node %d: right child head %d, expected %d", node, head[r], head[node] + count[l]);
    }

    if (parent_refs[0] != 0)
        fail("root has %d parents", parent_refs[0]);
    for (int node = 1; node < M; node++)
    {
        if (parent_refs[node] != 1)
            fail("node %d is referenced by %d parents (expected 1)", node, parent_refs[node]);
    }

    // 2. Every particle is in exactly one leaf range, found through list_x
    for (int leaf : leaves)
    {
        if (head[leaf] < 0 || count[leaf] < 0 || head[leaf] + count[leaf] > N)
            continue; // already reported
        for (int i = head[leaf]; i < head[leaf] + count[leaf]; i++)
        {
            int id = list_x[i];
            if (id < 0 || id >= N)
                fail("list_x[%d] holds invalid particle id %d", i, id);
            else if (leaf_of[id] != -1)
                fail("particle %d is in two leaf ranges (leaf %d and %d)", id, leaf_of[id], leaf);
            else
                leaf_of[id] = leaf;
        }
    }
    if (leaf_total != N)
        fail("leaves hold %lld particles in total, expected %d", leaf_total, N);
    for (int p = 0; p < N; p++)
    {
        if (leaf_of[p] == -1)
            fail("particle %d is not in any leaf", p);
    }

    // 3. list_y and list_z hold the same particles per leaf as list_x, and all three lists are sorted per leaf
    for (int pass = 0; pass < 3; pass++)
    {
        const std::vector<int> &list = (pass == 0) ? list_x : (pass == 1) ? list_y : list_z;
        std::vector<char> seen(N, 0);
        for (int leaf : leaves)
        {
            if (head[leaf] < 0 || count[leaf] < 0 || head[leaf] + count[leaf] > N)
                continue;
            for (int i = head[leaf]; i < head[leaf] + count[leaf]; i++)
            {
                int id = list[i];
                if (id < 0 || id >= N)
                {
                    fail("list %d [%d] holds invalid particle id %d", pass, i, id);
                    continue;
                }
                if (seen[id]++)
                    fail("list %d: particle %d appears twice", pass, id);
                if (leaf_of[id] != leaf)
                    fail("list %d: particle %d at %d is in leaf %d's range but belongs to leaf %d", pass, id, i, leaf, leaf_of[id]);

                if (i > head[leaf] && list[i - 1] >= 0 && list[i - 1] < N)
                {
                    const float3 &a = pos[list[i - 1]];
                    const float3 &b = pos[id];
                    float prev = (pass == 0) ? a.x : (pass == 1) ? a.y : a.z;
                    float cur = (pass == 0) ? b.x : (pass == 1) ? b.y : b.z;
                    if (cur < prev)
                        fail("list %d not sorted inside leaf %d at position %d", pass, leaf, i);
                }
            }
        }
    }

    // 4. Every particle is inside its leaf's bounding box, and _particle_to_node points at that leaf
    for (int p = 0; p < N; p++)
    {
        int leaf = leaf_of[p];
        if (leaf < 0)
            continue;
        const float3 &q = pos[p];
        if (q.x < bb_min_x[leaf] || q.x > bb_max_x[leaf] ||
            q.y < bb_min_y[leaf] || q.y > bb_max_y[leaf] ||
            q.z < bb_min_z[leaf] || q.z > bb_max_z[leaf])
        {
            fail("particle %d (%g, %g, %g) is outside the bounding box of leaf %d", p, q.x, q.y, q.z, leaf);
        }
        if (particle_to_node[p] != leaf)
            fail("_particle_to_node[%d] = %d but it is in leaf %d", p, particle_to_node[p], leaf);
    }

    printf("[KDTree validate] %s: %d nodes, %u levels, %d leaves (%d empty, %d above %d particles, largest %d), %d errors\n",
           errors == 0 ? "PASSED" : "FAILED", M, _num_levels, (int)leaves.size(), empty_leaves, big_leaves, leaf_threshold, max_leaf, errors);
    return errors == 0;
}

// TEMPORARY, see the note above. Checks the preorder tree in _d_out_nodes by walking it from the root.
bool KDTree::validatePreorderTree(const float3 *d_pos) const
{
    const int N = (int)num_particles;
    const int M = (int)_num_nodes;
    int errors = 0;
    auto fail = [&errors](const char *fmt, auto... args)
    {
        if (errors++ < 10) // only print the first few
        {
            printf("[KDTree validate preorder] ");
            printf(fmt, args...);
            printf("\n");
        }
    };

    if (N <= 0 || M <= 0)
    {
        fail("empty tree: %d nodes, %d particles", M, N);
        return false;
    }

    std::vector<KDNode> nodes = copyToHost(_d_out_nodes.addr(), M);
    std::vector<int> list_x = copyToHost(_list_x.addr(), N); // leaf particles are list_x[prim_start .. prim_start + count)
    std::vector<float3> pos = copyToHost(d_pos, N);

    std::vector<char> visited(M, 0);
    std::vector<int> covered(N, 0); // how many leaves hold each list position
    int leaves = 0, max_depth = 0;

    auto component = [](const float3 &v, int axis)
    { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); };
    auto boxInside = [](const KDNode &inner, const KDNode &outer)
    {
        return inner.bb_min.x >= outer.bb_min.x && inner.bb_min.y >= outer.bb_min.y && inner.bb_min.z >= outer.bb_min.z &&
               inner.bb_max.x <= outer.bb_max.x && inner.bb_max.y <= outer.bb_max.y && inner.bb_max.z <= outer.bb_max.z;
    };

    // Returns the number of nodes in the subtree at address, so the right child address can be checked
    std::function<int(int, int)> visit = [&](int address, int depth) -> int
    {
        if (address < 0 || address >= M)
        {
            fail("address %d is outside the tree (%d nodes)", address, M);
            return 0;
        }
        if (visited[address]++)
        {
            fail("node at address %d is reached twice", address);
            return 0;
        }
        max_depth = std::max(max_depth, depth);
        const KDNode &node = nodes[address];

        if (node.isLeaf())
        {
            leaves++;
            int start = node.prim_start, count = node.primCount();
            if (start < 0 || count < 0 || start + count > N)
            {
                fail("leaf at address %d has an invalid particle range: start %d, count %d", address, start, count);
                return 1;
            }
            for (int i = start; i < start + count; i++)
            {
                covered[i]++;
                int id = list_x[i];
                if (id < 0 || id >= N)
                {
                    fail("list_x[%d] holds invalid particle id %d", i, id);
                    continue;
                }
                const float3 &q = pos[id];
                if (q.x < node.bb_min.x || q.x > node.bb_max.x ||
                    q.y < node.bb_min.y || q.y > node.bb_max.y ||
                    q.z < node.bb_min.z || q.z > node.bb_max.z)
                {
                    fail("particle %d is outside the bounding box of the leaf at address %d", id, address);
                }
            }
            return 1;
        }

        int left = address + 1; // the left child always follows its parent
        int right = node.rightChild();
        int axis = node.axis();
        if (left >= M || right <= left || right >= M)
        {
            fail("node at address %d has invalid children: left %d, right %d (%d nodes)", address, left, right, M);
            return 1;
        }

        const KDNode &l = nodes[left];
        const KDNode &r = nodes[right];
        if (!boxInside(l, node) || !boxInside(r, node))
            fail("children of the node at address %d are not inside its bounding box", address);
        if (component(l.bb_max, axis) != node.split_pos || component(r.bb_min, axis) != node.split_pos)
            fail("children of the node at address %d do not meet at its split plane %g", address, node.split_pos);

        int left_size = visit(left, depth + 1);
        int right_size = visit(right, depth + 1);
        if (right != address + 1 + left_size)
            fail("node at address %d: right child at %d, expected %d (left subtree has %d nodes)", address, right, address + 1 + left_size, left_size);
        return 1 + left_size + right_size;
    };

    int root_size = visit(0, 0);
    if (root_size != M)
        fail("root subtree has %d nodes, expected %d", root_size, M);
    for (int a = 0; a < M; a++)
    {
        if (!visited[a])
            fail("node at address %d is not reachable from the root", a);
    }
    for (int i = 0; i < N; i++)
    {
        if (covered[i] != 1)
            fail("list position %d is held by %d leaves (expected 1)", i, covered[i]);
    }

    printf("[KDTree validate preorder] %s: %d nodes, %d leaves, depth %d, %d errors\n",
           errors == 0 ? "PASSED" : "FAILED", M, leaves, max_depth, errors);
    return errors == 0;
}
