#ifndef KD_TREE_CUH
#define KD_TREE_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <memory>
#include <vector>

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

// Preorder UPPASS for one tree level: computes how many KDNode slots the subtree of every node in the level takes
__global__ void upPassLevel(
    const int *__restrict__ left_child,
    const int *__restrict__ right_child,
    int *__restrict__ subtree_size,
    const int level_start,
    const int level_nodes);

// =========================================================================
// KD-Tree Node (final preorder tree)
// =========================================================================

// One node of the finished tree, stored in preorder: a node is followed by its whole left subtree, then its right
// subtree, so the left child of the node at address a is always at a + 1. Exactly 32 bytes and 32 byte aligned,
// which is the size of one GPU memory sector, so visiting a node costs one sector.
//
//   bytes  0..11   bb_min      bounding box of the node
//   bytes 12..23   bb_max
//   bytes 24..27   split_pos / prim_start   (union, which one is valid depends on the node type)
//   bytes 28..31   link                     (bit packed, see below)
//
// Internal and leaf nodes need different data, so they share the last two words:
//
//   internal node:  split_pos  = position of the split plane
//                   link       = right child address and split axis
//   leaf node:      prim_start = index of the first particle, the particles are list[prim_start .. prim_start + prim_count)
//                   link       = particle count and the leaf marker
//
// Bit packing of link: the lowest 2 bits are a tag, the upper 30 bits are a value.
//
//        bit 31                              bit 2   bit 1   bit 0
//        [ value (right child or count) ...............  ] [ tag (2 bits) ]
//
//   tag 0, 1, 2 = internal node, the tag is the split axis (0 = x, 1 = y, 2 = z), value = right child address
//   tag 3       = leaf node,                                                    value = particle count
//
//   encode:  link = (value << 2) | tag
//   decode:  tag  = link & 3        value = link >> 2
//
//   Example internal node, right child at address 5, split along y:  link = (5 << 2) | 1 = 21 = 0b10101
//       21 & 3 = 1 -> internal node, split axis y          21 >> 2 = 5 -> right child address
//   Example leaf node holding 7 particles:                          link = (7 << 2) | 3 = 31 = 0b11111
//       31 & 3 = 3 -> leaf                                 31 >> 2 = 7 -> particle count
//
// The value has 30 bits (kept non-negative so the shift is safe), so addresses and counts must stay below 2^29.
struct alignas(32) KDNode
{
    float3 bb_min;
    float3 bb_max;
    union
    {
        float split_pos; // internal node
        int prim_start;  // leaf node
    };
    int link; // tag in the lowest 2 bits, value in the rest, see above

    static constexpr int TAG_MASK = 3;
    static constexpr int LEAF_TAG = 3;

    __host__ __device__ bool isLeaf() const { return (link & TAG_MASK) == LEAF_TAG; }
    __host__ __device__ int axis() const { return link & TAG_MASK; }   // only valid for internal nodes
    __host__ __device__ int rightChild() const { return link >> 2; }   // only valid for internal nodes
    __host__ __device__ int primCount() const { return link >> 2; }    // only valid for leaf nodes

    __host__ __device__ void setInternal(int split_axis, float split, int right_child)
    {
        split_pos = split;
        link = (right_child << 2) | split_axis;
    }
    __host__ __device__ void setLeaf(int start, int count)
    {
        prim_start = start;
        link = (count << 2) | LEAF_TAG;
    }
};
static_assert(sizeof(KDNode) == 32, "KDNode must be exactly 32 bytes");

// =========================================================================
// Device side radius query on the finished preorder tree
// =========================================================================

// Size of the traversal stack. The stack holds at most one pending right child per tree level, so it has to be larger
// than MAX_TREE_DEPTH in KD_Tree.cu (checked there with a static_assert).
constexpr int KD_QUERY_STACK_SIZE = 72;

// Squared distance from a point to an axis aligned box, 0 if the point is inside the box
__host__ __device__ inline float kdDistSqToBox(const float3 &p, const float3 &bb_min, const float3 &bb_max)
{
    const float dx = fmaxf(fmaxf(bb_min.x - p.x, 0.0f), p.x - bb_max.x);
    const float dy = fmaxf(fmaxf(bb_min.y - p.y, 0.0f), p.y - bb_max.y);
    const float dz = fmaxf(fmaxf(bb_min.z - p.z, 0.0f), p.z - bb_max.z);
    return dx * dx + dy * dy + dz * dz;
}

// What a kernel needs to query the finished tree, a plain struct so it can be passed to kernels by value.
//   nodes         the preorder tree, node 0 is the root
//   particle_ids  the particle ids in leaf order, a leaf holds particle_ids[prim_start .. prim_start + count)
struct KDTreeView
{
    const KDNode *nodes;
    const int *particle_ids;
    int num_nodes;
};

// Calls visit(particle_id) for every particle stored in a leaf whose bounding box is within radius of center.
// This is a superset of the particles within the radius: the leaf boxes are the ones the tree was built from, the
// particles themselves are not tested. The caller does that test, with the positions it wants (they may have moved
// since the tree was built, see Solver::project). Every particle that was built into the tree at a position within
// radius of center is guaranteed to be visited.
// One thread runs one query, depth first with an explicit stack: the left child is the next node in memory, the right
// child is pushed and visited after the whole left subtree.
template <typename Visitor>
__device__ inline void kdRadiusQuery(const KDTreeView &tree, const float3 center, const float radius, Visitor &&visit)
{
    if (tree.num_nodes == 0)
    {
        return;
    }

    const float radius_sq = radius * radius;
    int stack[KD_QUERY_STACK_SIZE];
    int top = 0;
    int address = 0; // root

    while (true)
    {
        const KDNode node = tree.nodes[address];

        if (kdDistSqToBox(center, node.bb_min, node.bb_max) <= radius_sq)
        {
            if (!node.isLeaf())
            {
                stack[top++] = node.rightChild();
                address = address + 1; // left child
                continue;
            }

            const int start = node.prim_start;
            const int count = node.primCount();
            for (int k = 0; k < count; k++)
            {
                visit(tree.particle_ids[start + k]);
            }
        }

        if (top == 0)
        {
            return;
        }
        address = stack[--top];
    }
}

// Preorder DOWNPASS for one tree level: gives the children of every node in the level their final address and
// writes the node itself as a KDNode to its own address
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
    const int level_nodes);

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
                                // Final preorder tree
                                _d_out_nodes(0)
    {
    }
    ~KDTree() = default;

    void constructKDTree(const std::shared_ptr<GranularParticles> &particles);
    // Same, from any device array of positions (the particle id of a point is its index in the array)
    void constructKDTree(const float3 *d_pos, unsigned int count);

    // Handle for querying the finished tree on the device with kdRadiusQuery. Valid until the next constructKDTree.
    KDTreeView view() const { return KDTreeView{_d_out_nodes.addr(), _list_x.addr(), (int)_num_nodes}; }

    // Host side radius search on the finished preorder tree, returns the ids of all particles within radius of
    // center (or of the given particle, including that particle itself). d_pos are the positions the tree was built from.
    // Copies the tree to the host on every call, so it is slow and meant for checking the tree.
    std::vector<int> radiusSearch(const float3 *d_pos, float3 center, float radius) const;
    std::vector<int> radiusSearchParticle(const float3 *d_pos, int particle_id, float radius) const;

    // TEMPORARY debug check of the finished tree (slow, copies everything to the host), see KD_Tree.cu.
    // Remove together with its definition once the construction is validated.
    bool validateKDTree(const float3 *d_pos) const;
    // TEMPORARY, same as above but checks the preorder tree in _d_out_nodes (call after constructKDTree)
    bool validatePreorderTree(const float3 *d_pos) const;

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
    void upPass();   // fills _d_subtree_size, bottom-up over the levels
    void downPass(); // fills _d_preorder_address and _d_out_nodes, top-down over the levels
    void buildPreorderTree(); // organizes the finished tree in preorder: upPass() then downPass()

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
    unsigned int _num_levels = 0; // number of tree levels, _level_start / _level_size have this many entries
    // Per level info for the UPASS and DOWNPASS, which launch one kernel per level
    std::vector<int> _level_start; // index of the first node of every level
    std::vector<int> _level_size;  // number of nodes in every level
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

    // Once Tree is fully built, will store in preorder traversal, one KDNode per node (_num_nodes entries)
    DArray<KDNode> _d_out_nodes;
};

#endif // KD_TREE_CUH