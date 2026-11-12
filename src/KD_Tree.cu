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
        _particle_ids.resize(num_particles);
    }

    const float3 *d_pos = particles->get_pos_ptr();

    // Fills with incrementing numbers representing particle ids
    thrust::sequence(thrust::device, _list_x.addr(), _list_x.addr() + num_particles);
    thrust::sequence(thrust::device, _list_y.addr(), _list_y.addr() + num_particles);
    thrust::sequence(thrust::device, _list_z.addr(), _list_z.addr() + num_particles);

    // 2. Sort particle IDs in place along each axis using d_pos, performs merge sort which may be problematic
    thrust::sort(thrust::device, _list_x.addr(), _list_x.addr() + num_particles,
                 CompareAxisX(d_pos));
    thrust::sort(thrust::device, _list_y.addr(), _list_y.addr() + num_particles,
                 CompareAxisY(d_pos));
    thrust::sort(thrust::device, _list_z.addr(), _list_z.addr() + num_particles,
                 CompareAxisZ(d_pos));
}