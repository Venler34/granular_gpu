#pragma once

// Scoped NVTX ranges for Nsight Systems. Compiles to nothing when the NVTX3
// headers (header-only, shipped with the CUDA toolkit) are not available.
#if defined(__has_include)
#if __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
#define GRANULAR_HAS_NVTX 1
#endif
#endif

#ifdef GRANULAR_HAS_NVTX
#define NVTX_CONCAT_(a, b) a##b
#define NVTX_CONCAT(a, b) NVTX_CONCAT_(a, b)
// Range lasts until the end of the enclosing scope.
#define NVTX_RANGE(name)                                                       \
  nvtx3::scoped_range NVTX_CONCAT(nvtx_range_, __LINE__) { name }
#else
#define NVTX_RANGE(name)
#endif
