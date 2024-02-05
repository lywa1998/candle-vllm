#include <stdint.h>

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

template<typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
  scalar_t* __restrict__ arr,
  const scalar_t* __restrict__ cos_ptr,
  const scalar_t* __restrict__ sin_ptr,
  int rot_offset,
  int embed_dim)
{
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template<typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  scalar_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const scalar_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }
}

// Without neox
extern "C" __global__ void rotary_embedding_kernel_u8(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  uint8_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  uint8_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const uint8_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<uint8_t, false>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_u32(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  uint32_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  uint32_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const uint32_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<uint32_t, false>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_i64(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  int64_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  int64_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const int64_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<int64_t, false>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_f32(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  float* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  float* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const float* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<float, false>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_f64(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  double* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  double* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const double* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<double, false>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_f16(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  int16_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  int16_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const int16_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<int16_t, false>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_bf16(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  int16_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  int16_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const int16_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<int16_t, false>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

// With neox
extern "C" __global__ void rotary_embedding_kernel_u8_neox(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  uint8_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  uint8_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const uint8_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<uint8_t, true>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_u32_neox(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  uint32_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  uint32_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const uint32_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<uint32_t, true>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_i64_neox(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  int64_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  int64_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const int64_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<int64_t, true>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_f32_neox(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  float* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  float* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const float* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<float, true>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_f64_neox(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  double* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  double* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const double* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<double, true>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_f16_neox(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  int16_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  int16_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const int16_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<int16_t, true>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

extern "C" __global__ void rotary_embedding_kernel_bf16_neox(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  int16_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  int16_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const int16_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  rotary_embedding_kernel<int16_t, true>(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}