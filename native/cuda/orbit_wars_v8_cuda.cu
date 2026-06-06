#include "orbit_wars_v8_cuda.h"

#include <cutlass/arch/mma.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int CUDA_STATUS_OK = 0;
constexpr int CUDA_STATUS_BAD_ARGUMENT = 1;
constexpr int CUDA_STATUS_CUDA_ERROR = 3;
constexpr int TOKEN_FEATURES = 14;
constexpr int D_MODEL = 128;
constexpr int HEADS = 4;
constexpr int HEAD_DIM = 32;
constexpr int FF_DIM = 512;
constexpr int ENCODER_LAYERS = 4;
constexpr int DECODER_LAYERS = 2;
constexpr int ACTION_SLOTS = 8;
constexpr int PLANETS = 64;
constexpr int AMOUNTS = 16;
constexpr int MAX_TOKEN_FLEETS = 640;
constexpr int RESIDENT_TOKEN_COUNT = 1 + PLANETS + MAX_TOKEN_FLEETS;
constexpr float LAYER_NORM_EPS = 1.0e-5f;
constexpr float NEG_INF = -3.402823466e38f;

using CutlassLinearGemm = cutlass::gemm::device::Gemm<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>>;

using CutlassLinearGemmSimt = cutlass::gemm::device::Gemm<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>>;

using CutlassGroupedLinearOutputOp = cutlass::epilogue::thread::LinearCombination<
    float,
    1,
    float,
    float>;

using CutlassGroupedLinearKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    CutlassGroupedLinearOutputOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute,
    cutlass::arch::OpMultiplyAddFastF32>::GemmKernel;

using CutlassGroupedLinearGemm = cutlass::gemm::device::GemmGrouped<CutlassGroupedLinearKernel>;

OrbitWarsV8CudaStatus ok() {
  return {CUDA_STATUS_OK, "ok"};
}

OrbitWarsV8CudaStatus bad_argument(const char* message) {
  return {CUDA_STATUS_BAD_ARGUMENT, message};
}

OrbitWarsV8CudaStatus cuda_error(cudaError_t error) {
  return {CUDA_STATUS_CUDA_ERROR, cudaGetErrorString(error)};
}

bool valid_shape(const OrbitWarsV8CudaShape& shape) {
  return shape.batch_count > 0 &&
         shape.token_count > 0 &&
         shape.token_features == TOKEN_FEATURES &&
         shape.d_model == D_MODEL &&
         shape.head_count == HEADS &&
         shape.encoder_layer_count == ENCODER_LAYERS &&
         shape.decoder_layer_count == DECODER_LAYERS &&
         shape.action_slot_count == ACTION_SLOTS &&
         shape.planet_count == PLANETS &&
         shape.amount_class_count == AMOUNTS;
}

void debug_stage(const char* stage) {
  if (std::getenv("ORBIT_WARS_V8_CUDA_DEBUG")) {
    std::fprintf(stderr, "cuda_v8_stage=%s\n", stage);
    std::fflush(stderr);
  }
}

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t len = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (ptr) {
      cudaFree(ptr);
    }
  }

  cudaError_t allocate(size_t count) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      len = 0;
    }
    len = count;
    return cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * count);
  }

  cudaError_t ensure(size_t count) {
    if (count <= len) {
      return cudaSuccess;
    }
    return allocate(count);
  }

  cudaError_t copy_from_host(const T* source, size_t count) {
    if (count > len) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpy(ptr, source, sizeof(T) * count, cudaMemcpyHostToDevice);
  }

  cudaError_t copy_to_host(T* target, size_t count) const {
    if (count > len) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpy(target, ptr, sizeof(T) * count, cudaMemcpyDeviceToHost);
  }
};

struct DeviceTensor {
  const float* ptr = nullptr;
  size_t len = 0;
};

using TensorMap = std::unordered_map<std::string, DeviceTensor>;

struct EncoderLayerWeights {
  const float* self_attn_in_proj_weight = nullptr;
  const float* self_attn_in_proj_bias = nullptr;
  const float* self_attn_out_proj_weight = nullptr;
  const float* self_attn_out_proj_bias = nullptr;
  const float* linear1_weight = nullptr;
  const float* linear1_bias = nullptr;
  const float* linear2_weight = nullptr;
  const float* linear2_bias = nullptr;
  const float* norm1_weight = nullptr;
  const float* norm1_bias = nullptr;
  const float* norm2_weight = nullptr;
  const float* norm2_bias = nullptr;
};

struct DecoderLayerWeights {
  const float* self_attn_in_proj_weight = nullptr;
  const float* self_attn_in_proj_bias = nullptr;
  const float* self_attn_out_proj_weight = nullptr;
  const float* self_attn_out_proj_bias = nullptr;
  const float* cross_attn_in_proj_weight = nullptr;
  const float* cross_attn_in_proj_bias = nullptr;
  const float* cross_attn_out_proj_weight = nullptr;
  const float* cross_attn_out_proj_bias = nullptr;
  const float* linear1_weight = nullptr;
  const float* linear1_bias = nullptr;
  const float* linear2_weight = nullptr;
  const float* linear2_bias = nullptr;
  const float* norm1_weight = nullptr;
  const float* norm1_bias = nullptr;
  const float* norm2_weight = nullptr;
  const float* norm2_bias = nullptr;
  const float* norm3_weight = nullptr;
  const float* norm3_bias = nullptr;
};

struct PackedModelWeights {
  const float* token_projection_weight = nullptr;
  const float* token_projection_bias = nullptr;
  const float* type_embedding = nullptr;
  const float* owner_embedding = nullptr;
  const float* slot_queries = nullptr;
  EncoderLayerWeights encoder[ENCODER_LAYERS];
  DecoderLayerWeights decoder[DECODER_LAYERS];
  const float* fire_head_weight = nullptr;
  const float* fire_head_bias = nullptr;
  const float* source_head_weight = nullptr;
  const float* source_head_bias = nullptr;
  const float* target_head_weight = nullptr;
  const float* target_head_bias = nullptr;
  const float* amount_head_weight = nullptr;
  const float* amount_head_bias = nullptr;
  bool valid = false;
};

struct ForwardWorkspace {
  DeviceBuffer<float> d_tokens;
  DeviceBuffer<long long> d_token_type_ids;
  DeviceBuffer<long long> d_owner_ids;
  DeviceBuffer<unsigned char> d_padding_mask;
  DeviceBuffer<unsigned char> d_planet_mask;
  DeviceBuffer<float> hidden;
  DeviceBuffer<float> norm_token;
  DeviceBuffer<float> token_attn;
  DeviceBuffer<float> token_q;
  DeviceBuffer<float> token_k;
  DeviceBuffer<float> token_v;
  DeviceBuffer<float> token_qkv;
  DeviceBuffer<float> token_context;
  DeviceBuffer<float> token_ff_mid;
  DeviceBuffer<float> token_ff_out;
  DeviceBuffer<float> slots;
  DeviceBuffer<float> norm_slot;
  DeviceBuffer<float> slot_attn;
  DeviceBuffer<float> slot_q;
  DeviceBuffer<float> slot_k;
  DeviceBuffer<float> slot_v;
  DeviceBuffer<float> slot_qkv;
  DeviceBuffer<float> slot_context;
  DeviceBuffer<float> slot_ff_mid;
  DeviceBuffer<float> slot_ff_out;
  DeviceBuffer<float> d_fire;
  DeviceBuffer<float> d_source;
  DeviceBuffer<float> d_target;
  DeviceBuffer<float> d_amount;
  DeviceBuffer<int> request_game_indices;
  DeviceBuffer<int> request_player_ids;
  DeviceBuffer<OrbitWarsCudaActionLabel> d_action_labels;
  size_t last_request_count = 0;
  size_t last_token_count = RESIDENT_TOKEN_COUNT;
};

struct CudaModelState {
  std::vector<std::unique_ptr<DeviceBuffer<float>>> owned_tensors;
  TensorMap tensor_map;
  PackedModelWeights weights;
  ForwardWorkspace workspace;
};

struct CudaSimState {
  OrbitWarsCudaSimConfig config{};
  DeviceBuffer<OrbitWarsCudaPlanet> planets;
  DeviceBuffer<OrbitWarsCudaPlanet> initial_planets;
  DeviceBuffer<OrbitWarsCudaFleet> fleets;
  DeviceBuffer<int> next_fleet_ids;
  DeviceBuffer<float> angular_velocities;
  DeviceBuffer<int> fleet_counts;
  DeviceBuffer<OrbitWarsCudaAction> actions;
  DeviceBuffer<int> action_counts;
  DeviceBuffer<OrbitWarsCudaSimStats> stats;
  DeviceBuffer<OrbitWarsCudaSimStats> cumulative_stats;
  DeviceBuffer<OrbitWarsCudaGameStatus> statuses;
  DeviceBuffer<int> arrivals;
  DeviceBuffer<float> old_x;
  DeviceBuffer<float> old_y;
  DeviceBuffer<float> new_x;
  DeviceBuffer<float> new_y;
  DeviceBuffer<int> request_game_indices;
  DeviceBuffer<int> request_player_ids;
  DeviceBuffer<cutlass::gemm::GemmCoord> grouped_problem_sizes;
  DeviceBuffer<float*> grouped_ptr_a;
  DeviceBuffer<float*> grouped_ptr_b;
  DeviceBuffer<float*> grouped_ptr_c;
  DeviceBuffer<float*> grouped_ptr_d;
  DeviceBuffer<int64_t> grouped_lda;
  DeviceBuffer<int64_t> grouped_ldb;
  DeviceBuffer<int64_t> grouped_ldc;
  DeviceBuffer<int64_t> grouped_ldd;
  DeviceBuffer<uint8_t> grouped_workspace;
  size_t request_plan_count = 0;
  int resident_token_count = RESIDENT_TOKEN_COUNT;
};

struct CudaSimKernelState {
  OrbitWarsCudaSimConfig config;
  OrbitWarsCudaPlanet* planets;
  OrbitWarsCudaPlanet* initial_planets;
  OrbitWarsCudaFleet* fleets;
  int* next_fleet_ids;
  float* angular_velocities;
  int* fleet_counts;
  OrbitWarsCudaAction* actions;
  int* action_counts;
  OrbitWarsCudaSimStats* stats;
  OrbitWarsCudaSimStats* cumulative_stats;
  OrbitWarsCudaGameStatus* statuses;
  int* arrivals;
  float* old_x;
  float* old_y;
  float* new_x;
  float* new_y;
};

CudaSimKernelState sim_kernel_state(CudaSimState* state) {
  return CudaSimKernelState{
      state->config,
      state->planets.ptr,
      state->initial_planets.ptr,
      state->fleets.ptr,
      state->next_fleet_ids.ptr,
      state->angular_velocities.ptr,
      state->fleet_counts.ptr,
      state->actions.ptr,
      state->action_counts.ptr,
      state->stats.ptr,
      state->cumulative_stats.ptr,
      state->statuses.ptr,
      state->arrivals.ptr,
      state->old_x.ptr,
      state->old_y.ptr,
      state->new_x.ptr,
      state->new_y.ptr};
}

const float* tensor_ptr(const TensorMap& map, const char* name) {
  auto found = map.find(name);
  return found == map.end() ? nullptr : found->second.ptr;
}

bool build_packed_weights(const TensorMap& map, PackedModelWeights* weights) {
  if (!weights) return false;
  PackedModelWeights packed{};
  packed.token_projection_weight = tensor_ptr(map, "token_projection.weight");
  packed.token_projection_bias = tensor_ptr(map, "token_projection.bias");
  packed.type_embedding = tensor_ptr(map, "type_embedding.weight");
  packed.owner_embedding = tensor_ptr(map, "owner_embedding.weight");
  packed.slot_queries = tensor_ptr(map, "slot_queries");
  packed.fire_head_weight = tensor_ptr(map, "fire_head.weight");
  packed.fire_head_bias = tensor_ptr(map, "fire_head.bias");
  packed.source_head_weight = tensor_ptr(map, "source_head.weight");
  packed.source_head_bias = tensor_ptr(map, "source_head.bias");
  packed.target_head_weight = tensor_ptr(map, "target_head.weight");
  packed.target_head_bias = tensor_ptr(map, "target_head.bias");
  packed.amount_head_weight = tensor_ptr(map, "amount_head.weight");
  packed.amount_head_bias = tensor_ptr(map, "amount_head.bias");
  for (int layer = 0; layer < ENCODER_LAYERS; ++layer) {
    std::string prefix = "encoder.layers." + std::to_string(layer);
    EncoderLayerWeights& layer_weights = packed.encoder[layer];
    layer_weights.self_attn_in_proj_weight = tensor_ptr(map, (prefix + ".self_attn.in_proj_weight").c_str());
    layer_weights.self_attn_in_proj_bias = tensor_ptr(map, (prefix + ".self_attn.in_proj_bias").c_str());
    layer_weights.self_attn_out_proj_weight = tensor_ptr(map, (prefix + ".self_attn.out_proj.weight").c_str());
    layer_weights.self_attn_out_proj_bias = tensor_ptr(map, (prefix + ".self_attn.out_proj.bias").c_str());
    layer_weights.linear1_weight = tensor_ptr(map, (prefix + ".linear1.weight").c_str());
    layer_weights.linear1_bias = tensor_ptr(map, (prefix + ".linear1.bias").c_str());
    layer_weights.linear2_weight = tensor_ptr(map, (prefix + ".linear2.weight").c_str());
    layer_weights.linear2_bias = tensor_ptr(map, (prefix + ".linear2.bias").c_str());
    layer_weights.norm1_weight = tensor_ptr(map, (prefix + ".norm1.weight").c_str());
    layer_weights.norm1_bias = tensor_ptr(map, (prefix + ".norm1.bias").c_str());
    layer_weights.norm2_weight = tensor_ptr(map, (prefix + ".norm2.weight").c_str());
    layer_weights.norm2_bias = tensor_ptr(map, (prefix + ".norm2.bias").c_str());
  }
  for (int layer = 0; layer < DECODER_LAYERS; ++layer) {
    std::string prefix = "decoder.layers." + std::to_string(layer);
    DecoderLayerWeights& layer_weights = packed.decoder[layer];
    layer_weights.self_attn_in_proj_weight = tensor_ptr(map, (prefix + ".self_attn.in_proj_weight").c_str());
    layer_weights.self_attn_in_proj_bias = tensor_ptr(map, (prefix + ".self_attn.in_proj_bias").c_str());
    layer_weights.self_attn_out_proj_weight = tensor_ptr(map, (prefix + ".self_attn.out_proj.weight").c_str());
    layer_weights.self_attn_out_proj_bias = tensor_ptr(map, (prefix + ".self_attn.out_proj.bias").c_str());
    layer_weights.cross_attn_in_proj_weight = tensor_ptr(map, (prefix + ".multihead_attn.in_proj_weight").c_str());
    layer_weights.cross_attn_in_proj_bias = tensor_ptr(map, (prefix + ".multihead_attn.in_proj_bias").c_str());
    layer_weights.cross_attn_out_proj_weight = tensor_ptr(map, (prefix + ".multihead_attn.out_proj.weight").c_str());
    layer_weights.cross_attn_out_proj_bias = tensor_ptr(map, (prefix + ".multihead_attn.out_proj.bias").c_str());
    layer_weights.linear1_weight = tensor_ptr(map, (prefix + ".linear1.weight").c_str());
    layer_weights.linear1_bias = tensor_ptr(map, (prefix + ".linear1.bias").c_str());
    layer_weights.linear2_weight = tensor_ptr(map, (prefix + ".linear2.weight").c_str());
    layer_weights.linear2_bias = tensor_ptr(map, (prefix + ".linear2.bias").c_str());
    layer_weights.norm1_weight = tensor_ptr(map, (prefix + ".norm1.weight").c_str());
    layer_weights.norm1_bias = tensor_ptr(map, (prefix + ".norm1.bias").c_str());
    layer_weights.norm2_weight = tensor_ptr(map, (prefix + ".norm2.weight").c_str());
    layer_weights.norm2_bias = tensor_ptr(map, (prefix + ".norm2.bias").c_str());
    layer_weights.norm3_weight = tensor_ptr(map, (prefix + ".norm3.weight").c_str());
    layer_weights.norm3_bias = tensor_ptr(map, (prefix + ".norm3.bias").c_str());
  }
  const bool ok =
      packed.token_projection_weight && packed.token_projection_bias && packed.type_embedding &&
      packed.owner_embedding && packed.slot_queries && packed.fire_head_weight && packed.fire_head_bias &&
      packed.source_head_weight && packed.source_head_bias && packed.target_head_weight &&
      packed.target_head_bias && packed.amount_head_weight && packed.amount_head_bias;
  if (!ok) return false;
  for (int layer = 0; layer < ENCODER_LAYERS; ++layer) {
    const EncoderLayerWeights& w = packed.encoder[layer];
    if (!w.self_attn_in_proj_weight || !w.self_attn_in_proj_bias ||
        !w.self_attn_out_proj_weight || !w.self_attn_out_proj_bias ||
        !w.linear1_weight || !w.linear1_bias || !w.linear2_weight || !w.linear2_bias ||
        !w.norm1_weight || !w.norm1_bias || !w.norm2_weight || !w.norm2_bias) {
      return false;
    }
  }
  for (int layer = 0; layer < DECODER_LAYERS; ++layer) {
    const DecoderLayerWeights& w = packed.decoder[layer];
    if (!w.self_attn_in_proj_weight || !w.self_attn_in_proj_bias ||
        !w.self_attn_out_proj_weight || !w.self_attn_out_proj_bias ||
        !w.cross_attn_in_proj_weight || !w.cross_attn_in_proj_bias ||
        !w.cross_attn_out_proj_weight || !w.cross_attn_out_proj_bias ||
        !w.linear1_weight || !w.linear1_bias || !w.linear2_weight || !w.linear2_bias ||
        !w.norm1_weight || !w.norm1_bias || !w.norm2_weight || !w.norm2_bias ||
        !w.norm3_weight || !w.norm3_bias) {
      return false;
    }
  }
  packed.valid = true;
  *weights = packed;
  return true;
}

__device__ float gelu_device(float value) {
  return 0.5f * value *
         (1.0f + tanhf(0.7978846f * (value + 0.044715f * value * value * value)));
}

__global__ void token_projection_kernel(
    const float* tokens,
    const long long* token_type_ids,
    const long long* owner_ids,
    const float* projection_weight,
    const float* projection_bias,
    const float* type_embedding,
    const float* owner_embedding,
    float* hidden,
    int batch_count,
    int token_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * token_count * D_MODEL;
  if (index >= total) return;
  int dim = index % D_MODEL;
  int row_index = index / D_MODEL;
  int token_offset = row_index * TOKEN_FEATURES;
  float value = projection_bias[dim];
  for (int feature = 0; feature < TOKEN_FEATURES; ++feature) {
    value += projection_weight[dim * TOKEN_FEATURES + feature] * tokens[token_offset + feature];
  }
  int token_type = static_cast<int>(token_type_ids[row_index]);
  int owner = static_cast<int>(owner_ids[row_index]);
  value += type_embedding[token_type * D_MODEL + dim];
  value += owner_embedding[owner * D_MODEL + dim];
  hidden[index] = value;
}

__global__ void copy_slot_queries_kernel(
    const float* slot_queries,
    float* slots,
    int batch_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * ACTION_SLOTS * D_MODEL;
  if (index >= total) return;
  int slot_dim = index % (ACTION_SLOTS * D_MODEL);
  slots[index] = slot_queries[slot_dim];
}

__global__ void layer_norm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const float* in = input + row * D_MODEL;
  float mean = 0.0f;
  for (int dim = 0; dim < D_MODEL; ++dim) {
    mean += in[dim];
  }
  mean /= static_cast<float>(D_MODEL);
  float variance = 0.0f;
  for (int dim = 0; dim < D_MODEL; ++dim) {
    float delta = in[dim] - mean;
    variance += delta * delta;
  }
  variance /= static_cast<float>(D_MODEL);
  float scale = rsqrtf(variance + LAYER_NORM_EPS);
  float* out = output + row * D_MODEL;
  for (int dim = 0; dim < D_MODEL; ++dim) {
    out[dim] = (in[dim] - mean) * scale * weight[dim] + bias[dim];
  }
}

__global__ void add_linear_bias_kernel(float* output, const float* bias, int rows, int out_features) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * out_features;
  if (index >= total) return;
  output[index] += bias[index % out_features];
}

__global__ void split_qkv_kernel(const float* qkv, float* q, float* k, float* v, int rows) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * D_MODEL;
  if (index >= total) return;
  int row = index / D_MODEL;
  int dim = index % D_MODEL;
  const float* source = qkv + row * D_MODEL * 3 + dim;
  q[index] = source[0];
  k[index] = source[D_MODEL];
  v[index] = source[D_MODEL * 2];
}

__global__ void split_qkv_bias_kernel(const float* qkv, const float* bias, float* q, float* k, float* v, int rows) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * D_MODEL;
  if (index >= total) return;
  int row = index / D_MODEL;
  int dim = index % D_MODEL;
  const float* source = qkv + row * D_MODEL * 3 + dim;
  q[index] = source[0] + bias[dim];
  k[index] = source[D_MODEL] + bias[D_MODEL + dim];
  v[index] = source[D_MODEL * 2] + bias[D_MODEL * 2 + dim];
}

__global__ void split_kv_kernel(const float* kv, float* k, float* v, int rows) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * D_MODEL;
  if (index >= total) return;
  int row = index / D_MODEL;
  int dim = index % D_MODEL;
  const float* source = kv + row * D_MODEL * 2 + dim;
  k[index] = source[0];
  v[index] = source[D_MODEL];
}

__global__ void split_kv_bias_kernel(const float* kv, const float* bias, float* k, float* v, int rows) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * D_MODEL;
  if (index >= total) return;
  int row = index / D_MODEL;
  int dim = index % D_MODEL;
  const float* source = kv + row * D_MODEL * 2 + dim;
  k[index] = source[0] + bias[dim];
  v[index] = source[D_MODEL] + bias[D_MODEL + dim];
}

__global__ void gelu_kernel(float* values, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] = gelu_device(values[index]);
  }
}

__global__ void add_kernel(float* left, const float* right, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    left[index] += right[index];
  }
}

__global__ void add_bias_gelu_kernel(float* values, const float* bias, int rows, int features) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * features;
  if (index >= total) return;
  values[index] = gelu_device(values[index] + bias[index % features]);
}

__global__ void add_bias_residual_kernel(float* residual, const float* update, const float* bias, int rows, int features) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * features;
  if (index >= total) return;
  residual[index] += update[index] + bias[index % features];
}

__global__ void attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    const unsigned char* key_padding_mask,
    float* context,
    int batch_count,
    int query_count,
    int key_count) {
  int task = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * query_count * HEADS;
  if (task >= total) return;
  int head = task % HEADS;
  int query = (task / HEADS) % query_count;
  int batch = task / (HEADS * query_count);
  int head_offset = head * HEAD_DIM;
  const float* q_row = q + (batch * query_count + query) * D_MODEL + head_offset;
  float max_score = NEG_INF;
  for (int source = 0; source < key_count; ++source) {
    if (key_padding_mask && key_padding_mask[batch * key_count + source]) {
      continue;
    }
    const float* k_row = k + (batch * key_count + source) * D_MODEL + head_offset;
    float score = 0.0f;
    for (int dim = 0; dim < HEAD_DIM; ++dim) {
      score += q_row[dim] * k_row[dim];
    }
    score /= sqrtf(static_cast<float>(HEAD_DIM));
    max_score = fmaxf(max_score, score);
  }
  float* context_row = context + (batch * query_count + query) * D_MODEL + head_offset;
  for (int dim = 0; dim < HEAD_DIM; ++dim) {
    context_row[dim] = 0.0f;
  }
  if (!isfinite(max_score)) {
    return;
  }
  float denominator = 0.0f;
  for (int source = 0; source < key_count; ++source) {
    if (key_padding_mask && key_padding_mask[batch * key_count + source]) {
      continue;
    }
    const float* k_row = k + (batch * key_count + source) * D_MODEL + head_offset;
    float score = 0.0f;
    for (int dim = 0; dim < HEAD_DIM; ++dim) {
      score += q_row[dim] * k_row[dim];
    }
    score = expf(score / sqrtf(static_cast<float>(HEAD_DIM)) - max_score);
    denominator += score;
  }
  if (denominator <= 0.0f || !isfinite(denominator)) {
    return;
  }
  for (int source = 0; source < key_count; ++source) {
    if (key_padding_mask && key_padding_mask[batch * key_count + source]) {
      continue;
    }
    const float* k_row = k + (batch * key_count + source) * D_MODEL + head_offset;
    const float* v_row = v + (batch * key_count + source) * D_MODEL + head_offset;
    float score = 0.0f;
    for (int dim = 0; dim < HEAD_DIM; ++dim) {
      score += q_row[dim] * k_row[dim];
    }
    float attention = expf(score / sqrtf(static_cast<float>(HEAD_DIM)) - max_score) / denominator;
    for (int dim = 0; dim < HEAD_DIM; ++dim) {
      context_row[dim] += attention * v_row[dim];
    }
  }
}

__global__ void mask_planet_logits_kernel(
    float* logits,
    const unsigned char* planet_mask,
    int batch_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * ACTION_SLOTS * PLANETS;
  if (index >= total) return;
  int planet = index % PLANETS;
  int batch = index / (ACTION_SLOTS * PLANETS);
  if (!planet_mask[batch * PLANETS + planet]) {
    logits[index] = NEG_INF;
  }
}

int blocks_for(int count) {
  return (count + 255) / 256;
}

int resident_compact_token_count() {
  static int cached = 0;
  if (cached > 0) {
    return cached;
  }
  int fleet_cap = 128;
  if (const char* value = std::getenv("ORBIT_WARS_V8_TOKEN_FLEET_CAP")) {
    const int parsed = std::atoi(value);
    if (parsed > 0) {
      fleet_cap = parsed;
    }
  }
  fleet_cap = std::max(0, std::min(fleet_cap, MAX_TOKEN_FLEETS));
  cached = 1 + PLANETS + fleet_cap;
  return cached;
}

cudaError_t launch_cutlass_linear(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows,
    int in_features,
    int out_features) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0) {
    return cudaSuccess;
  }
  if (out_features >= 8) {
    CutlassLinearGemm gemm_op;
    CutlassLinearGemm::Arguments arguments(
        cutlass::gemm::GemmCoord(rows, out_features, in_features),
        {input, in_features},
        {weight, in_features},
        {output, out_features},
        {output, out_features},
        {1.0f, 0.0f});
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorNotSupported;
    }
    status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
  } else {
    CutlassLinearGemmSimt gemm_op;
    CutlassLinearGemmSimt::Arguments arguments(
        cutlass::gemm::GemmCoord(rows, out_features, in_features),
        {input, in_features},
        {weight, in_features},
        {output, out_features},
        {output, out_features},
        {1.0f, 0.0f});
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorNotSupported;
    }
    status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
  }
  add_linear_bias_kernel<<<blocks_for(rows * out_features), 256>>>(output, bias, rows, out_features);
  return cudaGetLastError();
}

cudaError_t launch_linear(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows,
    int in_features,
    int out_features) {
  return launch_cutlass_linear(input, weight, bias, output, rows, in_features, out_features);
}

cudaError_t launch_grouped_cutlass_linear(
    CudaSimState* sim_state,
    const std::vector<const float*>& inputs,
    const std::vector<const float*>& weights,
    const std::vector<float*>& outputs,
    const std::vector<int>& rows,
    int in_features,
    int out_features) {
  const int problem_count = static_cast<int>(inputs.size());
  if (problem_count == 0) return cudaSuccess;
  if (!sim_state || weights.size() != inputs.size() || outputs.size() != inputs.size() ||
      rows.size() != inputs.size()) {
    return cudaErrorInvalidValue;
  }
  std::vector<cutlass::gemm::GemmCoord> host_problem_sizes;
  std::vector<float*> host_ptr_a;
  std::vector<float*> host_ptr_b;
  std::vector<float*> host_ptr_c;
  std::vector<float*> host_ptr_d;
  std::vector<int64_t> host_lda;
  std::vector<int64_t> host_ldb;
  std::vector<int64_t> host_ldc;
  std::vector<int64_t> host_ldd;
  host_problem_sizes.reserve(problem_count);
  host_ptr_a.reserve(problem_count);
  host_ptr_b.reserve(problem_count);
  host_ptr_c.reserve(problem_count);
  host_ptr_d.reserve(problem_count);
  host_lda.reserve(problem_count);
  host_ldb.reserve(problem_count);
  host_ldc.reserve(problem_count);
  host_ldd.reserve(problem_count);
  for (int index = 0; index < problem_count; ++index) {
    if (rows[index] <= 0) continue;
    host_problem_sizes.emplace_back(rows[index], out_features, in_features);
    host_ptr_a.push_back(const_cast<float*>(inputs[index]));
    host_ptr_b.push_back(const_cast<float*>(weights[index]));
    host_ptr_c.push_back(outputs[index]);
    host_ptr_d.push_back(outputs[index]);
    host_lda.push_back(in_features);
    host_ldb.push_back(in_features);
    host_ldc.push_back(out_features);
    host_ldd.push_back(out_features);
  }
  const int active_count = static_cast<int>(host_problem_sizes.size());
  if (active_count == 0) return cudaSuccess;
  cudaError_t error = cudaSuccess;
#define ENSURE_GROUPED(buffer, count) \
  do { error = (buffer).ensure(count); if (error != cudaSuccess) return error; } while (0)
  ENSURE_GROUPED(sim_state->grouped_problem_sizes, active_count);
  ENSURE_GROUPED(sim_state->grouped_ptr_a, active_count);
  ENSURE_GROUPED(sim_state->grouped_ptr_b, active_count);
  ENSURE_GROUPED(sim_state->grouped_ptr_c, active_count);
  ENSURE_GROUPED(sim_state->grouped_ptr_d, active_count);
  ENSURE_GROUPED(sim_state->grouped_lda, active_count);
  ENSURE_GROUPED(sim_state->grouped_ldb, active_count);
  ENSURE_GROUPED(sim_state->grouped_ldc, active_count);
  ENSURE_GROUPED(sim_state->grouped_ldd, active_count);
#undef ENSURE_GROUPED
  error = sim_state->grouped_problem_sizes.copy_from_host(host_problem_sizes.data(), active_count);
  if (error != cudaSuccess) return error;
  error = sim_state->grouped_ptr_a.copy_from_host(host_ptr_a.data(), active_count);
  if (error != cudaSuccess) return error;
  error = sim_state->grouped_ptr_b.copy_from_host(host_ptr_b.data(), active_count);
  if (error != cudaSuccess) return error;
  error = sim_state->grouped_ptr_c.copy_from_host(host_ptr_c.data(), active_count);
  if (error != cudaSuccess) return error;
  error = sim_state->grouped_ptr_d.copy_from_host(host_ptr_d.data(), active_count);
  if (error != cudaSuccess) return error;
  error = sim_state->grouped_lda.copy_from_host(host_lda.data(), active_count);
  if (error != cudaSuccess) return error;
  error = sim_state->grouped_ldb.copy_from_host(host_ldb.data(), active_count);
  if (error != cudaSuccess) return error;
  error = sim_state->grouped_ldc.copy_from_host(host_ldc.data(), active_count);
  if (error != cudaSuccess) return error;
  error = sim_state->grouped_ldd.copy_from_host(host_ldd.data(), active_count);
  if (error != cudaSuccess) return error;

  CutlassGroupedLinearGemm gemm_op;
  int threadblock_count = CutlassGroupedLinearGemm::sufficient(host_problem_sizes.data(), active_count);
  if (threadblock_count <= 0) {
    std::fprintf(stderr, "grouped_cutlass_unsupported=sufficient active=%d in=%d out=%d\n",
                 active_count, in_features, out_features);
    std::fflush(stderr);
    return cudaErrorNotSupported;
  }
  CutlassGroupedLinearGemm::Arguments arguments(
      sim_state->grouped_problem_sizes.ptr,
      active_count,
      threadblock_count,
      CutlassGroupedLinearOutputOp::Params(1.0f, 0.0f),
      sim_state->grouped_ptr_a.ptr,
      sim_state->grouped_ptr_b.ptr,
      sim_state->grouped_ptr_c.ptr,
      sim_state->grouped_ptr_d.ptr,
      sim_state->grouped_lda.ptr,
      sim_state->grouped_ldb.ptr,
      sim_state->grouped_ldc.ptr,
      sim_state->grouped_ldd.ptr,
      host_problem_sizes.data());
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return cudaErrorNotSupported;
  size_t workspace_size = CutlassGroupedLinearGemm::get_workspace_size(arguments);
  if (workspace_size > 0) {
    error = sim_state->grouped_workspace.ensure(workspace_size);
    if (error != cudaSuccess) return error;
  }
  status = gemm_op.initialize(arguments, sim_state->grouped_workspace.ptr);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  return cudaSuccess;
}

cudaError_t launch_qkv(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows,
    int offset) {
  return launch_linear(
      input,
      weight + static_cast<size_t>(offset) * D_MODEL,
      bias + offset,
      output,
      rows,
      D_MODEL,
      D_MODEL);
}

cudaError_t launch_qkv_all(
    const float* input,
    const float* weight,
    const float* bias,
    float* qkv,
    float* q,
    float* k,
    float* v,
    int rows) {
  cudaError_t error = launch_linear(input, weight, bias, qkv, rows, D_MODEL, D_MODEL * 3);
  if (error != cudaSuccess) return error;
  split_qkv_kernel<<<blocks_for(rows * D_MODEL), 256>>>(qkv, q, k, v, rows);
  return cudaGetLastError();
}

cudaError_t launch_kv_all(
    const float* input,
    const float* weight,
    const float* bias,
    float* kv,
    float* k,
    float* v,
    int rows) {
  cudaError_t error = launch_linear(
      input,
      weight + static_cast<size_t>(D_MODEL) * D_MODEL,
      bias + D_MODEL,
      kv,
      rows,
      D_MODEL,
      D_MODEL * 2);
  if (error != cudaSuccess) return error;
  split_kv_kernel<<<blocks_for(rows * D_MODEL), 256>>>(kv, k, v, rows);
  return cudaGetLastError();
}

cudaError_t run_attention_block(
    const TensorMap& map,
    const char* prefix,
    const float* query,
    const float* key_value,
    const unsigned char* key_padding_mask,
    float* output,
    DeviceBuffer<float>& q,
    DeviceBuffer<float>& k,
    DeviceBuffer<float>& v,
    DeviceBuffer<float>& qkv,
    DeviceBuffer<float>& context,
    int batch_count,
    int query_count,
    int key_count) {
  const std::string base(prefix);
  const float* in_proj_weight = tensor_ptr(map, (base + ".in_proj_weight").c_str());
  const float* in_proj_bias = tensor_ptr(map, (base + ".in_proj_bias").c_str());
  const float* out_weight = tensor_ptr(map, (base + ".out_proj.weight").c_str());
  const float* out_bias = tensor_ptr(map, (base + ".out_proj.bias").c_str());
  if (!in_proj_weight || !in_proj_bias || !out_weight || !out_bias) {
    return cudaErrorInvalidSymbol;
  }
  cudaError_t error = cudaSuccess;
  if (query == key_value && query_count == key_count) {
    error = launch_qkv_all(query, in_proj_weight, in_proj_bias, qkv.ptr, q.ptr, k.ptr, v.ptr, batch_count * query_count);
  } else {
    error = launch_qkv(query, in_proj_weight, in_proj_bias, q.ptr, batch_count * query_count, 0);
    if (error != cudaSuccess) return error;
    error = launch_kv_all(key_value, in_proj_weight, in_proj_bias, qkv.ptr, k.ptr, v.ptr, batch_count * key_count);
  }
  if (error != cudaSuccess) return error;
  attention_kernel<<<blocks_for(batch_count * query_count * HEADS), 256>>>(
      q.ptr, k.ptr, v.ptr, key_padding_mask, context.ptr, batch_count, query_count, key_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  return launch_linear(context.ptr, out_weight, out_bias, output, batch_count * query_count, D_MODEL, D_MODEL);
}

cudaError_t encoder_layer(
    const TensorMap& map,
    int layer,
    float* hidden,
    const unsigned char* padding_mask,
    DeviceBuffer<float>& norm,
    DeviceBuffer<float>& attn,
    DeviceBuffer<float>& q,
    DeviceBuffer<float>& k,
    DeviceBuffer<float>& v,
    DeviceBuffer<float>& qkv,
    DeviceBuffer<float>& context,
    DeviceBuffer<float>& ff_mid,
    DeviceBuffer<float>& ff_out,
    int batch_count,
    int token_count) {
  std::string prefix = "encoder.layers." + std::to_string(layer);
  const float* norm1_weight = tensor_ptr(map, (prefix + ".norm1.weight").c_str());
  const float* norm1_bias = tensor_ptr(map, (prefix + ".norm1.bias").c_str());
  if (!norm1_weight || !norm1_bias) return cudaErrorInvalidSymbol;
  layer_norm_kernel<<<blocks_for(batch_count * token_count), 256>>>(
      hidden, norm1_weight, norm1_bias, norm.ptr, batch_count * token_count);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  error = run_attention_block(
      map, (prefix + ".self_attn").c_str(), norm.ptr, norm.ptr, padding_mask, attn.ptr,
      q, k, v, qkv, context, batch_count, token_count, token_count);
  if (error != cudaSuccess) return error;
  add_kernel<<<blocks_for(batch_count * token_count * D_MODEL), 256>>>(
      hidden, attn.ptr, batch_count * token_count * D_MODEL);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  const float* norm2_weight = tensor_ptr(map, (prefix + ".norm2.weight").c_str());
  const float* norm2_bias = tensor_ptr(map, (prefix + ".norm2.bias").c_str());
  const float* linear1_weight = tensor_ptr(map, (prefix + ".linear1.weight").c_str());
  const float* linear1_bias = tensor_ptr(map, (prefix + ".linear1.bias").c_str());
  const float* linear2_weight = tensor_ptr(map, (prefix + ".linear2.weight").c_str());
  const float* linear2_bias = tensor_ptr(map, (prefix + ".linear2.bias").c_str());
  if (!norm2_weight || !norm2_bias || !linear1_weight || !linear1_bias || !linear2_weight || !linear2_bias) {
    return cudaErrorInvalidSymbol;
  }
  layer_norm_kernel<<<blocks_for(batch_count * token_count), 256>>>(
      hidden, norm2_weight, norm2_bias, norm.ptr, batch_count * token_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  error = launch_linear(norm.ptr, linear1_weight, linear1_bias, ff_mid.ptr, batch_count * token_count, D_MODEL, FF_DIM);
  if (error != cudaSuccess) return error;
  gelu_kernel<<<blocks_for(batch_count * token_count * FF_DIM), 256>>>(ff_mid.ptr, batch_count * token_count * FF_DIM);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  error = launch_linear(ff_mid.ptr, linear2_weight, linear2_bias, ff_out.ptr, batch_count * token_count, FF_DIM, D_MODEL);
  if (error != cudaSuccess) return error;
  add_kernel<<<blocks_for(batch_count * token_count * D_MODEL), 256>>>(
      hidden, ff_out.ptr, batch_count * token_count * D_MODEL);
  return cudaGetLastError();
}

cudaError_t decoder_layer(
    const TensorMap& map,
    int layer,
    float* slots,
    const float* memory,
    const unsigned char* memory_padding_mask,
    DeviceBuffer<float>& norm,
    DeviceBuffer<float>& attn,
    DeviceBuffer<float>& q,
    DeviceBuffer<float>& k,
    DeviceBuffer<float>& v,
    DeviceBuffer<float>& qkv,
    DeviceBuffer<float>& context,
    DeviceBuffer<float>& ff_mid,
    DeviceBuffer<float>& ff_out,
    int batch_count,
    int token_count) {
  std::string prefix = "decoder.layers." + std::to_string(layer);
  const float* norm1_weight = tensor_ptr(map, (prefix + ".norm1.weight").c_str());
  const float* norm1_bias = tensor_ptr(map, (prefix + ".norm1.bias").c_str());
  if (!norm1_weight || !norm1_bias) return cudaErrorInvalidSymbol;
  layer_norm_kernel<<<blocks_for(batch_count * ACTION_SLOTS), 256>>>(
      slots, norm1_weight, norm1_bias, norm.ptr, batch_count * ACTION_SLOTS);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  error = run_attention_block(
      map, (prefix + ".self_attn").c_str(), norm.ptr, norm.ptr, nullptr, attn.ptr,
      q, k, v, qkv, context, batch_count, ACTION_SLOTS, ACTION_SLOTS);
  if (error != cudaSuccess) return error;
  add_kernel<<<blocks_for(batch_count * ACTION_SLOTS * D_MODEL), 256>>>(
      slots, attn.ptr, batch_count * ACTION_SLOTS * D_MODEL);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  const float* norm2_weight = tensor_ptr(map, (prefix + ".norm2.weight").c_str());
  const float* norm2_bias = tensor_ptr(map, (prefix + ".norm2.bias").c_str());
  if (!norm2_weight || !norm2_bias) return cudaErrorInvalidSymbol;
  layer_norm_kernel<<<blocks_for(batch_count * ACTION_SLOTS), 256>>>(
      slots, norm2_weight, norm2_bias, norm.ptr, batch_count * ACTION_SLOTS);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  error = run_attention_block(
      map, (prefix + ".multihead_attn").c_str(), norm.ptr, memory, memory_padding_mask, attn.ptr,
      q, k, v, qkv, context, batch_count, ACTION_SLOTS, token_count);
  if (error != cudaSuccess) return error;
  add_kernel<<<blocks_for(batch_count * ACTION_SLOTS * D_MODEL), 256>>>(
      slots, attn.ptr, batch_count * ACTION_SLOTS * D_MODEL);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  const float* norm3_weight = tensor_ptr(map, (prefix + ".norm3.weight").c_str());
  const float* norm3_bias = tensor_ptr(map, (prefix + ".norm3.bias").c_str());
  const float* linear1_weight = tensor_ptr(map, (prefix + ".linear1.weight").c_str());
  const float* linear1_bias = tensor_ptr(map, (prefix + ".linear1.bias").c_str());
  const float* linear2_weight = tensor_ptr(map, (prefix + ".linear2.weight").c_str());
  const float* linear2_bias = tensor_ptr(map, (prefix + ".linear2.bias").c_str());
  if (!norm3_weight || !norm3_bias || !linear1_weight || !linear1_bias || !linear2_weight || !linear2_bias) {
    return cudaErrorInvalidSymbol;
  }
  layer_norm_kernel<<<blocks_for(batch_count * ACTION_SLOTS), 256>>>(
      slots, norm3_weight, norm3_bias, norm.ptr, batch_count * ACTION_SLOTS);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  error = launch_linear(norm.ptr, linear1_weight, linear1_bias, ff_mid.ptr, batch_count * ACTION_SLOTS, D_MODEL, FF_DIM);
  if (error != cudaSuccess) return error;
  gelu_kernel<<<blocks_for(batch_count * ACTION_SLOTS * FF_DIM), 256>>>(ff_mid.ptr, batch_count * ACTION_SLOTS * FF_DIM);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  error = launch_linear(ff_mid.ptr, linear2_weight, linear2_bias, ff_out.ptr, batch_count * ACTION_SLOTS, FF_DIM, D_MODEL);
  if (error != cudaSuccess) return error;
  add_kernel<<<blocks_for(batch_count * ACTION_SLOTS * D_MODEL), 256>>>(
      slots, ff_out.ptr, batch_count * ACTION_SLOTS * D_MODEL);
  return cudaGetLastError();
}

cudaError_t grouped_encoder_layer(
    CudaSimState* sim_state,
    const std::vector<CudaModelState*>& models,
    const std::vector<int>& counts,
    int layer,
    int token_count) {
  if (models.empty()) return cudaSuccess;
  std::vector<const float*> inputs;
  std::vector<const float*> weights;
  std::vector<const float*> biases;
  std::vector<float*> outputs;
  std::vector<int> rows;
  inputs.reserve(models.size());
  weights.reserve(models.size());
  biases.reserve(models.size());
  outputs.reserve(models.size());
  rows.reserve(models.size());
  auto reset = [&]() {
    inputs.clear();
    weights.clear();
    biases.clear();
    outputs.clear();
    rows.clear();
  };
  cudaError_t error = cudaSuccess;
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const EncoderLayerWeights& layer_weights = model->weights.encoder[layer];
    const int row_count = counts[index] * token_count;
    layer_norm_kernel<<<blocks_for(row_count), 256>>>(
        workspace.hidden.ptr, layer_weights.norm1_weight, layer_weights.norm1_bias, workspace.norm_token.ptr, row_count);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    inputs.push_back(workspace.norm_token.ptr);
    weights.push_back(layer_weights.self_attn_in_proj_weight);
    biases.push_back(layer_weights.self_attn_in_proj_bias);
    outputs.push_back(workspace.token_qkv.ptr);
    rows.push_back(row_count);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, D_MODEL, D_MODEL * 3);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    ForwardWorkspace& workspace = models[index]->workspace;
    const EncoderLayerWeights& layer_weights = models[index]->weights.encoder[layer];
    const int row_count = counts[index] * token_count;
    split_qkv_bias_kernel<<<blocks_for(row_count * D_MODEL), 256>>>(
        workspace.token_qkv.ptr, layer_weights.self_attn_in_proj_bias, workspace.token_q.ptr, workspace.token_k.ptr, workspace.token_v.ptr, row_count);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    attention_kernel<<<blocks_for(counts[index] * token_count * HEADS), 256>>>(
        workspace.token_q.ptr,
        workspace.token_k.ptr,
        workspace.token_v.ptr,
        workspace.d_padding_mask.ptr,
        workspace.token_context.ptr,
        counts[index],
        token_count,
        token_count);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  reset();
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const EncoderLayerWeights& layer_weights = model->weights.encoder[layer];
    inputs.push_back(workspace.token_context.ptr);
    weights.push_back(layer_weights.self_attn_out_proj_weight);
    biases.push_back(layer_weights.self_attn_out_proj_bias);
    outputs.push_back(workspace.token_attn.ptr);
    rows.push_back(counts[index] * token_count);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, D_MODEL, D_MODEL);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const EncoderLayerWeights& layer_weights = model->weights.encoder[layer];
    const int element_count = counts[index] * token_count * D_MODEL;
    add_bias_residual_kernel<<<blocks_for(element_count), 256>>>(
        workspace.hidden.ptr, workspace.token_attn.ptr, layer_weights.self_attn_out_proj_bias, counts[index] * token_count, D_MODEL);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  reset();
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const EncoderLayerWeights& layer_weights = model->weights.encoder[layer];
    const int row_count = counts[index] * token_count;
    layer_norm_kernel<<<blocks_for(row_count), 256>>>(
        workspace.hidden.ptr, layer_weights.norm2_weight, layer_weights.norm2_bias, workspace.norm_token.ptr, row_count);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    inputs.push_back(workspace.norm_token.ptr);
    weights.push_back(layer_weights.linear1_weight);
    biases.push_back(layer_weights.linear1_bias);
    outputs.push_back(workspace.token_ff_mid.ptr);
    rows.push_back(row_count);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, D_MODEL, FF_DIM);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    const EncoderLayerWeights& layer_weights = models[index]->weights.encoder[layer];
    const int element_count = counts[index] * token_count * FF_DIM;
    add_bias_gelu_kernel<<<blocks_for(element_count), 256>>>(
        models[index]->workspace.token_ff_mid.ptr, layer_weights.linear1_bias, counts[index] * token_count, FF_DIM);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  reset();
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const EncoderLayerWeights& layer_weights = model->weights.encoder[layer];
    inputs.push_back(workspace.token_ff_mid.ptr);
    weights.push_back(layer_weights.linear2_weight);
    biases.push_back(layer_weights.linear2_bias);
    outputs.push_back(workspace.token_ff_out.ptr);
    rows.push_back(counts[index] * token_count);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, FF_DIM, D_MODEL);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const EncoderLayerWeights& layer_weights = model->weights.encoder[layer];
    const int element_count = counts[index] * token_count * D_MODEL;
    add_bias_residual_kernel<<<blocks_for(element_count), 256>>>(
        workspace.hidden.ptr, workspace.token_ff_out.ptr, layer_weights.linear2_bias, counts[index] * token_count, D_MODEL);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

cudaError_t grouped_decoder_layer(
    CudaSimState* sim_state,
    const std::vector<CudaModelState*>& models,
    const std::vector<int>& counts,
    int layer,
    int token_count) {
  if (models.empty()) return cudaSuccess;
  std::vector<const float*> inputs;
  std::vector<const float*> weights;
  std::vector<const float*> biases;
  std::vector<float*> outputs;
  std::vector<int> rows;
  inputs.reserve(models.size());
  weights.reserve(models.size());
  biases.reserve(models.size());
  outputs.reserve(models.size());
  rows.reserve(models.size());
  auto reset = [&]() {
    inputs.clear();
    weights.clear();
    biases.clear();
    outputs.clear();
    rows.clear();
  };
  cudaError_t error = cudaSuccess;

  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const DecoderLayerWeights& layer_weights = model->weights.decoder[layer];
    const int slot_rows = counts[index] * ACTION_SLOTS;
    layer_norm_kernel<<<blocks_for(slot_rows), 256>>>(
        workspace.slots.ptr, layer_weights.norm1_weight, layer_weights.norm1_bias, workspace.norm_slot.ptr, slot_rows);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    inputs.push_back(workspace.norm_slot.ptr);
    weights.push_back(layer_weights.self_attn_in_proj_weight);
    biases.push_back(layer_weights.self_attn_in_proj_bias);
    outputs.push_back(workspace.slot_qkv.ptr);
    rows.push_back(slot_rows);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, D_MODEL, D_MODEL * 3);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    ForwardWorkspace& workspace = models[index]->workspace;
    const DecoderLayerWeights& layer_weights = models[index]->weights.decoder[layer];
    const int slot_rows = counts[index] * ACTION_SLOTS;
    split_qkv_bias_kernel<<<blocks_for(slot_rows * D_MODEL), 256>>>(
        workspace.slot_qkv.ptr, layer_weights.self_attn_in_proj_bias, workspace.slot_q.ptr, workspace.slot_k.ptr, workspace.slot_v.ptr, slot_rows);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    attention_kernel<<<blocks_for(counts[index] * ACTION_SLOTS * HEADS), 256>>>(
        workspace.slot_q.ptr,
        workspace.slot_k.ptr,
        workspace.slot_v.ptr,
        nullptr,
        workspace.slot_context.ptr,
        counts[index],
        ACTION_SLOTS,
        ACTION_SLOTS);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  reset();
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const DecoderLayerWeights& layer_weights = model->weights.decoder[layer];
    inputs.push_back(workspace.slot_context.ptr);
    weights.push_back(layer_weights.self_attn_out_proj_weight);
    biases.push_back(layer_weights.self_attn_out_proj_bias);
    outputs.push_back(workspace.slot_attn.ptr);
    rows.push_back(counts[index] * ACTION_SLOTS);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, D_MODEL, D_MODEL);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    const DecoderLayerWeights& layer_weights = models[index]->weights.decoder[layer];
    const int element_count = counts[index] * ACTION_SLOTS * D_MODEL;
    add_bias_residual_kernel<<<blocks_for(element_count), 256>>>(
        models[index]->workspace.slots.ptr,
        models[index]->workspace.slot_attn.ptr,
        layer_weights.self_attn_out_proj_bias,
        counts[index] * ACTION_SLOTS,
        D_MODEL);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  reset();
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const DecoderLayerWeights& layer_weights = model->weights.decoder[layer];
    const int slot_rows = counts[index] * ACTION_SLOTS;
    layer_norm_kernel<<<blocks_for(slot_rows), 256>>>(
        workspace.slots.ptr, layer_weights.norm2_weight, layer_weights.norm2_bias, workspace.norm_slot.ptr, slot_rows);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    inputs.push_back(workspace.norm_slot.ptr);
    weights.push_back(layer_weights.cross_attn_in_proj_weight);
    biases.push_back(layer_weights.cross_attn_in_proj_bias);
    outputs.push_back(workspace.slot_q.ptr);
    rows.push_back(slot_rows);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, D_MODEL, D_MODEL);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    const DecoderLayerWeights& layer_weights = models[index]->weights.decoder[layer];
    const int slot_rows = counts[index] * ACTION_SLOTS;
    add_linear_bias_kernel<<<blocks_for(slot_rows * D_MODEL), 256>>>(
        models[index]->workspace.slot_q.ptr, layer_weights.cross_attn_in_proj_bias, slot_rows, D_MODEL);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  reset();
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const DecoderLayerWeights& layer_weights = model->weights.decoder[layer];
    inputs.push_back(workspace.hidden.ptr);
    weights.push_back(layer_weights.cross_attn_in_proj_weight + static_cast<size_t>(D_MODEL) * D_MODEL);
    biases.push_back(layer_weights.cross_attn_in_proj_bias + D_MODEL);
    outputs.push_back(workspace.slot_qkv.ptr);
    rows.push_back(counts[index] * token_count);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, D_MODEL, D_MODEL * 2);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    ForwardWorkspace& workspace = models[index]->workspace;
    const DecoderLayerWeights& layer_weights = models[index]->weights.decoder[layer];
    const int token_rows = counts[index] * token_count;
    split_kv_bias_kernel<<<blocks_for(token_rows * D_MODEL), 256>>>(
        workspace.slot_qkv.ptr, layer_weights.cross_attn_in_proj_bias + D_MODEL, workspace.slot_k.ptr, workspace.slot_v.ptr, token_rows);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    attention_kernel<<<blocks_for(counts[index] * ACTION_SLOTS * HEADS), 256>>>(
        workspace.slot_q.ptr,
        workspace.slot_k.ptr,
        workspace.slot_v.ptr,
        workspace.d_padding_mask.ptr,
        workspace.slot_context.ptr,
        counts[index],
        ACTION_SLOTS,
        token_count);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  reset();
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const DecoderLayerWeights& layer_weights = model->weights.decoder[layer];
    inputs.push_back(workspace.slot_context.ptr);
    weights.push_back(layer_weights.cross_attn_out_proj_weight);
    biases.push_back(layer_weights.cross_attn_out_proj_bias);
    outputs.push_back(workspace.slot_attn.ptr);
    rows.push_back(counts[index] * ACTION_SLOTS);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, D_MODEL, D_MODEL);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    const DecoderLayerWeights& layer_weights = models[index]->weights.decoder[layer];
    const int element_count = counts[index] * ACTION_SLOTS * D_MODEL;
    add_bias_residual_kernel<<<blocks_for(element_count), 256>>>(
        models[index]->workspace.slots.ptr,
        models[index]->workspace.slot_attn.ptr,
        layer_weights.cross_attn_out_proj_bias,
        counts[index] * ACTION_SLOTS,
        D_MODEL);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  reset();
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const DecoderLayerWeights& layer_weights = model->weights.decoder[layer];
    const int slot_rows = counts[index] * ACTION_SLOTS;
    layer_norm_kernel<<<blocks_for(slot_rows), 256>>>(
        workspace.slots.ptr, layer_weights.norm3_weight, layer_weights.norm3_bias, workspace.norm_slot.ptr, slot_rows);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    inputs.push_back(workspace.norm_slot.ptr);
    weights.push_back(layer_weights.linear1_weight);
    biases.push_back(layer_weights.linear1_bias);
    outputs.push_back(workspace.slot_ff_mid.ptr);
    rows.push_back(slot_rows);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, D_MODEL, FF_DIM);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    const DecoderLayerWeights& layer_weights = models[index]->weights.decoder[layer];
    const int element_count = counts[index] * ACTION_SLOTS * FF_DIM;
    add_bias_gelu_kernel<<<blocks_for(element_count), 256>>>(
        models[index]->workspace.slot_ff_mid.ptr, layer_weights.linear1_bias, counts[index] * ACTION_SLOTS, FF_DIM);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  reset();
  for (size_t index = 0; index < models.size(); ++index) {
    CudaModelState* model = models[index];
    ForwardWorkspace& workspace = model->workspace;
    const DecoderLayerWeights& layer_weights = model->weights.decoder[layer];
    inputs.push_back(workspace.slot_ff_mid.ptr);
    weights.push_back(layer_weights.linear2_weight);
    biases.push_back(layer_weights.linear2_bias);
    outputs.push_back(workspace.slot_ff_out.ptr);
    rows.push_back(counts[index] * ACTION_SLOTS);
  }
  error = launch_grouped_cutlass_linear(sim_state, inputs, weights, outputs, rows, FF_DIM, D_MODEL);
  if (error != cudaSuccess) return error;
  for (size_t index = 0; index < models.size(); ++index) {
    const DecoderLayerWeights& layer_weights = models[index]->weights.decoder[layer];
    const int element_count = counts[index] * ACTION_SLOTS * D_MODEL;
    add_bias_residual_kernel<<<blocks_for(element_count), 256>>>(
        models[index]->workspace.slots.ptr,
        models[index]->workspace.slot_ff_out.ptr,
        layer_weights.linear2_bias,
        counts[index] * ACTION_SLOTS,
        D_MODEL);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

OrbitWarsV8CudaStatus forward_with_tensor_map(
    const TensorMap& tensor_map,
    const float* tokens,
    const long long* token_type_ids,
    const long long* owner_ids,
    const unsigned char* padding_mask,
    const unsigned char* planet_mask,
    float* fire_logits,
    float* source_logits,
    float* target_logits,
    float* amount_logits,
    OrbitWarsV8CudaShape shape,
    ForwardWorkspace* reusable_workspace) {
  debug_stage("forward_begin");
  const int batch_count = static_cast<int>(shape.batch_count);
  const int token_count = static_cast<int>(shape.token_count);
  const int token_rows = batch_count * token_count;
  const int slot_rows = batch_count * ACTION_SLOTS;

  const float* token_projection_weight = tensor_ptr(tensor_map, "token_projection.weight");
  const float* token_projection_bias = tensor_ptr(tensor_map, "token_projection.bias");
  const float* type_embedding = tensor_ptr(tensor_map, "type_embedding.weight");
  const float* owner_embedding = tensor_ptr(tensor_map, "owner_embedding.weight");
  const float* slot_queries = tensor_ptr(tensor_map, "slot_queries");
  if (!token_projection_weight || !token_projection_bias || !type_embedding || !owner_embedding || !slot_queries) {
    return bad_argument("missing required OWV8 tensor");
  }
  debug_stage("tensors_ready");

  ForwardWorkspace local_workspace;
  ForwardWorkspace& workspace = reusable_workspace ? *reusable_workspace : local_workspace;
  DeviceBuffer<float>& d_tokens = workspace.d_tokens;
  DeviceBuffer<long long>& d_token_type_ids = workspace.d_token_type_ids;
  DeviceBuffer<long long>& d_owner_ids = workspace.d_owner_ids;
  DeviceBuffer<unsigned char>& d_padding_mask = workspace.d_padding_mask;
  DeviceBuffer<unsigned char>& d_planet_mask = workspace.d_planet_mask;
  DeviceBuffer<float>& hidden = workspace.hidden;
  DeviceBuffer<float>& norm_token = workspace.norm_token;
  DeviceBuffer<float>& token_attn = workspace.token_attn;
  DeviceBuffer<float>& token_q = workspace.token_q;
  DeviceBuffer<float>& token_k = workspace.token_k;
  DeviceBuffer<float>& token_v = workspace.token_v;
  DeviceBuffer<float>& token_qkv = workspace.token_qkv;
  DeviceBuffer<float>& token_context = workspace.token_context;
  DeviceBuffer<float>& token_ff_mid = workspace.token_ff_mid;
  DeviceBuffer<float>& token_ff_out = workspace.token_ff_out;
  DeviceBuffer<float>& slots = workspace.slots;
  DeviceBuffer<float>& norm_slot = workspace.norm_slot;
  DeviceBuffer<float>& slot_attn = workspace.slot_attn;
  DeviceBuffer<float>& slot_q = workspace.slot_q;
  DeviceBuffer<float>& slot_k = workspace.slot_k;
  DeviceBuffer<float>& slot_v = workspace.slot_v;
  DeviceBuffer<float>& slot_qkv = workspace.slot_qkv;
  DeviceBuffer<float>& slot_context = workspace.slot_context;
  DeviceBuffer<float>& slot_ff_mid = workspace.slot_ff_mid;
  DeviceBuffer<float>& slot_ff_out = workspace.slot_ff_out;
  DeviceBuffer<float>& d_fire = workspace.d_fire;
  DeviceBuffer<float>& d_source = workspace.d_source;
  DeviceBuffer<float>& d_target = workspace.d_target;
  DeviceBuffer<float>& d_amount = workspace.d_amount;

  cudaError_t error = cudaSuccess;
#define ENSURE(buffer, count) \
  do { error = (buffer).ensure(count); if (error != cudaSuccess) return cuda_error(error); } while (0)
  ENSURE(d_tokens, static_cast<size_t>(token_rows) * TOKEN_FEATURES);
  ENSURE(d_token_type_ids, token_rows);
  ENSURE(d_owner_ids, token_rows);
  ENSURE(d_padding_mask, token_rows);
  ENSURE(d_planet_mask, static_cast<size_t>(batch_count) * PLANETS);
  ENSURE(hidden, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(norm_token, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(token_attn, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(token_q, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(token_k, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(token_v, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(token_qkv, static_cast<size_t>(token_rows) * D_MODEL * 3);
  ENSURE(token_context, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(token_ff_mid, static_cast<size_t>(token_rows) * FF_DIM);
  ENSURE(token_ff_out, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(slots, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(norm_slot, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(slot_attn, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(slot_q, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(slot_k, static_cast<size_t>(std::max(token_rows, slot_rows)) * D_MODEL);
  ENSURE(slot_v, static_cast<size_t>(std::max(token_rows, slot_rows)) * D_MODEL);
  ENSURE(slot_qkv, static_cast<size_t>(std::max(token_rows * 2, slot_rows * 3)) * D_MODEL);
  ENSURE(slot_context, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(slot_ff_mid, static_cast<size_t>(slot_rows) * FF_DIM);
  ENSURE(slot_ff_out, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(d_fire, static_cast<size_t>(slot_rows));
  ENSURE(d_source, static_cast<size_t>(slot_rows) * PLANETS);
  ENSURE(d_target, static_cast<size_t>(slot_rows) * PLANETS);
  ENSURE(d_amount, static_cast<size_t>(slot_rows) * AMOUNTS);
#undef ENSURE

  error = d_tokens.copy_from_host(tokens, static_cast<size_t>(token_rows) * TOKEN_FEATURES);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_token_type_ids.copy_from_host(token_type_ids, token_rows);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_owner_ids.copy_from_host(owner_ids, token_rows);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_padding_mask.copy_from_host(padding_mask, token_rows);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_planet_mask.copy_from_host(planet_mask, static_cast<size_t>(batch_count) * PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("inputs_copied");

  token_projection_kernel<<<blocks_for(token_rows * D_MODEL), 256>>>(
      d_tokens.ptr, d_token_type_ids.ptr, d_owner_ids.ptr, token_projection_weight,
      token_projection_bias, type_embedding, owner_embedding, hidden.ptr, batch_count, token_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("token_projection");

  for (int layer = 0; layer < ENCODER_LAYERS; ++layer) {
    error = encoder_layer(
        tensor_map, layer, hidden.ptr, d_padding_mask.ptr, norm_token, token_attn,
        token_q, token_k, token_v, token_qkv, token_context, token_ff_mid, token_ff_out,
        batch_count, token_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  debug_stage("encoder_done");

  copy_slot_queries_kernel<<<blocks_for(slot_rows * D_MODEL), 256>>>(slot_queries, slots.ptr, batch_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);

  for (int layer = 0; layer < DECODER_LAYERS; ++layer) {
    error = decoder_layer(
        tensor_map, layer, slots.ptr, hidden.ptr, d_padding_mask.ptr, norm_slot, slot_attn,
        slot_q, slot_k, slot_v, slot_qkv, slot_context, slot_ff_mid, slot_ff_out,
        batch_count, token_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  debug_stage("decoder_done");

  const float* fire_weight = tensor_ptr(tensor_map, "fire_head.weight");
  const float* fire_bias = tensor_ptr(tensor_map, "fire_head.bias");
  const float* source_weight = tensor_ptr(tensor_map, "source_head.weight");
  const float* source_bias = tensor_ptr(tensor_map, "source_head.bias");
  const float* target_weight = tensor_ptr(tensor_map, "target_head.weight");
  const float* target_bias = tensor_ptr(tensor_map, "target_head.bias");
  const float* amount_weight = tensor_ptr(tensor_map, "amount_head.weight");
  const float* amount_bias = tensor_ptr(tensor_map, "amount_head.bias");
  if (!fire_weight || !fire_bias || !source_weight || !source_bias || !target_weight ||
      !target_bias || !amount_weight || !amount_bias) {
    return bad_argument("missing OWV8 head tensor");
  }
  error = launch_linear(slots.ptr, fire_weight, fire_bias, d_fire.ptr, slot_rows, D_MODEL, 1);
  if (error != cudaSuccess) return cuda_error(error);
  error = launch_linear(slots.ptr, source_weight, source_bias, d_source.ptr, slot_rows, D_MODEL, PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = launch_linear(slots.ptr, target_weight, target_bias, d_target.ptr, slot_rows, D_MODEL, PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = launch_linear(slots.ptr, amount_weight, amount_bias, d_amount.ptr, slot_rows, D_MODEL, AMOUNTS);
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("heads_done");
  mask_planet_logits_kernel<<<blocks_for(slot_rows * PLANETS), 256>>>(
      d_source.ptr, d_planet_mask.ptr, batch_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  mask_planet_logits_kernel<<<blocks_for(slot_rows * PLANETS), 256>>>(
      d_target.ptr, d_planet_mask.ptr, batch_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);

  error = d_fire.copy_to_host(fire_logits, static_cast<size_t>(slot_rows));
  if (error != cudaSuccess) return cuda_error(error);
  error = d_source.copy_to_host(source_logits, static_cast<size_t>(slot_rows) * PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_target.copy_to_host(target_logits, static_cast<size_t>(slot_rows) * PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_amount.copy_to_host(amount_logits, static_cast<size_t>(slot_rows) * AMOUNTS);
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("outputs_copied");
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("forward_done");
  return ok();
}

OrbitWarsV8CudaStatus forward_workspace_with_tensor_map(
    const TensorMap& tensor_map,
    ForwardWorkspace& workspace,
    OrbitWarsV8CudaShape shape,
    bool compute_heads = true) {
  const int batch_count = static_cast<int>(shape.batch_count);
  const int token_count = static_cast<int>(shape.token_count);
  const int token_rows = batch_count * token_count;
  const int slot_rows = batch_count * ACTION_SLOTS;
  const float* token_projection_weight = tensor_ptr(tensor_map, "token_projection.weight");
  const float* token_projection_bias = tensor_ptr(tensor_map, "token_projection.bias");
  const float* type_embedding = tensor_ptr(tensor_map, "type_embedding.weight");
  const float* owner_embedding = tensor_ptr(tensor_map, "owner_embedding.weight");
  const float* slot_queries = tensor_ptr(tensor_map, "slot_queries");
  if (!token_projection_weight || !token_projection_bias || !type_embedding || !owner_embedding || !slot_queries) {
    return bad_argument("missing required OWV8 tensor");
  }
  cudaError_t error = cudaSuccess;
#define ENSURE(buffer, count) \
  do { error = (buffer).ensure(count); if (error != cudaSuccess) return cuda_error(error); } while (0)
  ENSURE(workspace.hidden, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(workspace.norm_token, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(workspace.token_attn, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(workspace.token_q, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(workspace.token_k, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(workspace.token_v, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(workspace.token_qkv, static_cast<size_t>(token_rows) * D_MODEL * 3);
  ENSURE(workspace.token_context, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(workspace.token_ff_mid, static_cast<size_t>(token_rows) * FF_DIM);
  ENSURE(workspace.token_ff_out, static_cast<size_t>(token_rows) * D_MODEL);
  ENSURE(workspace.slots, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(workspace.norm_slot, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(workspace.slot_attn, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(workspace.slot_q, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(workspace.slot_k, static_cast<size_t>(std::max(token_rows, slot_rows)) * D_MODEL);
  ENSURE(workspace.slot_v, static_cast<size_t>(std::max(token_rows, slot_rows)) * D_MODEL);
  ENSURE(workspace.slot_qkv, static_cast<size_t>(std::max(token_rows * 2, slot_rows * 3)) * D_MODEL);
  ENSURE(workspace.slot_context, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(workspace.slot_ff_mid, static_cast<size_t>(slot_rows) * FF_DIM);
  ENSURE(workspace.slot_ff_out, static_cast<size_t>(slot_rows) * D_MODEL);
  ENSURE(workspace.d_fire, static_cast<size_t>(slot_rows));
  ENSURE(workspace.d_source, static_cast<size_t>(slot_rows) * PLANETS);
  ENSURE(workspace.d_target, static_cast<size_t>(slot_rows) * PLANETS);
  ENSURE(workspace.d_amount, static_cast<size_t>(slot_rows) * AMOUNTS);
#undef ENSURE

  token_projection_kernel<<<blocks_for(token_rows * D_MODEL), 256>>>(
      workspace.d_tokens.ptr,
      workspace.d_token_type_ids.ptr,
      workspace.d_owner_ids.ptr,
      token_projection_weight,
      token_projection_bias,
      type_embedding,
      owner_embedding,
      workspace.hidden.ptr,
      batch_count,
      token_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  for (int layer = 0; layer < ENCODER_LAYERS; ++layer) {
    error = encoder_layer(
        tensor_map,
        layer,
        workspace.hidden.ptr,
        workspace.d_padding_mask.ptr,
        workspace.norm_token,
        workspace.token_attn,
        workspace.token_q,
        workspace.token_k,
        workspace.token_v,
        workspace.token_qkv,
        workspace.token_context,
        workspace.token_ff_mid,
        workspace.token_ff_out,
        batch_count,
        token_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  copy_slot_queries_kernel<<<blocks_for(slot_rows * D_MODEL), 256>>>(slot_queries, workspace.slots.ptr, batch_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  for (int layer = 0; layer < DECODER_LAYERS; ++layer) {
    error = decoder_layer(
        tensor_map,
        layer,
        workspace.slots.ptr,
        workspace.hidden.ptr,
        workspace.d_padding_mask.ptr,
        workspace.norm_slot,
        workspace.slot_attn,
        workspace.slot_q,
        workspace.slot_k,
        workspace.slot_v,
        workspace.slot_qkv,
        workspace.slot_context,
        workspace.slot_ff_mid,
        workspace.slot_ff_out,
        batch_count,
        token_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  if (!compute_heads) {
    return ok();
  }
  const float* fire_weight = tensor_ptr(tensor_map, "fire_head.weight");
  const float* fire_bias = tensor_ptr(tensor_map, "fire_head.bias");
  const float* source_weight = tensor_ptr(tensor_map, "source_head.weight");
  const float* source_bias = tensor_ptr(tensor_map, "source_head.bias");
  const float* target_weight = tensor_ptr(tensor_map, "target_head.weight");
  const float* target_bias = tensor_ptr(tensor_map, "target_head.bias");
  const float* amount_weight = tensor_ptr(tensor_map, "amount_head.weight");
  const float* amount_bias = tensor_ptr(tensor_map, "amount_head.bias");
  if (!fire_weight || !fire_bias || !source_weight || !source_bias || !target_weight ||
      !target_bias || !amount_weight || !amount_bias) {
    return bad_argument("missing OWV8 head tensor");
  }
  error = launch_linear(workspace.slots.ptr, fire_weight, fire_bias, workspace.d_fire.ptr, slot_rows, D_MODEL, 1);
  if (error != cudaSuccess) return cuda_error(error);
  error = launch_linear(workspace.slots.ptr, source_weight, source_bias, workspace.d_source.ptr, slot_rows, D_MODEL, PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = launch_linear(workspace.slots.ptr, target_weight, target_bias, workspace.d_target.ptr, slot_rows, D_MODEL, PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = launch_linear(workspace.slots.ptr, amount_weight, amount_bias, workspace.d_amount.ptr, slot_rows, D_MODEL, AMOUNTS);
  if (error != cudaSuccess) return cuda_error(error);
  mask_planet_logits_kernel<<<blocks_for(slot_rows * PLANETS), 256>>>(workspace.d_source.ptr, workspace.d_planet_mask.ptr, batch_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  mask_planet_logits_kernel<<<blocks_for(slot_rows * PLANETS), 256>>>(workspace.d_target.ptr, workspace.d_planet_mask.ptr, batch_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

__device__ float sim_fleet_speed(float ships, const OrbitWarsCudaSimConfig& config) {
  const float clamped_ships = fmaxf(ships, 1.0f);
  const float speed_ratio = fminf(fmaxf(logf(clamped_ships) / logf(config.fleet_speed_reference_ships), 0.0f), 1.0f);
  return 1.0f + (config.fleet_speed_max - 1.0f) * powf(speed_ratio, config.fleet_speed_curve_power);
}

__device__ int owner_embedding_id_device(int owner) {
  int value = owner + 1;
  if (value < 0) return 0;
  if (value > 7) return 7;
  return value;
}

__global__ void resident_build_tokens_kernel(
    CudaSimKernelState sim,
    int* request_game_indices,
    int* request_player_ids,
    int request_count,
    int step,
    float* tokens,
    long long* token_type_ids,
    long long* owner_ids,
    unsigned char* padding_mask,
    unsigned char* planet_mask) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = request_count * RESIDENT_TOKEN_COUNT;
  if (flat >= total) return;
  const int request = flat / RESIDENT_TOKEN_COUNT;
  const int row = flat % RESIDENT_TOKEN_COUNT;
  const int game = request_game_indices[request];
  const int player = request_player_ids[request];
  const int token_offset = flat * TOKEN_FEATURES;
  for (int feature = 0; feature < TOKEN_FEATURES; ++feature) {
    tokens[token_offset + feature] = 0.0f;
  }
  token_type_ids[flat] = 0;
  owner_ids[flat] = owner_embedding_id_device(player);
  padding_mask[flat] = 1;
  if (row < PLANETS) {
    planet_mask[request * PLANETS + row] = 0;
  }
  OrbitWarsCudaPlanet* planets = sim.planets + static_cast<size_t>(game) * sim.config.planet_count;
  OrbitWarsCudaFleet* fleets = sim.fleets + static_cast<size_t>(game) * sim.config.max_fleets_per_game;
  if (row == 0) {
    int alive_fleets = 0;
    for (int fleet = 0; fleet < static_cast<int>(sim.config.max_fleets_per_game) && alive_fleets < MAX_TOKEN_FLEETS; ++fleet) {
      if (fleets[fleet].alive) {
        alive_fleets += 1;
      }
    }
    tokens[token_offset + 0] = static_cast<float>(player) / 3.0f;
    tokens[token_offset + 1] = static_cast<float>(step) / 500.0f;
    tokens[token_offset + 2] = sim.angular_velocities ? sim.angular_velocities[game] : sim.config.angular_velocity;
    tokens[token_offset + 3] = static_cast<float>(PLANETS) / static_cast<float>(PLANETS);
    tokens[token_offset + 4] = static_cast<float>(alive_fleets) / static_cast<float>(MAX_TOKEN_FLEETS);
    token_type_ids[flat] = 0;
    owner_ids[flat] = owner_embedding_id_device(player);
    padding_mask[flat] = 0;
    return;
  }
  if (row <= PLANETS) {
    const int planet_row = row - 1;
    OrbitWarsCudaPlanet planet = planets[planet_row];
    if (planet.id < 0 || planet.owner <= -9) {
      return;
    }
    tokens[token_offset + 0] = static_cast<float>(planet.id) / 128.0f;
    tokens[token_offset + 1] = static_cast<float>(planet.owner) / 3.0f;
    tokens[token_offset + 2] = planet.x / 100.0f;
    tokens[token_offset + 3] = planet.y / 100.0f;
    tokens[token_offset + 4] = planet.radius / 10.0f;
    tokens[token_offset + 5] = log1pf(fmaxf(planet.ships, 0.0f)) / 10.0f;
    tokens[token_offset + 6] = planet.production / 20.0f;
    tokens[token_offset + 7] = planet.velocity_x / 10.0f;
    tokens[token_offset + 8] = planet.velocity_y / 10.0f;
    tokens[token_offset + 11] = -1.0f;
    tokens[token_offset + 12] = static_cast<float>(planet_row) / static_cast<float>(PLANETS);
    token_type_ids[flat] = 1;
    owner_ids[flat] = owner_embedding_id_device(planet.owner);
    padding_mask[flat] = 0;
    planet_mask[request * PLANETS + planet_row] = 1;
    return;
  }
  const int fleet_token = row - 1 - PLANETS;
  int seen = 0;
  OrbitWarsCudaFleet selected{};
  bool found = false;
  for (int fleet = 0; fleet < static_cast<int>(sim.config.max_fleets_per_game); ++fleet) {
    if (!fleets[fleet].alive) continue;
    if (seen == fleet_token) {
      selected = fleets[fleet];
      found = true;
      break;
    }
    seen += 1;
    if (seen >= MAX_TOKEN_FLEETS) break;
  }
  if (!found) {
    return;
  }
  tokens[token_offset + 0] = static_cast<float>(selected.id) / 2048.0f;
  tokens[token_offset + 1] = static_cast<float>(selected.owner) / 3.0f;
  tokens[token_offset + 2] = selected.x / 100.0f;
  tokens[token_offset + 3] = selected.y / 100.0f;
  tokens[token_offset + 5] = log1pf(fmaxf(selected.ships, 0.0f)) / 10.0f;
  tokens[token_offset + 9] = sinf(selected.angle);
  tokens[token_offset + 10] = cosf(selected.angle);
  tokens[token_offset + 11] = static_cast<float>(selected.from_planet_id) / 128.0f;
  tokens[token_offset + 12] = -1.0f;
  tokens[token_offset + 13] = 0.0f;
  token_type_ids[flat] = 2;
  owner_ids[flat] = owner_embedding_id_device(selected.owner);
  padding_mask[flat] = 0;
}

__device__ int resident_argmax(const float* values, int count, int skip) {
  int best = -1;
  float best_value = -INFINITY;
  for (int index = 0; index < count; ++index) {
    if (index == skip) continue;
    const float value = values[index];
    if (!isfinite(value) && !isinf(value)) return -1;
    if (value > best_value) {
      best_value = value;
      best = index;
    }
  }
  return best;
}

__device__ int resident_amount_ships(int amount_index, float source_ships) {
  float requested = source_ships;
  if (amount_index == 1) requested = source_ships * 0.25f;
  else if (amount_index == 2) requested = source_ships * 0.50f;
  else if (amount_index == 3) requested = source_ships * 0.67f;
  else if (amount_index == 4) requested = source_ships * 0.70f;
  else if (amount_index == 5) requested = source_ships * 0.75f;
  else if (amount_index == 6) requested = source_ships * 0.90f;
  else if (amount_index == 7) requested = source_ships * 0.95f;
  else if (amount_index == 8) requested = 10.0f;
  else if (amount_index == 9) requested = 20.0f;
  else if (amount_index == 10) requested = 50.0f;
  else if (amount_index == 11) requested = 100.0f;
  else if (amount_index == 12) requested = 200.0f;
  else if (amount_index == 13) requested = 500.0f;
  else if (amount_index == 14) requested = 1000.0f;
  else if (amount_index == 15) requested = 2000.0f;
  return static_cast<int>(floorf(fmaxf(fminf(requested, source_ships), 0.0f)));
}

__global__ void resident_decode_actions_kernel(
    CudaSimKernelState sim,
    int* request_game_indices,
    int* request_player_ids,
    int request_count,
    const float* fire_logits,
    const float* source_logits,
    const float* target_logits,
    const float* amount_logits,
    OrbitWarsCudaActionLabel* labels) {
  const int request = blockIdx.x * blockDim.x + threadIdx.x;
  if (request >= request_count) return;
  const int game = request_game_indices[request];
  const int player = request_player_ids[request];
  OrbitWarsCudaPlanet* planets = sim.planets + static_cast<size_t>(game) * sim.config.planet_count;
  OrbitWarsCudaAction* actions = sim.actions + (static_cast<size_t>(game) * sim.config.max_players + player) * sim.config.max_actions_per_player;
  int* action_count = sim.action_counts + game * sim.config.max_players + player;
  *action_count = 0;
  OrbitWarsCudaActionLabel label{};
  for (int index = 0; index < ACTION_SLOTS; ++index) {
    label.source_row[index] = -1;
    label.target_row[index] = -1;
    label.amount_class[index] = -1;
    label.source_planet_id[index] = -1;
    label.target_planet_id[index] = -1;
  }
  if (labels) {
    labels[request] = label;
  }
  if (sim.statuses && sim.statuses[game].done) return;
  bool used_sources[PLANETS];
  for (int index = 0; index < PLANETS; ++index) used_sources[index] = false;
  bool used_slots[ACTION_SLOTS];
  for (int index = 0; index < ACTION_SLOTS; ++index) used_slots[index] = false;
  int written = 0;
  for (int slot_pick = 0; slot_pick < ACTION_SLOTS && written < static_cast<int>(sim.config.max_actions_per_player); ++slot_pick) {
    int best_slot = -1;
    float best_fire = -INFINITY;
    for (int slot = 0; slot < ACTION_SLOTS; ++slot) {
      if (used_slots[slot]) continue;
      const float value = fire_logits[request * ACTION_SLOTS + slot];
      if (!isfinite(value) || value <= 0.0f) continue;
      if (value > best_fire) {
        best_fire = value;
        best_slot = slot;
      }
    }
    if (best_slot < 0) break;
    used_slots[best_slot] = true;
    const float* source_row_logits = source_logits + (request * ACTION_SLOTS + best_slot) * PLANETS;
    const int source_row = resident_argmax(source_row_logits, PLANETS, -1);
    if (source_row < 0 || source_row >= PLANETS || used_sources[source_row]) continue;
    OrbitWarsCudaPlanet source = planets[source_row];
    if (source.id < 0 || source.owner != player || source.ships < 1.0f) continue;
    const float* target_row_logits = target_logits + (request * ACTION_SLOTS + best_slot) * PLANETS;
    const int target_row = resident_argmax(target_row_logits, PLANETS, source_row);
    if (target_row < 0 || target_row >= PLANETS) continue;
    OrbitWarsCudaPlanet target = planets[target_row];
    if (target.id < 0) continue;
    const float* amount_row_logits = amount_logits + (request * ACTION_SLOTS + best_slot) * AMOUNTS;
    const int amount_index = resident_argmax(amount_row_logits, AMOUNTS, -1);
    if (amount_index < 0) continue;
    const int ships = resident_amount_ships(amount_index, source.ships);
    if (ships < 1) continue;
    label.fire[written] = 1;
    label.source_row[written] = source_row;
    label.target_row[written] = target_row;
    label.amount_class[written] = amount_index;
    label.source_planet_id[written] = source.id;
    label.target_planet_id[written] = target.id;
    label.ship_count[written] = ships;
    label.confidence[written] = best_fire;
    actions[written] = OrbitWarsCudaAction{
        source.id,
        atan2f(target.y - source.y, target.x - source.x),
        ships};
    used_sources[source_row] = true;
    written += 1;
  }
  *action_count = written;
  if (labels) {
    labels[request] = label;
  }
}

__device__ bool sim_segment_intersects_sun(
    float start_x, float start_y, float end_x, float end_y, const OrbitWarsCudaSimConfig& config) {
  const float dx = end_x - start_x;
  const float dy = end_y - start_y;
  const float length_squared = dx * dx + dy * dy;
  float closest_x = start_x;
  float closest_y = start_y;
  if (length_squared > 0.0f) {
    const float projection = fminf(
        fmaxf(((config.board_center - start_x) * dx + (config.board_center - start_y) * dy) / length_squared, 0.0f),
        1.0f);
    closest_x = start_x + projection * dx;
    closest_y = start_y + projection * dy;
  }
  const float sun_dx = closest_x - config.board_center;
  const float sun_dy = closest_y - config.board_center;
  return sun_dx * sun_dx + sun_dy * sun_dy < config.sun_radius * config.sun_radius;
}

__device__ bool sim_swept_pair_hit(
    float fleet_start_x,
    float fleet_start_y,
    float fleet_end_x,
    float fleet_end_y,
    float planet_start_x,
    float planet_start_y,
    float planet_end_x,
    float planet_end_y,
    float radius) {
  const float delta_start_x = fleet_start_x - planet_start_x;
  const float delta_start_y = fleet_start_y - planet_start_y;
  const float delta_velocity_x = (fleet_end_x - fleet_start_x) - (planet_end_x - planet_start_x);
  const float delta_velocity_y = (fleet_end_y - fleet_start_y) - (planet_end_y - planet_start_y);
  const float a = delta_velocity_x * delta_velocity_x + delta_velocity_y * delta_velocity_y;
  const float b = 2.0f * (delta_start_x * delta_velocity_x + delta_start_y * delta_velocity_y);
  const float c = delta_start_x * delta_start_x + delta_start_y * delta_start_y - radius * radius;
  if (a < 1.1920929e-7f) {
    return c <= 0.0f;
  }
  const float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f) {
    return false;
  }
  const float root = sqrtf(discriminant);
  const float entry = (-b - root) / (2.0f * a);
  const float exit = (-b + root) / (2.0f * a);
  return exit >= 0.0f && entry <= 1.0f;
}

__device__ bool sim_point_out_of_bounds(float x, float y, const OrbitWarsCudaSimConfig& config) {
  return x < 0.0f || y < 0.0f || x > config.board_size || y > config.board_size;
}

__device__ int sim_find_planet_index(const OrbitWarsCudaPlanet* planets, int planet_count, int planet_id) {
  for (int index = 0; index < planet_count; ++index) {
    if (planets[index].id == planet_id) {
      return index;
    }
  }
  return -1;
}

__device__ int sim_find_free_fleet(OrbitWarsCudaFleet* fleets, int max_fleets) {
  for (int index = 0; index < max_fleets; ++index) {
    if (!fleets[index].alive) {
      return index;
    }
  }
  return -1;
}

__global__ void sim_step_kernel(
    OrbitWarsCudaPlanet* planets,
    const OrbitWarsCudaPlanet* initial_planets,
    OrbitWarsCudaFleet* fleets,
    int* next_fleet_ids,
    const OrbitWarsCudaAction* actions,
    const int* action_counts,
    OrbitWarsCudaSimStats* stats,
    OrbitWarsCudaSimConfig config) {
  const int game = blockIdx.x;
  if (threadIdx.x != 0 || game >= static_cast<int>(config.game_count)) {
    return;
  }
  OrbitWarsCudaPlanet* game_planets = planets + static_cast<size_t>(game) * config.planet_count;
  const OrbitWarsCudaPlanet* game_initial = initial_planets + static_cast<size_t>(game) * config.planet_count;
  OrbitWarsCudaFleet* game_fleets = fleets + static_cast<size_t>(game) * config.max_fleets_per_game;
  OrbitWarsCudaSimStats* game_stats = stats + static_cast<size_t>(game) * config.max_players;
  for (int player = 0; player < static_cast<int>(config.max_players); ++player) {
    game_stats[player] = OrbitWarsCudaSimStats{};
  }

  for (int player = 0; player < static_cast<int>(config.max_players); ++player) {
    const int action_count = action_counts[game * config.max_players + player];
    for (int action_index = 0; action_index < action_count; ++action_index) {
      const size_t action_offset =
          ((static_cast<size_t>(game) * config.max_players + player) * config.max_actions_per_player) + action_index;
      const OrbitWarsCudaAction action = actions[action_offset];
      if (action.ship_count < 1 || !isfinite(action.direction_angle)) {
        continue;
      }
      const int source_index = sim_find_planet_index(game_planets, static_cast<int>(config.planet_count), action.from_planet_id);
      if (source_index < 0 || game_planets[source_index].owner != player) {
        continue;
      }
      const int available = static_cast<int>(floorf(game_planets[source_index].ships));
      if (action.ship_count > available) {
        continue;
      }
      const int fleet_slot = sim_find_free_fleet(game_fleets, static_cast<int>(config.max_fleets_per_game));
      if (fleet_slot < 0) {
        game_stats[player].overflow_fleet_count += 1;
        continue;
      }
      const OrbitWarsCudaPlanet source = game_planets[source_index];
      game_planets[source_index].ships -= static_cast<float>(action.ship_count);
      const float spawn_distance = source.radius + config.fleet_spawn_offset;
      game_fleets[fleet_slot] = OrbitWarsCudaFleet{
          next_fleet_ids[game]++,
          player,
          source.x + cosf(action.direction_angle) * spawn_distance,
          source.y + sinf(action.direction_angle) * spawn_distance,
          action.direction_angle,
          action.from_planet_id,
          static_cast<float>(action.ship_count),
          1};
      game_stats[player].launched_fleet_count += 1;
      game_stats[player].launched_ship_count += action.ship_count;
    }
  }

  for (int planet = 0; planet < static_cast<int>(config.planet_count); ++planet) {
    if (game_planets[planet].owner >= 0) {
      game_planets[planet].ships += game_planets[planet].production;
    }
  }

  float old_x[PLANETS];
  float old_y[PLANETS];
  float new_x[PLANETS];
  float new_y[PLANETS];
  for (int planet = 0; planet < static_cast<int>(config.planet_count); ++planet) {
    old_x[planet] = game_planets[planet].x;
    old_y[planet] = game_planets[planet].y;
    new_x[planet] = game_planets[planet].x;
    new_y[planet] = game_planets[planet].y;
    const int initial_index = sim_find_planet_index(game_initial, static_cast<int>(config.planet_count), game_planets[planet].id);
    if (initial_index >= 0) {
      const float dx = game_initial[initial_index].x - config.board_center;
      const float dy = game_initial[initial_index].y - config.board_center;
      const float orbital_radius = sqrtf(dx * dx + dy * dy);
      if (orbital_radius + game_planets[planet].radius < config.rotation_radius_limit) {
        const float initial_angle = atan2f(dy, dx);
        const float rotation_step = static_cast<float>(config.step > 1 ? config.step : 1);
        const float current_angle = initial_angle + config.angular_velocity * rotation_step;
        new_x[planet] = config.board_center + orbital_radius * cosf(current_angle);
        new_y[planet] = config.board_center + orbital_radius * sinf(current_angle);
      }
    }
  }

  int arrivals[PLANETS][4];
  for (int planet = 0; planet < PLANETS; ++planet) {
    for (int player = 0; player < 4; ++player) {
      arrivals[planet][player] = 0;
    }
  }

  for (int fleet_index = 0; fleet_index < static_cast<int>(config.max_fleets_per_game); ++fleet_index) {
    OrbitWarsCudaFleet fleet = game_fleets[fleet_index];
    if (!fleet.alive) {
      continue;
    }
    const int owner = fleet.owner;
    const float speed = sim_fleet_speed(fleet.ships, config);
    const float fleet_new_x = fleet.x + cosf(fleet.angle) * speed;
    const float fleet_new_y = fleet.y + sinf(fleet.angle) * speed;
    int hit_planet = -1;
    for (int planet = 0; planet < static_cast<int>(config.planet_count); ++planet) {
      if (sim_swept_pair_hit(
              fleet.x,
              fleet.y,
              fleet_new_x,
              fleet_new_y,
              old_x[planet],
              old_y[planet],
              new_x[planet],
              new_y[planet],
              game_planets[planet].radius)) {
        hit_planet = planet;
        break;
      }
    }
    const int fleet_ships = static_cast<int>(floorf(fleet.ships));
    if (hit_planet >= 0) {
      if (owner >= 0 && owner < static_cast<int>(config.max_players)) {
        game_stats[owner].hit_fleet_count += 1;
        game_stats[owner].hit_ship_count += fleet_ships;
        if (owner < 4) {
          arrivals[hit_planet][owner] += fleet_ships;
        }
      }
      game_fleets[fleet_index].alive = 0;
    } else if (sim_point_out_of_bounds(fleet_new_x, fleet_new_y, config)) {
      if (owner >= 0 && owner < static_cast<int>(config.max_players)) {
        game_stats[owner].out_of_bounds_destroyed_fleet_count += 1;
        game_stats[owner].out_of_bounds_destroyed_ship_count += fleet_ships;
      }
      game_fleets[fleet_index].alive = 0;
    } else if (sim_segment_intersects_sun(fleet.x, fleet.y, fleet_new_x, fleet_new_y, config)) {
      if (owner >= 0 && owner < static_cast<int>(config.max_players)) {
        game_stats[owner].sun_destroyed_fleet_count += 1;
        game_stats[owner].sun_destroyed_ship_count += fleet_ships;
      }
      game_fleets[fleet_index].alive = 0;
    } else {
      game_fleets[fleet_index].x = fleet_new_x;
      game_fleets[fleet_index].y = fleet_new_y;
    }
  }

  for (int planet = 0; planet < static_cast<int>(config.planet_count); ++planet) {
    game_planets[planet].velocity_x = new_x[planet] - old_x[planet];
    game_planets[planet].velocity_y = new_y[planet] - old_y[planet];
    game_planets[planet].x = new_x[planet];
    game_planets[planet].y = new_y[planet];
  }

  for (int planet = 0; planet < static_cast<int>(config.planet_count); ++planet) {
    int largest_owner = -1;
    int largest_ships = 0;
    int second_ships = 0;
    for (int player = 0; player < static_cast<int>(config.max_players) && player < 4; ++player) {
      const int ships = arrivals[planet][player];
      if (ships > largest_ships) {
        second_ships = largest_ships;
        largest_ships = ships;
        largest_owner = player;
      } else if (ships > second_ships) {
        second_ships = ships;
      }
    }
    if (largest_owner < 0 || largest_ships == second_ships) {
      continue;
    }
    const int surviving_attackers = largest_ships - second_ships;
    const int previous_owner = game_planets[planet].owner;
    if (largest_owner == previous_owner) {
      game_planets[planet].ships = floorf(game_planets[planet].ships) + surviving_attackers;
    } else if (surviving_attackers > static_cast<int>(floorf(game_planets[planet].ships))) {
      game_planets[planet].owner = largest_owner;
      game_planets[planet].ships = static_cast<float>(surviving_attackers - static_cast<int>(floorf(game_planets[planet].ships)));
      if (largest_owner >= 0 && largest_owner < static_cast<int>(config.max_players)) {
        game_stats[largest_owner].captured_planet_count += 1;
      }
    } else {
      game_planets[planet].ships = floorf(game_planets[planet].ships) - surviving_attackers;
    }
  }
}

__global__ void sim_prepare_kernel(CudaSimKernelState state, int step) {
  const OrbitWarsCudaSimConfig config = state.config;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stats_total = static_cast<int>(config.game_count * config.max_players);
  const int arrivals_total = static_cast<int>(config.game_count * config.planet_count * config.max_players);
  if (index < stats_total) {
    state.stats[index] = OrbitWarsCudaSimStats{};
  }
  if (index < arrivals_total) {
    state.arrivals[index] = 0;
  }
  if (index < static_cast<int>(config.game_count)) {
    if (state.statuses && state.statuses[index].done) {
      state.fleet_counts[index] = 0;
      return;
    }
    int count = 0;
    OrbitWarsCudaFleet* fleets = state.fleets + static_cast<size_t>(index) * config.max_fleets_per_game;
    for (int fleet = 0; fleet < static_cast<int>(config.max_fleets_per_game); ++fleet) {
      if (fleets[fleet].alive) {
        count = max(count, fleet + 1);
      }
    }
    state.fleet_counts[index] = count;
  }
  if (index == 0) {
    state.config.step = step;
  }
}

__global__ void sim_apply_actions_kernel(CudaSimKernelState state) {
  const OrbitWarsCudaSimConfig config = state.config;
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = static_cast<int>(config.game_count * config.max_players * config.max_actions_per_player);
  if (flat >= total) {
    return;
  }
  const int action_index = flat % static_cast<int>(config.max_actions_per_player);
  const int player = (flat / static_cast<int>(config.max_actions_per_player)) % static_cast<int>(config.max_players);
  const int game = flat / static_cast<int>(config.max_actions_per_player * config.max_players);
  if (state.statuses && state.statuses[game].done) {
    return;
  }
  const int count = state.action_counts[game * config.max_players + player];
  if (action_index >= count) {
    return;
  }
  const OrbitWarsCudaAction action = state.actions[flat];
  if (action.ship_count < 1 || !isfinite(action.direction_angle)) {
    return;
  }
  OrbitWarsCudaPlanet* planets = state.planets + static_cast<size_t>(game) * config.planet_count;
  const int source_index = sim_find_planet_index(planets, static_cast<int>(config.planet_count), action.from_planet_id);
  if (source_index < 0 || planets[source_index].owner != player) {
    return;
  }
  const int available = static_cast<int>(floorf(planets[source_index].ships));
  if (action.ship_count > available) {
    return;
  }
  const int slot = atomicAdd(state.fleet_counts + game, 1);
  OrbitWarsCudaSimStats* stats = state.stats + static_cast<size_t>(game) * config.max_players;
  if (slot >= static_cast<int>(config.max_fleets_per_game)) {
    atomicAdd(&stats[player].overflow_fleet_count, 1);
    return;
  }
  const OrbitWarsCudaPlanet source = planets[source_index];
  atomicAdd(&planets[source_index].ships, -static_cast<float>(action.ship_count));
  OrbitWarsCudaFleet* fleets = state.fleets + static_cast<size_t>(game) * config.max_fleets_per_game;
  const float spawn_distance = source.radius + config.fleet_spawn_offset;
  fleets[slot] = OrbitWarsCudaFleet{
      atomicAdd(state.next_fleet_ids + game, 1),
      player,
      source.x + cosf(action.direction_angle) * spawn_distance,
      source.y + sinf(action.direction_angle) * spawn_distance,
      action.direction_angle,
      action.from_planet_id,
      static_cast<float>(action.ship_count),
      1};
  atomicAdd(&stats[player].launched_fleet_count, 1);
  atomicAdd(&stats[player].launched_ship_count, action.ship_count);
}

__global__ void sim_produce_rotate_kernel(CudaSimKernelState state) {
  const OrbitWarsCudaSimConfig config = state.config;
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = static_cast<int>(config.game_count * config.planet_count);
  if (flat >= total) {
    return;
  }
  const int game = flat / static_cast<int>(config.planet_count);
  if (state.statuses && state.statuses[game].done) {
    return;
  }
  const int planet = flat % static_cast<int>(config.planet_count);
  OrbitWarsCudaPlanet* planets = state.planets + static_cast<size_t>(game) * config.planet_count;
  const OrbitWarsCudaPlanet* initial_planets = state.initial_planets + static_cast<size_t>(game) * config.planet_count;
  OrbitWarsCudaPlanet& item = planets[planet];
  if (item.owner >= 0) {
    item.ships += item.production;
  }
  state.old_x[flat] = item.x;
  state.old_y[flat] = item.y;
  float px = item.x;
  float py = item.y;
  const int initial_index = sim_find_planet_index(initial_planets, static_cast<int>(config.planet_count), item.id);
  if (initial_index >= 0) {
    const float dx = initial_planets[initial_index].x - config.board_center;
    const float dy = initial_planets[initial_index].y - config.board_center;
    const float orbital_radius = sqrtf(dx * dx + dy * dy);
    if (orbital_radius + item.radius < config.rotation_radius_limit) {
    const float initial_angle = atan2f(dy, dx);
    const float rotation_step = static_cast<float>(config.step > 1 ? config.step : 1);
      const float angular_velocity = state.angular_velocities ? state.angular_velocities[game] : config.angular_velocity;
      const float current_angle = initial_angle + angular_velocity * rotation_step;
      px = config.board_center + orbital_radius * cosf(current_angle);
      py = config.board_center + orbital_radius * sinf(current_angle);
    }
  }
  state.new_x[flat] = px;
  state.new_y[flat] = py;
}

__global__ void sim_move_fleets_kernel(CudaSimKernelState state) {
  const OrbitWarsCudaSimConfig config = state.config;
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = static_cast<int>(config.game_count * config.max_fleets_per_game);
  if (flat >= total) {
    return;
  }
  const int game = flat / static_cast<int>(config.max_fleets_per_game);
  if (state.statuses && state.statuses[game].done) {
    return;
  }
  const int fleet_index = flat % static_cast<int>(config.max_fleets_per_game);
  OrbitWarsCudaFleet* fleets = state.fleets + static_cast<size_t>(game) * config.max_fleets_per_game;
  OrbitWarsCudaFleet fleet = fleets[fleet_index];
  if (!fleet.alive) {
    return;
  }
  OrbitWarsCudaPlanet* planets = state.planets + static_cast<size_t>(game) * config.planet_count;
  const float* old_x = state.old_x + static_cast<size_t>(game) * config.planet_count;
  const float* old_y = state.old_y + static_cast<size_t>(game) * config.planet_count;
  const float* new_x = state.new_x + static_cast<size_t>(game) * config.planet_count;
  const float* new_y = state.new_y + static_cast<size_t>(game) * config.planet_count;
  OrbitWarsCudaSimStats* stats = state.stats + static_cast<size_t>(game) * config.max_players;
  const int owner = fleet.owner;
  const float speed = sim_fleet_speed(fleet.ships, config);
  const float fleet_new_x = fleet.x + cosf(fleet.angle) * speed;
  const float fleet_new_y = fleet.y + sinf(fleet.angle) * speed;
  int hit_planet = -1;
  for (int planet = 0; planet < static_cast<int>(config.planet_count); ++planet) {
    if (sim_swept_pair_hit(
            fleet.x,
            fleet.y,
            fleet_new_x,
            fleet_new_y,
            old_x[planet],
            old_y[planet],
            new_x[planet],
            new_y[planet],
            planets[planet].radius)) {
      hit_planet = planet;
      break;
    }
  }
  const int fleet_ships = static_cast<int>(floorf(fleet.ships));
  if (hit_planet >= 0) {
    if (owner >= 0 && owner < static_cast<int>(config.max_players)) {
      atomicAdd(&stats[owner].hit_fleet_count, 1);
      atomicAdd(&stats[owner].hit_ship_count, fleet_ships);
      atomicAdd(
          state.arrivals + ((static_cast<size_t>(game) * config.planet_count + hit_planet) * config.max_players + owner),
          fleet_ships);
    }
    fleets[fleet_index].alive = 0;
  } else if (sim_point_out_of_bounds(fleet_new_x, fleet_new_y, config)) {
    if (owner >= 0 && owner < static_cast<int>(config.max_players)) {
      atomicAdd(&stats[owner].out_of_bounds_destroyed_fleet_count, 1);
      atomicAdd(&stats[owner].out_of_bounds_destroyed_ship_count, fleet_ships);
    }
    fleets[fleet_index].alive = 0;
  } else if (sim_segment_intersects_sun(fleet.x, fleet.y, fleet_new_x, fleet_new_y, config)) {
    if (owner >= 0 && owner < static_cast<int>(config.max_players)) {
      atomicAdd(&stats[owner].sun_destroyed_fleet_count, 1);
      atomicAdd(&stats[owner].sun_destroyed_ship_count, fleet_ships);
    }
    fleets[fleet_index].alive = 0;
  } else {
    fleets[fleet_index].x = fleet_new_x;
    fleets[fleet_index].y = fleet_new_y;
  }
}

__global__ void sim_apply_planets_resolve_kernel(CudaSimKernelState state) {
  const OrbitWarsCudaSimConfig config = state.config;
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = static_cast<int>(config.game_count * config.planet_count);
  if (flat >= total) {
    return;
  }
  const int game = flat / static_cast<int>(config.planet_count);
  if (state.statuses && state.statuses[game].done) {
    return;
  }
  const int planet = flat % static_cast<int>(config.planet_count);
  OrbitWarsCudaPlanet* planets = state.planets + static_cast<size_t>(game) * config.planet_count;
  OrbitWarsCudaPlanet& item = planets[planet];
  item.velocity_x = state.new_x[flat] - state.old_x[flat];
  item.velocity_y = state.new_y[flat] - state.old_y[flat];
  item.x = state.new_x[flat];
  item.y = state.new_y[flat];

  int largest_owner = -1;
  int largest_ships = 0;
  int second_ships = 0;
  for (int player = 0; player < static_cast<int>(config.max_players); ++player) {
    const int ships = state.arrivals[(static_cast<size_t>(game) * config.planet_count + planet) * config.max_players + player];
    if (ships > largest_ships) {
      second_ships = largest_ships;
      largest_ships = ships;
      largest_owner = player;
    } else if (ships > second_ships) {
      second_ships = ships;
    }
  }
  if (largest_owner < 0 || largest_ships == second_ships) {
    return;
  }
  const int surviving_attackers = largest_ships - second_ships;
  const int previous_owner = item.owner;
  const int planet_ships = static_cast<int>(floorf(item.ships));
  if (largest_owner == previous_owner) {
    item.ships = static_cast<float>(planet_ships + surviving_attackers);
  } else if (surviving_attackers > planet_ships) {
    item.owner = largest_owner;
    item.ships = static_cast<float>(surviving_attackers - planet_ships);
    OrbitWarsCudaSimStats* stats = state.stats + static_cast<size_t>(game) * config.max_players;
    atomicAdd(&stats[largest_owner].captured_planet_count, 1);
  } else {
    item.ships = static_cast<float>(planet_ships - surviving_attackers);
  }
}

__global__ void sim_update_status_kernel(CudaSimKernelState state, int step_after) {
  const OrbitWarsCudaSimConfig config = state.config;
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= static_cast<int>(config.game_count) || !state.statuses) {
    return;
  }
  OrbitWarsCudaGameStatus& status = state.statuses[game];
  if (status.done) {
    return;
  }
  bool alive[4] = {false, false, false, false};
  int scores[4] = {0, 0, 0, 0};
  OrbitWarsCudaPlanet* planets = state.planets + static_cast<size_t>(game) * config.planet_count;
  for (int planet = 0; planet < static_cast<int>(config.planet_count); ++planet) {
    const int owner = planets[planet].owner;
    if (owner >= 0 && owner < static_cast<int>(config.max_players) && owner < 4) {
      alive[owner] = true;
      scores[owner] += static_cast<int>(floorf(planets[planet].ships));
    }
  }
  OrbitWarsCudaFleet* fleets = state.fleets + static_cast<size_t>(game) * config.max_fleets_per_game;
  for (int fleet = 0; fleet < static_cast<int>(config.max_fleets_per_game); ++fleet) {
    if (!fleets[fleet].alive) continue;
    const int owner = fleets[fleet].owner;
    if (owner >= 0 && owner < static_cast<int>(config.max_players) && owner < 4) {
      alive[owner] = true;
      scores[owner] += static_cast<int>(floorf(fleets[fleet].ships));
    }
  }
  int alive_count = 0;
  for (int player = 0; player < static_cast<int>(config.max_players) && player < 4; ++player) {
    if (alive[player]) alive_count += 1;
  }
  status.step = step_after;
  if (step_after < config.episode_steps && alive_count > 1) {
    return;
  }
  int best_score = -2147483647;
  int best_player = -1;
  int best_count = 0;
  for (int player = 0; player < static_cast<int>(config.max_players) && player < 4; ++player) {
    if (scores[player] > best_score) {
      best_score = scores[player];
      best_player = player;
      best_count = 1;
    } else if (scores[player] == best_score) {
      best_count += 1;
    }
  }
  status.done = 1;
  status.winner = best_count == 1 ? best_player : -1;
}

__device__ void add_sim_stats(OrbitWarsCudaSimStats& dst, const OrbitWarsCudaSimStats& src) {
  dst.launched_fleet_count += src.launched_fleet_count;
  dst.launched_ship_count += src.launched_ship_count;
  dst.captured_planet_count += src.captured_planet_count;
  dst.hit_fleet_count += src.hit_fleet_count;
  dst.hit_ship_count += src.hit_ship_count;
  dst.out_of_bounds_destroyed_fleet_count += src.out_of_bounds_destroyed_fleet_count;
  dst.out_of_bounds_destroyed_ship_count += src.out_of_bounds_destroyed_ship_count;
  dst.sun_destroyed_fleet_count += src.sun_destroyed_fleet_count;
  dst.sun_destroyed_ship_count += src.sun_destroyed_ship_count;
  dst.overflow_fleet_count += src.overflow_fleet_count;
}

__global__ void sim_accumulate_stats_kernel(CudaSimKernelState state) {
  const OrbitWarsCudaSimConfig config = state.config;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stats_total = static_cast<int>(config.game_count * config.max_players);
  if (index >= stats_total || !state.cumulative_stats) {
    return;
  }
  add_sim_stats(state.cumulative_stats[index], state.stats[index]);
}

cudaError_t sim_allocate_state(CudaSimState* state) {
  const auto& config = state->config;
  cudaError_t error = state->planets.allocate(config.game_count * config.planet_count);
  if (error != cudaSuccess) return error;
  error = state->initial_planets.allocate(config.game_count * config.planet_count);
  if (error != cudaSuccess) return error;
  error = state->fleets.allocate(config.game_count * config.max_fleets_per_game);
  if (error != cudaSuccess) return error;
  error = state->next_fleet_ids.allocate(config.game_count);
  if (error != cudaSuccess) return error;
  error = state->angular_velocities.allocate(config.game_count);
  if (error != cudaSuccess) return error;
  error = state->fleet_counts.allocate(config.game_count);
  if (error != cudaSuccess) return error;
  error = state->actions.allocate(config.game_count * config.max_players * config.max_actions_per_player);
  if (error != cudaSuccess) return error;
  error = state->action_counts.allocate(config.game_count * config.max_players);
  if (error != cudaSuccess) return error;
  error = state->stats.allocate(config.game_count * config.max_players);
  if (error != cudaSuccess) return error;
  error = state->cumulative_stats.allocate(config.game_count * config.max_players);
  if (error != cudaSuccess) return error;
  error = state->statuses.allocate(config.game_count);
  if (error != cudaSuccess) return error;
  error = state->arrivals.allocate(config.game_count * config.planet_count * config.max_players);
  if (error != cudaSuccess) return error;
  error = state->old_x.allocate(config.game_count * config.planet_count);
  if (error != cudaSuccess) return error;
  error = state->old_y.allocate(config.game_count * config.planet_count);
  if (error != cudaSuccess) return error;
  error = state->new_x.allocate(config.game_count * config.planet_count);
  if (error != cudaSuccess) return error;
  return state->new_y.allocate(config.game_count * config.planet_count);
}

}  // namespace

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_status(void) {
  cudaError_t error = cudaFree(nullptr);
  if (error != cudaSuccess) {
    return cuda_error(error);
  }
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_step(
    OrbitWarsCudaPlanet* planets,
    const OrbitWarsCudaPlanet* initial_planets,
    OrbitWarsCudaFleet* fleets,
    int* next_fleet_ids,
    const OrbitWarsCudaAction* actions,
    const int* action_counts,
    OrbitWarsCudaSimStats* stats,
    OrbitWarsCudaSimConfig config) {
  if (!planets || !initial_planets || !fleets || !next_fleet_ids || !actions || !action_counts || !stats) {
    return bad_argument("null sim pointer");
  }
  if (config.game_count == 0 || config.planet_count == 0 || config.planet_count > PLANETS ||
      config.max_fleets_per_game == 0 || config.max_players == 0 || config.max_players > 4 ||
      config.max_actions_per_player == 0) {
    return bad_argument("invalid sim shape");
  }
  const size_t planet_total = config.game_count * config.planet_count;
  const size_t fleet_total = config.game_count * config.max_fleets_per_game;
  const size_t action_total = config.game_count * config.max_players * config.max_actions_per_player;
  const size_t action_count_total = config.game_count * config.max_players;
  const size_t stats_total = config.game_count * config.max_players;

  DeviceBuffer<OrbitWarsCudaPlanet> d_planets;
  DeviceBuffer<OrbitWarsCudaPlanet> d_initial_planets;
  DeviceBuffer<OrbitWarsCudaFleet> d_fleets;
  DeviceBuffer<int> d_next_fleet_ids;
  DeviceBuffer<OrbitWarsCudaAction> d_actions;
  DeviceBuffer<int> d_action_counts;
  DeviceBuffer<OrbitWarsCudaSimStats> d_stats;
  cudaError_t error = d_planets.allocate(planet_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_initial_planets.allocate(planet_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_fleets.allocate(fleet_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_next_fleet_ids.allocate(config.game_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_actions.allocate(action_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_action_counts.allocate(action_count_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_stats.allocate(stats_total);
  if (error != cudaSuccess) return cuda_error(error);

  error = d_planets.copy_from_host(planets, planet_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_initial_planets.copy_from_host(initial_planets, planet_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_fleets.copy_from_host(fleets, fleet_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_next_fleet_ids.copy_from_host(next_fleet_ids, config.game_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_actions.copy_from_host(actions, action_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_action_counts.copy_from_host(action_counts, action_count_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = cudaMemset(d_stats.ptr, 0, sizeof(OrbitWarsCudaSimStats) * stats_total);
  if (error != cudaSuccess) return cuda_error(error);

  sim_step_kernel<<<static_cast<unsigned int>(config.game_count), 1>>>(
      d_planets.ptr,
      d_initial_planets.ptr,
      d_fleets.ptr,
      d_next_fleet_ids.ptr,
      d_actions.ptr,
      d_action_counts.ptr,
      d_stats.ptr,
      config);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return cuda_error(error);

  error = d_planets.copy_to_host(planets, planet_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_fleets.copy_to_host(fleets, fleet_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_next_fleet_ids.copy_to_host(next_fleet_ids, config.game_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_stats.copy_to_host(stats, stats_total);
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_create(
    OrbitWarsCudaSimConfig config,
    OrbitWarsCudaSimState** out_state) {
  if (!out_state) {
    return bad_argument("null sim create output");
  }
  if (config.game_count == 0 || config.planet_count == 0 || config.planet_count > PLANETS ||
      config.max_fleets_per_game == 0 || config.max_players == 0 || config.max_players > 4 ||
      config.max_actions_per_player == 0) {
    return bad_argument("invalid persistent sim shape");
  }
  auto state = std::make_unique<CudaSimState>();
  state->config = config;
  state->resident_token_count = std::max(1 + PLANETS, std::min(resident_compact_token_count(), RESIDENT_TOKEN_COUNT));
  cudaError_t error = sim_allocate_state(state.get());
  if (error != cudaSuccess) {
    return cuda_error(error);
  }
  *out_state = reinterpret_cast<OrbitWarsCudaSimState*>(state.release());
  return ok();
}

extern "C" void orbit_wars_cuda_sim_destroy(OrbitWarsCudaSimState* state) {
  delete reinterpret_cast<CudaSimState*>(state);
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_load(
    OrbitWarsCudaSimState* opaque,
    const OrbitWarsCudaPlanet* planets,
    const OrbitWarsCudaPlanet* initial_planets,
    const OrbitWarsCudaFleet* fleets,
    const int* next_fleet_ids) {
  if (!opaque) {
    return bad_argument("null sim load argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  std::vector<float> angular_velocities(state->config.game_count, state->config.angular_velocity);
  return orbit_wars_cuda_sim_load_with_angular_velocities(
      opaque,
      planets,
      initial_planets,
      fleets,
      next_fleet_ids,
      angular_velocities.data());
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_load_with_angular_velocities(
    OrbitWarsCudaSimState* opaque,
    const OrbitWarsCudaPlanet* planets,
    const OrbitWarsCudaPlanet* initial_planets,
    const OrbitWarsCudaFleet* fleets,
    const int* next_fleet_ids,
    const float* angular_velocities) {
  if (!opaque || !planets || !initial_planets || !fleets || !next_fleet_ids) {
    return bad_argument("null sim load argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  const auto& config = state->config;
  cudaError_t error = state->planets.copy_from_host(planets, config.game_count * config.planet_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->initial_planets.copy_from_host(initial_planets, config.game_count * config.planet_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->fleets.copy_from_host(fleets, config.game_count * config.max_fleets_per_game);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->next_fleet_ids.copy_from_host(next_fleet_ids, config.game_count);
  if (error != cudaSuccess) return cuda_error(error);
  if (angular_velocities) {
    error = state->angular_velocities.copy_from_host(angular_velocities, config.game_count);
    if (error != cudaSuccess) return cuda_error(error);
  } else {
    std::vector<float> defaults(config.game_count, config.angular_velocity);
    error = state->angular_velocities.copy_from_host(defaults.data(), config.game_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  error = cudaMemset(state->stats.ptr, 0, sizeof(OrbitWarsCudaSimStats) * config.game_count * config.max_players);
  if (error != cudaSuccess) return cuda_error(error);
  error = cudaMemset(state->cumulative_stats.ptr, 0, sizeof(OrbitWarsCudaSimStats) * config.game_count * config.max_players);
  if (error != cudaSuccess) return cuda_error(error);
  error = cudaMemset(state->statuses.ptr, 0, sizeof(OrbitWarsCudaGameStatus) * config.game_count);
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_step_persistent(
    OrbitWarsCudaSimState* opaque,
    const OrbitWarsCudaAction* actions,
    const int* action_counts,
    int step) {
  if (!opaque || !actions || !action_counts) {
    return bad_argument("null sim step argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  const auto& config = state->config;
  const size_t action_total = config.game_count * config.max_players * config.max_actions_per_player;
  const size_t action_count_total = config.game_count * config.max_players;
  cudaError_t error = state->actions.copy_from_host(actions, action_total);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->action_counts.copy_from_host(action_counts, action_count_total);
  if (error != cudaSuccess) return cuda_error(error);
  return orbit_wars_cuda_sim_step_device_actions(opaque, step);
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_clear_actions(
    OrbitWarsCudaSimState* opaque) {
  if (!opaque) {
    return bad_argument("null sim clear actions argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  const auto& config = state->config;
  cudaError_t error = cudaMemset(
      state->actions.ptr,
      0,
      sizeof(OrbitWarsCudaAction) * config.game_count * config.max_players * config.max_actions_per_player);
  if (error != cudaSuccess) return cuda_error(error);
  error = cudaMemset(
      state->action_counts.ptr,
      0,
      sizeof(int) * config.game_count * config.max_players);
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_load_request_plan(
    OrbitWarsCudaSimState* opaque,
    const int* request_game_indices,
    const int* request_player_ids,
    size_t request_count) {
  if (!opaque || (!request_game_indices && request_count > 0) || (!request_player_ids && request_count > 0)) {
    return bad_argument("null request plan argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  cudaError_t error = state->request_game_indices.ensure(request_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->request_player_ids.ensure(request_count);
  if (error != cudaSuccess) return cuda_error(error);
  if (request_count > 0) {
    error = state->request_game_indices.copy_from_host(request_game_indices, request_count);
    if (error != cudaSuccess) return cuda_error(error);
    error = state->request_player_ids.copy_from_host(request_player_ids, request_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  state->request_plan_count = request_count;
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_step_device_actions(
    OrbitWarsCudaSimState* opaque,
    int step) {
  if (!opaque) {
    return bad_argument("null sim step device argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  const auto& config = state->config;
  const size_t action_total = config.game_count * config.max_players * config.max_actions_per_player;
  cudaError_t error = cudaSuccess;
  CudaSimKernelState kernel_state = sim_kernel_state(state);
  kernel_state.config.step = step;
  const int stats_total = static_cast<int>(config.game_count * config.max_players);
  const int arrivals_total = static_cast<int>(config.game_count * config.planet_count * config.max_players);
  sim_prepare_kernel<<<blocks_for(std::max(stats_total, arrivals_total)), 256>>>(kernel_state, step);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  sim_apply_actions_kernel<<<blocks_for(static_cast<int>(action_total)), 256>>>(kernel_state);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  sim_produce_rotate_kernel<<<blocks_for(static_cast<int>(config.game_count * config.planet_count)), 256>>>(kernel_state);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  sim_move_fleets_kernel<<<blocks_for(static_cast<int>(config.game_count * config.max_fleets_per_game)), 256>>>(kernel_state);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  sim_apply_planets_resolve_kernel<<<blocks_for(static_cast<int>(config.game_count * config.planet_count)), 256>>>(kernel_state);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  sim_update_status_kernel<<<blocks_for(static_cast<int>(config.game_count)), 256>>>(kernel_state, step + 1);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  sim_accumulate_stats_kernel<<<blocks_for(static_cast<int>(config.game_count * config.max_players)), 256>>>(kernel_state);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

OrbitWarsV8CudaStatus resident_model_decode_device(
    CudaModelState* model_state,
    CudaSimState* sim_state,
    int* request_game_indices_device,
    int* request_player_ids_device,
    size_t request_count,
    int step) {
  if (!model_state || !sim_state || !request_game_indices_device || !request_player_ids_device) {
    return bad_argument("null resident decode argument");
  }
  if (request_count == 0) {
    return ok();
  }
  ForwardWorkspace& workspace = model_state->workspace;
  cudaError_t error = cudaSuccess;
#define ENSURE(buffer, count) \
  do { error = (buffer).ensure(count); if (error != cudaSuccess) return cuda_error(error); } while (0)
  ENSURE(workspace.d_tokens, request_count * RESIDENT_TOKEN_COUNT * TOKEN_FEATURES);
  ENSURE(workspace.d_token_type_ids, request_count * RESIDENT_TOKEN_COUNT);
  ENSURE(workspace.d_owner_ids, request_count * RESIDENT_TOKEN_COUNT);
  ENSURE(workspace.d_padding_mask, request_count * RESIDENT_TOKEN_COUNT);
  ENSURE(workspace.d_planet_mask, request_count * PLANETS);
  ENSURE(workspace.d_action_labels, request_count);
#undef ENSURE
  workspace.last_request_count = request_count;
  CudaSimKernelState kernel_state = sim_kernel_state(sim_state);
  kernel_state.config.step = step;
  resident_build_tokens_kernel<<<blocks_for(static_cast<int>(request_count * RESIDENT_TOKEN_COUNT)), 256>>>(
      kernel_state,
      request_game_indices_device,
      request_player_ids_device,
      static_cast<int>(request_count),
      step,
      workspace.d_tokens.ptr,
      workspace.d_token_type_ids.ptr,
      workspace.d_owner_ids.ptr,
      workspace.d_padding_mask.ptr,
      workspace.d_planet_mask.ptr);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  OrbitWarsV8CudaShape shape{
      request_count,
      RESIDENT_TOKEN_COUNT,
      TOKEN_FEATURES,
      D_MODEL,
      HEADS,
      ENCODER_LAYERS,
      DECODER_LAYERS,
      ACTION_SLOTS,
      PLANETS,
      AMOUNTS};
  OrbitWarsV8CudaStatus status = forward_workspace_with_tensor_map(model_state->tensor_map, workspace, shape);
  if (status.code != CUDA_STATUS_OK) {
    return status;
  }
  resident_decode_actions_kernel<<<blocks_for(static_cast<int>(request_count)), 256>>>(
      kernel_state,
      request_game_indices_device,
      request_player_ids_device,
      static_cast<int>(request_count),
      workspace.d_fire.ptr,
      workspace.d_source.ptr,
      workspace.d_target.ptr,
      workspace.d_amount.ptr,
      workspace.d_action_labels.ptr);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_resident_model_decode(
    OrbitWarsV8CudaModel* model,
    OrbitWarsCudaSimState* opaque,
    const int* request_game_indices,
    const int* request_player_ids,
    size_t request_count,
    int step) {
  auto* model_state = reinterpret_cast<CudaModelState*>(model);
  auto* sim_state = reinterpret_cast<CudaSimState*>(opaque);
  if (!model_state || !sim_state || !request_game_indices || !request_player_ids) {
    return bad_argument("null resident decode argument");
  }
  if (request_count == 0) {
    return ok();
  }
  ForwardWorkspace& workspace = model_state->workspace;
  cudaError_t error = workspace.request_game_indices.ensure(request_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = workspace.request_player_ids.ensure(request_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = workspace.request_game_indices.copy_from_host(request_game_indices, request_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = workspace.request_player_ids.copy_from_host(request_player_ids, request_count);
  if (error != cudaSuccess) return cuda_error(error);
  return resident_model_decode_device(
      model_state,
      sim_state,
      workspace.request_game_indices.ptr,
      workspace.request_player_ids.ptr,
      request_count,
      step);
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_resident_models_decode(
    OrbitWarsV8CudaModel** models,
    size_t model_count,
    OrbitWarsCudaSimState* opaque,
    const int* request_offsets,
    const int* request_counts,
    const int* request_game_indices,
    const int* request_player_ids,
    size_t request_total,
    int step) {
  if (!models || !opaque || !request_offsets || !request_counts ||
      !request_game_indices || !request_player_ids) {
    return bad_argument("null resident models decode argument");
  }
  for (size_t model_index = 0; model_index < model_count; ++model_index) {
    if (!models[model_index]) {
      return bad_argument("null resident model entry");
    }
    const int offset = request_offsets[model_index];
    const int count = request_counts[model_index];
    if (offset < 0 || count < 0 || static_cast<size_t>(offset + count) > request_total) {
      return bad_argument("resident models request range out of bounds");
    }
    if (count == 0) {
      continue;
    }
    OrbitWarsV8CudaStatus status = orbit_wars_cuda_v8_resident_model_decode(
        models[model_index],
        opaque,
        request_game_indices + offset,
        request_player_ids + offset,
        static_cast<size_t>(count),
        step);
    if (status.code != CUDA_STATUS_OK) {
      return status;
    }
  }
  return ok();
}

OrbitWarsV8CudaStatus resident_models_decode_plan_impl(
    OrbitWarsV8CudaModel** models,
    size_t model_count,
    OrbitWarsCudaSimState* opaque,
    const int* request_offsets,
    const int* request_counts,
    size_t request_total,
    int step,
    bool force_full_tokens) {
  if (!models || !opaque || !request_offsets || !request_counts) {
    return bad_argument("null resident models decode plan argument");
  }
  auto* sim_state = reinterpret_cast<CudaSimState*>(opaque);
  if (request_total > sim_state->request_plan_count) {
    return bad_argument("resident request plan range out of bounds");
  }
  const int resident_token_count = force_full_tokens ? RESIDENT_TOKEN_COUNT : sim_state->resident_token_count;
  std::vector<CudaModelState*> active_models;
  std::vector<int> active_offsets;
  std::vector<int> active_counts;
  active_models.reserve(model_count);
  active_offsets.reserve(model_count);
  active_counts.reserve(model_count);
  for (size_t model_index = 0; model_index < model_count; ++model_index) {
    auto* model_state = reinterpret_cast<CudaModelState*>(models[model_index]);
    if (!model_state) {
      return bad_argument("null resident model entry");
    }
    const int offset = request_offsets[model_index];
    const int count = request_counts[model_index];
    if (offset < 0 || count < 0 || static_cast<size_t>(offset + count) > request_total) {
      return bad_argument("resident models request plan range out of bounds");
    }
    if (count == 0) {
      continue;
    }
    ForwardWorkspace& workspace = model_state->workspace;
    cudaError_t error = cudaSuccess;
#define ENSURE_RESIDENT(buffer, element_count) \
    do { error = (buffer).ensure(element_count); if (error != cudaSuccess) return cuda_error(error); } while (0)
    ENSURE_RESIDENT(workspace.d_tokens, static_cast<size_t>(count) * RESIDENT_TOKEN_COUNT * TOKEN_FEATURES);
    ENSURE_RESIDENT(workspace.d_token_type_ids, static_cast<size_t>(count) * RESIDENT_TOKEN_COUNT);
    ENSURE_RESIDENT(workspace.d_owner_ids, static_cast<size_t>(count) * RESIDENT_TOKEN_COUNT);
    ENSURE_RESIDENT(workspace.d_padding_mask, static_cast<size_t>(count) * RESIDENT_TOKEN_COUNT);
    ENSURE_RESIDENT(workspace.d_planet_mask, static_cast<size_t>(count) * PLANETS);
    ENSURE_RESIDENT(workspace.d_action_labels, static_cast<size_t>(count));
#undef ENSURE_RESIDENT
    workspace.last_request_count = static_cast<size_t>(count);
    workspace.last_token_count = static_cast<size_t>(resident_token_count);
    const int token_rows = count * resident_token_count;
    const int slot_rows = count * ACTION_SLOTS;
#define ENSURE_RESIDENT_BODY(buffer, element_count) \
    do { error = (buffer).ensure(element_count); if (error != cudaSuccess) return cuda_error(error); } while (0)
    ENSURE_RESIDENT_BODY(workspace.hidden, static_cast<size_t>(token_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.norm_token, static_cast<size_t>(token_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.token_attn, static_cast<size_t>(token_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.token_q, static_cast<size_t>(token_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.token_k, static_cast<size_t>(token_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.token_v, static_cast<size_t>(token_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.token_qkv, static_cast<size_t>(token_rows) * D_MODEL * 3);
    ENSURE_RESIDENT_BODY(workspace.token_context, static_cast<size_t>(token_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.token_ff_mid, static_cast<size_t>(token_rows) * FF_DIM);
    ENSURE_RESIDENT_BODY(workspace.token_ff_out, static_cast<size_t>(token_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.slots, static_cast<size_t>(slot_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.norm_slot, static_cast<size_t>(slot_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.slot_attn, static_cast<size_t>(slot_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.slot_q, static_cast<size_t>(slot_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.slot_k, static_cast<size_t>(std::max(token_rows, slot_rows)) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.slot_v, static_cast<size_t>(std::max(token_rows, slot_rows)) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.slot_qkv, static_cast<size_t>(std::max(token_rows * 2, slot_rows * 3)) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.slot_context, static_cast<size_t>(slot_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.slot_ff_mid, static_cast<size_t>(slot_rows) * FF_DIM);
    ENSURE_RESIDENT_BODY(workspace.slot_ff_out, static_cast<size_t>(slot_rows) * D_MODEL);
    ENSURE_RESIDENT_BODY(workspace.d_fire, static_cast<size_t>(slot_rows));
    ENSURE_RESIDENT_BODY(workspace.d_source, static_cast<size_t>(slot_rows) * PLANETS);
    ENSURE_RESIDENT_BODY(workspace.d_target, static_cast<size_t>(slot_rows) * PLANETS);
    ENSURE_RESIDENT_BODY(workspace.d_amount, static_cast<size_t>(slot_rows) * AMOUNTS);
#undef ENSURE_RESIDENT_BODY
    CudaSimKernelState kernel_state = sim_kernel_state(sim_state);
    kernel_state.config.step = step;
    resident_build_tokens_kernel<<<blocks_for(count * resident_token_count), 256>>>(
        kernel_state,
        sim_state->request_game_indices.ptr + offset,
        sim_state->request_player_ids.ptr + offset,
        count,
        step,
        workspace.d_tokens.ptr,
        workspace.d_token_type_ids.ptr,
        workspace.d_owner_ids.ptr,
        workspace.d_padding_mask.ptr,
        workspace.d_planet_mask.ptr);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_error(error);
    const PackedModelWeights& weights = model_state->weights;
    if (!weights.valid) return bad_argument("invalid packed resident model weights");
    if (!weights.token_projection_weight || !weights.token_projection_bias || !weights.type_embedding || !weights.owner_embedding) {
      return bad_argument("missing grouped body token projection tensor");
    }
    token_projection_kernel<<<blocks_for(token_rows * D_MODEL), 256>>>(
        workspace.d_tokens.ptr,
        workspace.d_token_type_ids.ptr,
        workspace.d_owner_ids.ptr,
        weights.token_projection_weight,
        weights.token_projection_bias,
        weights.type_embedding,
        weights.owner_embedding,
        workspace.hidden.ptr,
        count,
        resident_token_count);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_error(error);
    active_models.push_back(model_state);
    active_offsets.push_back(offset);
    active_counts.push_back(count);
  }
  cudaError_t error = cudaSuccess;
  for (int layer = 0; layer < ENCODER_LAYERS; ++layer) {
    error = grouped_encoder_layer(sim_state, active_models, active_counts, layer, resident_token_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  for (size_t index = 0; index < active_models.size(); ++index) {
    ForwardWorkspace& workspace = active_models[index]->workspace;
    const float* slot_queries = active_models[index]->weights.slot_queries;
    if (!slot_queries) return bad_argument("missing grouped body slot query tensor");
    const int slot_rows = active_counts[index] * ACTION_SLOTS;
    copy_slot_queries_kernel<<<blocks_for(slot_rows * D_MODEL), 256>>>(
        slot_queries,
        workspace.slots.ptr,
        active_counts[index]);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_error(error);
  }
  for (int layer = 0; layer < DECODER_LAYERS; ++layer) {
    error = grouped_decoder_layer(sim_state, active_models, active_counts, layer, resident_token_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  std::vector<const float*> head_inputs;
  std::vector<const float*> head_weights;
  std::vector<const float*> head_biases;
  std::vector<float*> head_outputs;
  std::vector<int> head_rows;
  auto prepare_grouped_head = [&](const char* weight_name, const char* bias_name, auto output_ptr) -> OrbitWarsV8CudaStatus {
    head_inputs.clear();
    head_weights.clear();
    head_biases.clear();
    head_outputs.clear();
    head_rows.clear();
    for (size_t index = 0; index < active_models.size(); ++index) {
      const PackedModelWeights& packed = active_models[index]->weights;
      const float* weight = nullptr;
      const float* bias = nullptr;
      if (std::strcmp(weight_name, "source_head.weight") == 0) {
        weight = packed.source_head_weight;
        bias = packed.source_head_bias;
      } else if (std::strcmp(weight_name, "target_head.weight") == 0) {
        weight = packed.target_head_weight;
        bias = packed.target_head_bias;
      } else if (std::strcmp(weight_name, "amount_head.weight") == 0) {
        weight = packed.amount_head_weight;
        bias = packed.amount_head_bias;
      }
      if (!weight || !bias) return bad_argument("missing grouped head tensor");
      head_inputs.push_back(active_models[index]->workspace.slots.ptr);
      head_weights.push_back(weight);
      head_biases.push_back(bias);
      head_outputs.push_back(output_ptr(active_models[index]->workspace));
      head_rows.push_back(active_counts[index] * ACTION_SLOTS);
    }
    return ok();
  };
  OrbitWarsV8CudaStatus grouped_status = prepare_grouped_head(
      "source_head.weight",
      "source_head.bias",
      [](ForwardWorkspace& workspace) { return workspace.d_source.ptr; });
  if (grouped_status.code != CUDA_STATUS_OK) return grouped_status;
  error = launch_grouped_cutlass_linear(sim_state, head_inputs, head_weights, head_outputs, head_rows, D_MODEL, PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  for (size_t index = 0; index < active_models.size(); ++index) {
    add_linear_bias_kernel<<<blocks_for(head_rows[index] * PLANETS), 256>>>(
        active_models[index]->workspace.d_source.ptr,
        head_biases[index],
        head_rows[index],
        PLANETS);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_error(error);
  }
  grouped_status = prepare_grouped_head(
      "target_head.weight",
      "target_head.bias",
      [](ForwardWorkspace& workspace) { return workspace.d_target.ptr; });
  if (grouped_status.code != CUDA_STATUS_OK) return grouped_status;
  error = launch_grouped_cutlass_linear(sim_state, head_inputs, head_weights, head_outputs, head_rows, D_MODEL, PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  for (size_t index = 0; index < active_models.size(); ++index) {
    add_linear_bias_kernel<<<blocks_for(head_rows[index] * PLANETS), 256>>>(
        active_models[index]->workspace.d_target.ptr,
        head_biases[index],
        head_rows[index],
        PLANETS);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_error(error);
  }
  grouped_status = prepare_grouped_head(
      "amount_head.weight",
      "amount_head.bias",
      [](ForwardWorkspace& workspace) { return workspace.d_amount.ptr; });
  if (grouped_status.code != CUDA_STATUS_OK) return grouped_status;
  error = launch_grouped_cutlass_linear(sim_state, head_inputs, head_weights, head_outputs, head_rows, D_MODEL, AMOUNTS);
  if (error != cudaSuccess) return cuda_error(error);
  for (size_t index = 0; index < active_models.size(); ++index) {
    add_linear_bias_kernel<<<blocks_for(head_rows[index] * AMOUNTS), 256>>>(
        active_models[index]->workspace.d_amount.ptr,
        head_biases[index],
        head_rows[index],
        AMOUNTS);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_error(error);
  }
  for (size_t index = 0; index < active_models.size(); ++index) {
    ForwardWorkspace& workspace = active_models[index]->workspace;
    const float* fire_weight = active_models[index]->weights.fire_head_weight;
    const float* fire_bias = active_models[index]->weights.fire_head_bias;
    if (!fire_weight || !fire_bias) return bad_argument("missing grouped fire head tensor");
    const int count = active_counts[index];
    const int offset = active_offsets[index];
    const int slot_rows = count * ACTION_SLOTS;
    error = launch_linear(workspace.slots.ptr, fire_weight, fire_bias, workspace.d_fire.ptr, slot_rows, D_MODEL, 1);
    if (error != cudaSuccess) return cuda_error(error);
    mask_planet_logits_kernel<<<blocks_for(slot_rows * PLANETS), 256>>>(workspace.d_source.ptr, workspace.d_planet_mask.ptr, count);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_error(error);
    mask_planet_logits_kernel<<<blocks_for(slot_rows * PLANETS), 256>>>(workspace.d_target.ptr, workspace.d_planet_mask.ptr, count);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_error(error);
    CudaSimKernelState kernel_state = sim_kernel_state(sim_state);
    kernel_state.config.step = step;
    resident_decode_actions_kernel<<<blocks_for(count), 256>>>(
        kernel_state,
        sim_state->request_game_indices.ptr + offset,
        sim_state->request_player_ids.ptr + offset,
        count,
        workspace.d_fire.ptr,
        workspace.d_source.ptr,
        workspace.d_target.ptr,
        workspace.d_amount.ptr,
        workspace.d_action_labels.ptr);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_error(error);
  }
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_resident_models_decode_plan(
    OrbitWarsV8CudaModel** models,
    size_t model_count,
    OrbitWarsCudaSimState* opaque,
    const int* request_offsets,
    const int* request_counts,
    size_t request_total,
    int step) {
  return resident_models_decode_plan_impl(
      models,
      model_count,
      opaque,
      request_offsets,
      request_counts,
      request_total,
      step,
      false);
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_resident_models_step_plan(
    OrbitWarsV8CudaModel** models,
    size_t model_count,
    OrbitWarsCudaSimState* opaque,
    const int* request_offsets,
    const int* request_counts,
    size_t request_total,
    int step) {
  OrbitWarsV8CudaStatus status = resident_models_decode_plan_impl(
      models,
      model_count,
      opaque,
      request_offsets,
      request_counts,
      request_total,
      step,
      false);
  if (status.code != CUDA_STATUS_OK) {
    return status;
  }
  return orbit_wars_cuda_sim_step_device_actions(opaque, step);
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_read_last_batch(
    OrbitWarsV8CudaModel* model,
    float* tokens,
    long long* token_type_ids,
    long long* owner_ids,
    unsigned char* padding_mask,
    unsigned char* planet_mask,
    OrbitWarsCudaActionLabel* labels,
    size_t request_capacity,
    size_t* out_request_count) {
  auto* model_state = reinterpret_cast<CudaModelState*>(model);
  if (!model_state || !tokens || !token_type_ids || !owner_ids || !padding_mask ||
      !planet_mask || !labels || !out_request_count) {
    return bad_argument("null read last batch argument");
  }
  ForwardWorkspace& workspace = model_state->workspace;
  const size_t request_count = workspace.last_request_count;
  *out_request_count = request_count;
  if (request_count == 0) {
    return ok();
  }
  if (request_capacity < request_count) {
    return bad_argument("read last batch capacity too small");
  }
  const size_t token_count = std::max<size_t>(1 + PLANETS, std::min<size_t>(workspace.last_token_count, RESIDENT_TOKEN_COUNT));
  const size_t compact_token_values = request_count * token_count * TOKEN_FEATURES;
  const size_t compact_meta_values = request_count * token_count;
  const size_t full_token_values = request_count * RESIDENT_TOKEN_COUNT * TOKEN_FEATURES;
  const size_t full_meta_values = request_count * RESIDENT_TOKEN_COUNT;
  std::vector<float> compact_tokens(compact_token_values);
  std::vector<long long> compact_token_type_ids(compact_meta_values);
  std::vector<long long> compact_owner_ids(compact_meta_values);
  std::vector<unsigned char> compact_padding_mask(compact_meta_values);
  cudaError_t error = workspace.d_tokens.copy_to_host(compact_tokens.data(), compact_token_values);
  if (error != cudaSuccess) return cuda_error(error);
  error = workspace.d_token_type_ids.copy_to_host(compact_token_type_ids.data(), compact_meta_values);
  if (error != cudaSuccess) return cuda_error(error);
  error = workspace.d_owner_ids.copy_to_host(compact_owner_ids.data(), compact_meta_values);
  if (error != cudaSuccess) return cuda_error(error);
  error = workspace.d_padding_mask.copy_to_host(compact_padding_mask.data(), compact_meta_values);
  if (error != cudaSuccess) return cuda_error(error);
  std::fill(tokens, tokens + full_token_values, 0.0f);
  std::fill(token_type_ids, token_type_ids + full_meta_values, 0);
  std::fill(owner_ids, owner_ids + full_meta_values, 0);
  std::fill(padding_mask, padding_mask + full_meta_values, static_cast<unsigned char>(1));
  for (size_t request = 0; request < request_count; ++request) {
    const size_t compact_meta_base = request * token_count;
    const size_t full_meta_base = request * RESIDENT_TOKEN_COUNT;
    std::copy(
        compact_tokens.begin() + compact_meta_base * TOKEN_FEATURES,
        compact_tokens.begin() + (compact_meta_base + token_count) * TOKEN_FEATURES,
        tokens + full_meta_base * TOKEN_FEATURES);
    std::copy(
        compact_token_type_ids.begin() + compact_meta_base,
        compact_token_type_ids.begin() + compact_meta_base + token_count,
        token_type_ids + full_meta_base);
    std::copy(
        compact_owner_ids.begin() + compact_meta_base,
        compact_owner_ids.begin() + compact_meta_base + token_count,
        owner_ids + full_meta_base);
    std::copy(
        compact_padding_mask.begin() + compact_meta_base,
        compact_padding_mask.begin() + compact_meta_base + token_count,
        padding_mask + full_meta_base);
  }
  error = workspace.d_planet_mask.copy_to_host(
      planet_mask,
      request_count * PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = workspace.d_action_labels.copy_to_host(labels, request_count);
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_last_batch_device_view(
    OrbitWarsV8CudaModel* model,
    OrbitWarsV8CudaDeviceBatchView* out_view) {
  auto* model_state = reinterpret_cast<CudaModelState*>(model);
  if (!model_state || !out_view) {
    return bad_argument("null last batch device view argument");
  }
  ForwardWorkspace& workspace = model_state->workspace;
  const size_t request_count = workspace.last_request_count;
  const size_t token_count = std::max<size_t>(
      1 + PLANETS,
      std::min<size_t>(workspace.last_token_count, RESIDENT_TOKEN_COUNT));
  *out_view = OrbitWarsV8CudaDeviceBatchView{
      workspace.d_tokens.ptr,
      workspace.d_token_type_ids.ptr,
      workspace.d_owner_ids.ptr,
      workspace.d_padding_mask.ptr,
      workspace.d_planet_mask.ptr,
      workspace.d_action_labels.ptr,
      request_count,
      token_count,
      TOKEN_FEATURES,
      PLANETS,
      ACTION_SLOTS};
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_read(
    OrbitWarsCudaSimState* opaque,
    OrbitWarsCudaPlanet* planets,
    OrbitWarsCudaFleet* fleets,
    int* next_fleet_ids,
    OrbitWarsCudaSimStats* stats) {
  if (!opaque || !planets || !fleets || !next_fleet_ids || !stats) {
    return bad_argument("null sim read argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  const auto& config = state->config;
  cudaError_t error = state->planets.copy_to_host(planets, config.game_count * config.planet_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->fleets.copy_to_host(fleets, config.game_count * config.max_fleets_per_game);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->next_fleet_ids.copy_to_host(next_fleet_ids, config.game_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->cumulative_stats.copy_to_host(stats, config.game_count * config.max_players);
  if (error != cudaSuccess) return cuda_error(error);
  error = cudaMemset(state->cumulative_stats.ptr, 0, sizeof(OrbitWarsCudaSimStats) * config.game_count * config.max_players);
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_read_planets_stats(
    OrbitWarsCudaSimState* opaque,
    OrbitWarsCudaPlanet* planets,
    int* next_fleet_ids,
    OrbitWarsCudaSimStats* stats) {
  if (!opaque || !planets || !next_fleet_ids || !stats) {
    return bad_argument("null sim read planets stats argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  const auto& config = state->config;
  cudaError_t error = state->planets.copy_to_host(planets, config.game_count * config.planet_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->next_fleet_ids.copy_to_host(next_fleet_ids, config.game_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->cumulative_stats.copy_to_host(stats, config.game_count * config.max_players);
  if (error != cudaSuccess) return cuda_error(error);
  error = cudaMemset(state->cumulative_stats.ptr, 0, sizeof(OrbitWarsCudaSimStats) * config.game_count * config.max_players);
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_read_status_stats(
    OrbitWarsCudaSimState* opaque,
    OrbitWarsCudaGameStatus* statuses,
    OrbitWarsCudaSimStats* stats) {
  if (!opaque || !statuses || !stats) {
    return bad_argument("null persistent sim status read argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  const auto& config = state->config;
  cudaError_t error = state->statuses.copy_to_host(statuses, config.game_count);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->stats.copy_to_host(stats, config.game_count * config.max_players);
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_sim_read_actions(
    OrbitWarsCudaSimState* opaque,
    OrbitWarsCudaAction* actions,
    int* action_counts) {
  if (!opaque || !actions || !action_counts) {
    return bad_argument("null sim read actions argument");
  }
  auto* state = reinterpret_cast<CudaSimState*>(opaque);
  const auto& config = state->config;
  cudaError_t error = state->actions.copy_to_host(
      actions,
      config.game_count * config.max_players * config.max_actions_per_player);
  if (error != cudaSuccess) return cuda_error(error);
  error = state->action_counts.copy_to_host(action_counts, config.game_count * config.max_players);
  if (error != cudaSuccess) return cuda_error(error);
  return ok();
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_model_create(
    const OrbitWarsV8CudaTensor* tensors,
    size_t tensor_count,
    OrbitWarsV8CudaModel** out_model) {
  if (!tensors || tensor_count == 0 || !out_model) {
    return bad_argument("null cuda model create argument");
  }
  auto model = std::make_unique<CudaModelState>();
  model->owned_tensors.reserve(tensor_count);
  for (size_t index = 0; index < tensor_count; ++index) {
    if (!tensors[index].name || !tensors[index].data || tensors[index].len == 0) {
      return bad_argument("bad cuda tensor entry");
    }
    auto buffer = std::make_unique<DeviceBuffer<float>>();
    cudaError_t error = buffer->allocate(tensors[index].len);
    if (error != cudaSuccess) return cuda_error(error);
    error = buffer->copy_from_host(tensors[index].data, tensors[index].len);
    if (error != cudaSuccess) return cuda_error(error);
    model->tensor_map.emplace(
        std::string(tensors[index].name),
        DeviceTensor{buffer->ptr, tensors[index].len});
    model->owned_tensors.emplace_back(std::move(buffer));
  }
  if (!build_packed_weights(model->tensor_map, &model->weights)) {
    return bad_argument("missing required packed OWV8 tensor");
  }
  *out_model = reinterpret_cast<OrbitWarsV8CudaModel*>(model.release());
  return ok();
}

extern "C" void orbit_wars_cuda_v8_model_destroy(OrbitWarsV8CudaModel* model) {
  delete reinterpret_cast<CudaModelState*>(model);
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_forward_cached(
    OrbitWarsV8CudaModel* model,
    const float* tokens,
    const long long* token_type_ids,
    const long long* owner_ids,
    const unsigned char* padding_mask,
    const unsigned char* planet_mask,
    float* fire_logits,
    float* source_logits,
    float* target_logits,
    float* amount_logits,
    OrbitWarsV8CudaShape shape) {
  auto* state = reinterpret_cast<CudaModelState*>(model);
  if (!state || !tokens || !token_type_ids || !owner_ids || !padding_mask || !planet_mask ||
      !fire_logits || !source_logits || !target_logits || !amount_logits) {
    return bad_argument("null cached v8 forward buffer");
  }
  if (!valid_shape(shape)) {
    return bad_argument("unsupported OWV8 CUDA shape");
  }
  return forward_with_tensor_map(
      state->tensor_map, tokens, token_type_ids, owner_ids, padding_mask, planet_mask,
      fire_logits, source_logits, target_logits, amount_logits, shape, &state->workspace);
}

extern "C" OrbitWarsV8CudaStatus orbit_wars_cuda_v8_forward(
    const float* tokens,
    const long long* token_type_ids,
    const long long* owner_ids,
    const unsigned char* padding_mask,
    const unsigned char* planet_mask,
    const OrbitWarsV8CudaTensor* tensors,
    size_t tensor_count,
    float* fire_logits,
    float* source_logits,
    float* target_logits,
    float* amount_logits,
    OrbitWarsV8CudaShape shape) {
  if (!tokens || !token_type_ids || !owner_ids || !padding_mask || !planet_mask ||
      !tensors || tensor_count == 0 || !fire_logits || !source_logits ||
      !target_logits || !amount_logits) {
    return bad_argument("null v8 forward buffer");
  }
  if (!valid_shape(shape)) {
    return bad_argument("unsupported OWV8 CUDA shape");
  }
  const int batch_count = static_cast<int>(shape.batch_count);
  const int token_count = static_cast<int>(shape.token_count);
  const int token_rows = batch_count * token_count;
  const int slot_rows = batch_count * ACTION_SLOTS;

  std::vector<std::unique_ptr<DeviceBuffer<float>>> owned_tensors;
  TensorMap tensor_map;
  owned_tensors.reserve(tensor_count);
  for (size_t index = 0; index < tensor_count; ++index) {
    if (!tensors[index].name || !tensors[index].data || tensors[index].len == 0) {
      return bad_argument("bad cuda tensor entry");
    }
    auto buffer = std::make_unique<DeviceBuffer<float>>();
    cudaError_t error = buffer->allocate(tensors[index].len);
    if (error != cudaSuccess) return cuda_error(error);
    error = buffer->copy_from_host(tensors[index].data, tensors[index].len);
    if (error != cudaSuccess) return cuda_error(error);
    tensor_map.emplace(
        std::string(tensors[index].name),
        DeviceTensor{buffer->ptr, tensors[index].len});
    owned_tensors.emplace_back(std::move(buffer));
  }

  const float* token_projection_weight = tensor_ptr(tensor_map, "token_projection.weight");
  const float* token_projection_bias = tensor_ptr(tensor_map, "token_projection.bias");
  const float* type_embedding = tensor_ptr(tensor_map, "type_embedding.weight");
  const float* owner_embedding = tensor_ptr(tensor_map, "owner_embedding.weight");
  const float* slot_queries = tensor_ptr(tensor_map, "slot_queries");
  if (!token_projection_weight || !token_projection_bias || !type_embedding || !owner_embedding || !slot_queries) {
    return bad_argument("missing required OWV8 tensor");
  }
  debug_stage("tensors_ready");

  DeviceBuffer<float> d_tokens;
  DeviceBuffer<long long> d_token_type_ids;
  DeviceBuffer<long long> d_owner_ids;
  DeviceBuffer<unsigned char> d_padding_mask;
  DeviceBuffer<unsigned char> d_planet_mask;
  DeviceBuffer<float> hidden;
  DeviceBuffer<float> norm_token;
  DeviceBuffer<float> token_attn;
  DeviceBuffer<float> token_q;
  DeviceBuffer<float> token_k;
  DeviceBuffer<float> token_v;
  DeviceBuffer<float> token_qkv;
  DeviceBuffer<float> token_context;
  DeviceBuffer<float> token_ff_mid;
  DeviceBuffer<float> token_ff_out;
  DeviceBuffer<float> slots;
  DeviceBuffer<float> norm_slot;
  DeviceBuffer<float> slot_attn;
  DeviceBuffer<float> slot_q;
  DeviceBuffer<float> slot_k;
  DeviceBuffer<float> slot_v;
  DeviceBuffer<float> slot_qkv;
  DeviceBuffer<float> slot_context;
  DeviceBuffer<float> slot_ff_mid;
  DeviceBuffer<float> slot_ff_out;
  DeviceBuffer<float> d_fire;
  DeviceBuffer<float> d_source;
  DeviceBuffer<float> d_target;
  DeviceBuffer<float> d_amount;

  cudaError_t error = cudaSuccess;
#define ALLOC(buffer, count) \
  do { error = (buffer).allocate(count); if (error != cudaSuccess) return cuda_error(error); } while (0)
  ALLOC(d_tokens, static_cast<size_t>(token_rows) * TOKEN_FEATURES);
  ALLOC(d_token_type_ids, token_rows);
  ALLOC(d_owner_ids, token_rows);
  ALLOC(d_padding_mask, token_rows);
  ALLOC(d_planet_mask, static_cast<size_t>(batch_count) * PLANETS);
  ALLOC(hidden, static_cast<size_t>(token_rows) * D_MODEL);
  ALLOC(norm_token, static_cast<size_t>(token_rows) * D_MODEL);
  ALLOC(token_attn, static_cast<size_t>(token_rows) * D_MODEL);
  ALLOC(token_q, static_cast<size_t>(token_rows) * D_MODEL);
  ALLOC(token_k, static_cast<size_t>(token_rows) * D_MODEL);
  ALLOC(token_v, static_cast<size_t>(token_rows) * D_MODEL);
  ALLOC(token_qkv, static_cast<size_t>(token_rows) * D_MODEL * 3);
  ALLOC(token_context, static_cast<size_t>(token_rows) * D_MODEL);
  ALLOC(token_ff_mid, static_cast<size_t>(token_rows) * FF_DIM);
  ALLOC(token_ff_out, static_cast<size_t>(token_rows) * D_MODEL);
  ALLOC(slots, static_cast<size_t>(slot_rows) * D_MODEL);
  ALLOC(norm_slot, static_cast<size_t>(slot_rows) * D_MODEL);
  ALLOC(slot_attn, static_cast<size_t>(slot_rows) * D_MODEL);
  ALLOC(slot_q, static_cast<size_t>(slot_rows) * D_MODEL);
  ALLOC(slot_k, static_cast<size_t>(std::max(token_rows, slot_rows)) * D_MODEL);
  ALLOC(slot_v, static_cast<size_t>(std::max(token_rows, slot_rows)) * D_MODEL);
  ALLOC(slot_qkv, static_cast<size_t>(std::max(token_rows * 2, slot_rows * 3)) * D_MODEL);
  ALLOC(slot_context, static_cast<size_t>(slot_rows) * D_MODEL);
  ALLOC(slot_ff_mid, static_cast<size_t>(slot_rows) * FF_DIM);
  ALLOC(slot_ff_out, static_cast<size_t>(slot_rows) * D_MODEL);
  ALLOC(d_fire, static_cast<size_t>(slot_rows));
  ALLOC(d_source, static_cast<size_t>(slot_rows) * PLANETS);
  ALLOC(d_target, static_cast<size_t>(slot_rows) * PLANETS);
  ALLOC(d_amount, static_cast<size_t>(slot_rows) * AMOUNTS);
#undef ALLOC

  error = d_tokens.copy_from_host(tokens, static_cast<size_t>(token_rows) * TOKEN_FEATURES);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_token_type_ids.copy_from_host(token_type_ids, token_rows);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_owner_ids.copy_from_host(owner_ids, token_rows);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_padding_mask.copy_from_host(padding_mask, token_rows);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_planet_mask.copy_from_host(planet_mask, static_cast<size_t>(batch_count) * PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("inputs_copied");

  token_projection_kernel<<<blocks_for(token_rows * D_MODEL), 256>>>(
      d_tokens.ptr, d_token_type_ids.ptr, d_owner_ids.ptr, token_projection_weight,
      token_projection_bias, type_embedding, owner_embedding, hidden.ptr, batch_count, token_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("token_projection");

  for (int layer = 0; layer < ENCODER_LAYERS; ++layer) {
    error = encoder_layer(
        tensor_map, layer, hidden.ptr, d_padding_mask.ptr, norm_token, token_attn,
        token_q, token_k, token_v, token_qkv, token_context, token_ff_mid, token_ff_out,
        batch_count, token_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  debug_stage("encoder_done");

  copy_slot_queries_kernel<<<blocks_for(slot_rows * D_MODEL), 256>>>(slot_queries, slots.ptr, batch_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);

  for (int layer = 0; layer < DECODER_LAYERS; ++layer) {
    error = decoder_layer(
        tensor_map, layer, slots.ptr, hidden.ptr, d_padding_mask.ptr, norm_slot, slot_attn,
        slot_q, slot_k, slot_v, slot_qkv, slot_context, slot_ff_mid, slot_ff_out,
        batch_count, token_count);
    if (error != cudaSuccess) return cuda_error(error);
  }
  debug_stage("decoder_done");

  const float* fire_weight = tensor_ptr(tensor_map, "fire_head.weight");
  const float* fire_bias = tensor_ptr(tensor_map, "fire_head.bias");
  const float* source_weight = tensor_ptr(tensor_map, "source_head.weight");
  const float* source_bias = tensor_ptr(tensor_map, "source_head.bias");
  const float* target_weight = tensor_ptr(tensor_map, "target_head.weight");
  const float* target_bias = tensor_ptr(tensor_map, "target_head.bias");
  const float* amount_weight = tensor_ptr(tensor_map, "amount_head.weight");
  const float* amount_bias = tensor_ptr(tensor_map, "amount_head.bias");
  if (!fire_weight || !fire_bias || !source_weight || !source_bias || !target_weight ||
      !target_bias || !amount_weight || !amount_bias) {
    return bad_argument("missing OWV8 head tensor");
  }
  error = launch_linear(slots.ptr, fire_weight, fire_bias, d_fire.ptr, slot_rows, D_MODEL, 1);
  if (error != cudaSuccess) return cuda_error(error);
  error = launch_linear(slots.ptr, source_weight, source_bias, d_source.ptr, slot_rows, D_MODEL, PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = launch_linear(slots.ptr, target_weight, target_bias, d_target.ptr, slot_rows, D_MODEL, PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = launch_linear(slots.ptr, amount_weight, amount_bias, d_amount.ptr, slot_rows, D_MODEL, AMOUNTS);
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("heads_done");
  mask_planet_logits_kernel<<<blocks_for(slot_rows * PLANETS), 256>>>(
      d_source.ptr, d_planet_mask.ptr, batch_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);
  mask_planet_logits_kernel<<<blocks_for(slot_rows * PLANETS), 256>>>(
      d_target.ptr, d_planet_mask.ptr, batch_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_error(error);

  error = d_fire.copy_to_host(fire_logits, static_cast<size_t>(slot_rows));
  if (error != cudaSuccess) return cuda_error(error);
  error = d_source.copy_to_host(source_logits, static_cast<size_t>(slot_rows) * PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_target.copy_to_host(target_logits, static_cast<size_t>(slot_rows) * PLANETS);
  if (error != cudaSuccess) return cuda_error(error);
  error = d_amount.copy_to_host(amount_logits, static_cast<size_t>(slot_rows) * AMOUNTS);
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("outputs_copied");
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return cuda_error(error);
  debug_stage("forward_done");
  return ok();
}

