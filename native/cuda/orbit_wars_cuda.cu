#include "orbit_wars_cuda.h"

#include <algorithm>
#include <cfloat>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int CUDA_STATUS_OK = 0;
constexpr int CUDA_STATUS_BAD_ARGUMENT = 1;
constexpr int CUDA_STATUS_RUNTIME_ERROR = 2;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int TRUE2D_ATTENTION_WARP_SIZE = 32;
constexpr int TRUE2D_ATTENTION_WARPS_PER_BLOCK =
    THREADS_PER_BLOCK / TRUE2D_ATTENTION_WARP_SIZE;
constexpr unsigned CUDA_FULL_WARP_MASK = 0xffffffffu;
constexpr size_t CUDA_INPUT_FEATURE_COUNT = 2;
constexpr size_t CUDA_OUTPUT_FEATURE_COUNT = 2;
constexpr size_t TRUE2D_INPUT_FEATURE_COUNT = 7;
constexpr size_t TRUE2D_ACTION_TARGETS_PER_SOURCE = 12;
constexpr size_t TRUE2D_ACTION_TARGET_FEATURES = 2;
constexpr size_t TRUE2D_OUTPUT_FEATURE_COUNT =
    TRUE2D_ACTION_TARGETS_PER_SOURCE * TRUE2D_ACTION_TARGET_FEATURES;
constexpr size_t CUDA_MIN_D_MODEL = 8;
constexpr size_t CUDA_MAX_D_MODEL = 256;
constexpr size_t TRUE2D_INPUT_PAIR_FEATURE_WEIGHTS = TRUE2D_INPUT_FEATURE_COUNT * 2;
constexpr size_t TRUE2D_EXPAND_BIAS_WEIGHTS = 1;
constexpr size_t TRUE2D_ATTENTION_HEAD_WEIGHTS = 7;
constexpr size_t TRUE2D_LAYER_SHARED_WEIGHTS = 6;
constexpr size_t TRUE2D_MATRIX_LAYER_CELL_WEIGHTS = 12;
constexpr size_t TRUE2D_OUTPUT_SCALAR_WEIGHTS = 5;
constexpr size_t TRUE2D_SOURCE_OWNER_WEIGHT_INDEX = 0;
constexpr size_t TRUE2D_SOURCE_SHIP_WEIGHT_INDEX = 1;
constexpr size_t TRUE2D_SOURCE_X_WEIGHT_INDEX = 2;
constexpr size_t TRUE2D_SOURCE_Y_WEIGHT_INDEX = 3;
constexpr size_t TRUE2D_SOURCE_PRODUCTION_WEIGHT_INDEX = 4;
constexpr size_t TRUE2D_SOURCE_VELOCITY_X_WEIGHT_INDEX = 5;
constexpr size_t TRUE2D_SOURCE_VELOCITY_Y_WEIGHT_INDEX = 6;
constexpr size_t TRUE2D_TARGET_OWNER_WEIGHT_INDEX = 7;
constexpr size_t TRUE2D_TARGET_SHIP_WEIGHT_INDEX = 8;
constexpr size_t TRUE2D_TARGET_X_WEIGHT_INDEX = 9;
constexpr size_t TRUE2D_TARGET_Y_WEIGHT_INDEX = 10;
constexpr size_t TRUE2D_TARGET_PRODUCTION_WEIGHT_INDEX = 11;
constexpr size_t TRUE2D_TARGET_VELOCITY_X_WEIGHT_INDEX = 12;
constexpr size_t TRUE2D_TARGET_VELOCITY_Y_WEIGHT_INDEX = 13;
constexpr size_t TRUE2D_HEAD_QUERY_SCALE_INDEX = 0;
constexpr size_t TRUE2D_HEAD_QUERY_BIAS_INDEX = 1;
constexpr size_t TRUE2D_HEAD_KEY_SCALE_INDEX = 2;
constexpr size_t TRUE2D_HEAD_KEY_BIAS_INDEX = 3;
constexpr size_t TRUE2D_HEAD_VALUE_SCALE_INDEX = 4;
constexpr size_t TRUE2D_HEAD_VALUE_BIAS_INDEX = 5;
constexpr size_t TRUE2D_HEAD_OUTPUT_SCALE_INDEX = 6;
constexpr size_t TRUE2D_LAYER_RESIDUAL_SCALE_INDEX = 0;
constexpr size_t TRUE2D_LAYER_ATTENTION_BIAS_INDEX = 1;
constexpr size_t TRUE2D_LAYER_FFN_INPUT_SCALE_INDEX = 2;
constexpr size_t TRUE2D_LAYER_FFN_INPUT_BIAS_INDEX = 3;
constexpr size_t TRUE2D_LAYER_FFN_OUTPUT_SCALE_INDEX = 4;
constexpr size_t TRUE2D_LAYER_FFN_OUTPUT_BIAS_INDEX = 5;
constexpr size_t TRUE2D_MATRIX_RESIDUAL_SCALE_INDEX = 0;
constexpr size_t TRUE2D_MATRIX_RESIDUAL_BIAS_INDEX = 1;
constexpr size_t TRUE2D_MATRIX_HIDDEN_A_SCALE_INDEX = 2;
constexpr size_t TRUE2D_MATRIX_HIDDEN_A_BIAS_INDEX = 3;
constexpr size_t TRUE2D_MATRIX_HIDDEN_B_CELL_SCALE_INDEX = 4;
constexpr size_t TRUE2D_MATRIX_HIDDEN_B_HIDDEN_SCALE_INDEX = 5;
constexpr size_t TRUE2D_MATRIX_HIDDEN_B_BIAS_INDEX = 6;
constexpr size_t TRUE2D_MATRIX_GATE_SCALE_INDEX = 7;
constexpr size_t TRUE2D_MATRIX_GATE_BIAS_INDEX = 8;
constexpr size_t TRUE2D_MATRIX_UPDATE_HIDDEN_B_SCALE_INDEX = 9;
constexpr size_t TRUE2D_MATRIX_UPDATE_HIDDEN_A_SCALE_INDEX = 10;
constexpr size_t TRUE2D_MATRIX_UPDATE_BIAS_INDEX = 11;
constexpr size_t TRUE2D_TARGET_OUTPUT_SCALE_INDEX = 0;
constexpr size_t TRUE2D_SEND_KEY_SCALE_INDEX = 1;
constexpr size_t TRUE2D_SEND_VALUE_SCALE_INDEX = 2;
constexpr size_t TRUE2D_SEND_OUTPUT_SCALE_INDEX = 3;
constexpr size_t TRUE2D_SEND_OUTPUT_BIAS_INDEX = 4;
constexpr size_t TRUE2D_OUTPUT_TARGET_FEATURE_INDEX = 0;
constexpr size_t TRUE2D_OUTPUT_SEND_FEATURE_INDEX = 1;
constexpr size_t TRUE2D_MIN_ROW_COUNT = 2;
constexpr size_t TRUE2D_MIN_HEAD_COUNT = 1;
constexpr size_t TRUE2D_MAX_HEAD_COUNT = 16;
constexpr float TRUE2D_OWNER_CLASS_EPSILON = 0.000001f;

__global__ void copy_features_kernel(const float* input, float* output, size_t value_count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < value_count) {
    output[index] = input[index];
  }
}

__device__ float sigmoid_device(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ float clamp_probability_device(float value) {
  return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ bool owner_class_matches(float left, float right) {
  return fabsf(left - right) <= TRUE2D_OWNER_CLASS_EPSILON;
}

__device__ float warp_reduce_max(float value) {
  for (int offset = TRUE2D_ATTENTION_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(CUDA_FULL_WARP_MASK, value, offset));
  }
  return value;
}

__device__ float warp_reduce_sum(float value) {
  for (int offset = TRUE2D_ATTENTION_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(CUDA_FULL_WARP_MASK, value, offset);
  }
  return value;
}

__device__ float bf16_round_float(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ float true2d_matrix_layer_forward_value(
    float cell_value,
    const float* cell_weights) {
  const float hidden_a =
      tanhf(cell_value * cell_weights[TRUE2D_MATRIX_HIDDEN_A_SCALE_INDEX] +
            cell_weights[TRUE2D_MATRIX_HIDDEN_A_BIAS_INDEX]);
  const float hidden_b =
      tanhf(cell_value * cell_weights[TRUE2D_MATRIX_HIDDEN_B_CELL_SCALE_INDEX] +
            hidden_a * cell_weights[TRUE2D_MATRIX_HIDDEN_B_HIDDEN_SCALE_INDEX] +
            cell_weights[TRUE2D_MATRIX_HIDDEN_B_BIAS_INDEX]);
  const float gate =
      sigmoid_device(cell_value * cell_weights[TRUE2D_MATRIX_GATE_SCALE_INDEX] +
                     cell_weights[TRUE2D_MATRIX_GATE_BIAS_INDEX]);
  const float update =
      hidden_b * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_B_SCALE_INDEX] +
      hidden_a * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_A_SCALE_INDEX] +
      cell_weights[TRUE2D_MATRIX_UPDATE_BIAS_INDEX];
  const float residual =
      cell_value * (1.0f + cell_weights[TRUE2D_MATRIX_RESIDUAL_SCALE_INDEX]) +
      cell_weights[TRUE2D_MATRIX_RESIDUAL_BIAS_INDEX];
  return tanhf(residual + gate * update);
}

__global__ void input_projection_kernel(
    const float* input_rows,
    const float* weights,
    float* hidden,
    size_t game_count,
    size_t row_count,
    size_t d_model) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t value_count = game_count * row_count * d_model;
  if (index >= value_count) {
    return;
  }

  const size_t dim = index % d_model;
  const size_t row = (index / d_model) % row_count;
  const size_t game = index / (row_count * d_model);
  const size_t input_offset = (game * row_count + row) * CUDA_INPUT_FEATURE_COUNT;
  const size_t position_offset = CUDA_INPUT_FEATURE_COUNT * d_model;
  const float position = static_cast<float>(row) / static_cast<float>(row_count);
  const float owner = input_rows[input_offset];
  const float ships = input_rows[input_offset + 1];
  const float value = owner * weights[dim] + ships * weights[d_model + dim] +
                      position * weights[position_offset + dim];
  hidden[index] = tanhf(value);
}

__global__ void attention_kernel(
    const float* hidden,
    float* next,
    size_t game_count,
    size_t row_count,
    size_t d_model,
    size_t head_count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t value_count = game_count * row_count * d_model;
  if (index >= value_count) {
    return;
  }

  const size_t dim = index % d_model;
  const size_t source_row = (index / d_model) % row_count;
  const size_t game = index / (row_count * d_model);
  const size_t head_dim = d_model / head_count;
  const size_t head_start = (dim / head_dim) * head_dim;
  const size_t source_base = (game * row_count + source_row) * d_model;
  const float scale = sqrtf(static_cast<float>(head_dim));

  float max_score = -FLT_MAX;
  for (size_t target_row = 0; target_row < row_count; ++target_row) {
    const size_t target_base = (game * row_count + target_row) * d_model;
    float dot = 0.0f;
    for (size_t dim_offset = 0; dim_offset < head_dim; ++dim_offset) {
      const size_t head_dim_index = head_start + dim_offset;
      dot += hidden[source_base + head_dim_index] * hidden[target_base + head_dim_index];
    }
    max_score = fmaxf(max_score, dot / scale);
  }

  float denominator = 0.0f;
  float weighted_value = 0.0f;
  for (size_t target_row = 0; target_row < row_count; ++target_row) {
    const size_t target_base = (game * row_count + target_row) * d_model;
    float dot = 0.0f;
    for (size_t dim_offset = 0; dim_offset < head_dim; ++dim_offset) {
      const size_t head_dim_index = head_start + dim_offset;
      dot += hidden[source_base + head_dim_index] * hidden[target_base + head_dim_index];
    }
    const float attention_weight = expf((dot / scale) - max_score);
    denominator += attention_weight;
    weighted_value += attention_weight * hidden[target_base + dim];
  }

  next[index] = weighted_value / denominator;
}

__global__ void layer_apply_kernel(
    float* hidden,
    const float* next,
    const float* weights,
    size_t game_count,
    size_t row_count,
    size_t d_model,
    size_t layer_count,
    size_t layer_index) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t value_count = game_count * row_count * d_model;
  if (index >= value_count) {
    return;
  }

  const size_t dim = index % d_model;
  const size_t layer_scales_offset = CUDA_INPUT_FEATURE_COUNT * d_model + d_model;
  const size_t layer_scale_index = layer_scales_offset + layer_index * d_model + dim;
  if (layer_index >= layer_count) {
    return;
  }
  hidden[index] = tanhf(hidden[index] + next[index] * weights[layer_scale_index]);
}

__global__ void output_projection_kernel(
    const float* hidden,
    const float* weights,
    float* output_rows,
    size_t game_count,
    size_t row_count,
    size_t d_model,
    size_t layer_count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row_value_count = game_count * row_count;
  if (index >= row_value_count) {
    return;
  }

  const size_t hidden_offset = index * d_model;
  const size_t output_offset = index * CUDA_OUTPUT_FEATURE_COUNT;
  const size_t output_weights_offset =
      CUDA_INPUT_FEATURE_COUNT * d_model + d_model + layer_count * d_model;
  const float scale = sqrtf(static_cast<float>(d_model));
  float target_logit = 0.0f;
  float send_logit = 0.0f;
  for (size_t dim = 0; dim < d_model; ++dim) {
    target_logit += hidden[hidden_offset + dim] * weights[output_weights_offset + dim];
    send_logit += hidden[hidden_offset + dim] * weights[output_weights_offset + d_model + dim];
  }
  output_rows[output_offset] = sigmoid_device(target_logit / scale);
  output_rows[output_offset + 1] = sigmoid_device(send_logit / scale);
}

OrbitWarsCudaStatus status_from_cuda(cudaError_t error) {
  if (error == cudaSuccess) {
    return {CUDA_STATUS_OK, "ok"};
  }
  return {CUDA_STATUS_RUNTIME_ERROR, cudaGetErrorString(error)};
}

bool shape_is_valid(OrbitWarsCudaTransformerShape shape) {
  return shape.layer_count > 0 && shape.d_model >= CUDA_MIN_D_MODEL &&
         shape.d_model <= CUDA_MAX_D_MODEL && shape.head_count > 0 && shape.row_count > 0 &&
         shape.d_model % shape.head_count == 0;
}

size_t transformer_weight_count(OrbitWarsCudaTransformerShape shape) {
  return CUDA_INPUT_FEATURE_COUNT * shape.d_model + shape.d_model +
         shape.layer_count * shape.d_model + shape.d_model * CUDA_OUTPUT_FEATURE_COUNT;
}

bool true2d_shape_is_valid(OrbitWarsCudaTrue2DShape shape) {
  return shape.layer_count > 0 && shape.row_count >= TRUE2D_MIN_ROW_COUNT &&
         shape.head_count >= TRUE2D_MIN_HEAD_COUNT && shape.head_count <= TRUE2D_MAX_HEAD_COUNT;
}

__host__ __device__ size_t true2d_cell_count(size_t row_count) {
  return row_count * row_count;
}

__host__ __device__ size_t true2d_source_row_embeddings_offset() {
  return TRUE2D_INPUT_PAIR_FEATURE_WEIGHTS;
}

__host__ __device__ size_t true2d_target_row_embeddings_offset(size_t row_count) {
  return true2d_source_row_embeddings_offset() + row_count;
}

__host__ __device__ size_t true2d_pair_embeddings_offset(size_t row_count) {
  return true2d_target_row_embeddings_offset(row_count) + row_count;
}

__host__ __device__ size_t true2d_expand_bias_offset(size_t row_count) {
  return true2d_pair_embeddings_offset(row_count) + true2d_cell_count(row_count);
}

__host__ __device__ size_t true2d_matrix_layer_weights_offset(size_t row_count) {
  return true2d_expand_bias_offset(row_count) + TRUE2D_EXPAND_BIAS_WEIGHTS;
}

__host__ __device__ size_t true2d_matrix_layer_weight_count(size_t row_count) {
  return true2d_cell_count(row_count) * TRUE2D_MATRIX_LAYER_CELL_WEIGHTS;
}

__host__ __device__ size_t true2d_attention_weights_offset(
    size_t row_count,
    size_t layer_count) {
  return true2d_matrix_layer_weights_offset(row_count) +
         layer_count * true2d_matrix_layer_weight_count(row_count);
}

__host__ __device__ size_t true2d_attention_weight_count(size_t head_count) {
  return head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS + TRUE2D_LAYER_SHARED_WEIGHTS;
}

__host__ __device__ size_t true2d_target_output_bias_offset(
    size_t row_count,
    size_t layer_count,
    size_t head_count) {
  return true2d_attention_weights_offset(row_count, layer_count) +
         true2d_attention_weight_count(head_count);
}

__host__ __device__ size_t true2d_send_output_bias_offset(
    size_t row_count,
    size_t layer_count,
    size_t head_count) {
  return true2d_target_output_bias_offset(row_count, layer_count, head_count) +
         row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE;
}

__host__ __device__ size_t true2d_output_scalar_weights_offset(
    size_t row_count,
    size_t layer_count,
    size_t head_count) {
  return true2d_send_output_bias_offset(row_count, layer_count, head_count) +
         row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE;
}

__host__ __device__ size_t true2d_matrix_history_offset(
    size_t sample,
    size_t state_index,
    size_t cell_count,
    size_t layer_count) {
  return (sample * (layer_count + 2) + state_index) * cell_count;
}

__host__ __device__ size_t true2d_attention_history_offset(
    size_t sample,
    size_t cell_count) {
  return sample * cell_count;
}

__host__ __device__ size_t true2d_attention_head_history_offset(
    size_t sample,
    size_t head_index,
    size_t cell_count,
    size_t head_count) {
  return (sample * head_count + head_index) * cell_count;
}

__host__ __device__ size_t true2d_attention_head_cell_gradient_offset(
    size_t sample,
    size_t head_index,
    size_t cell_index,
    size_t parameter_index,
    size_t cell_count,
    size_t head_count) {
  return (true2d_attention_head_history_offset(
              sample, head_index, cell_count, head_count) +
          cell_index) *
             TRUE2D_ATTENTION_HEAD_WEIGHTS +
         parameter_index;
}

__host__ __device__ size_t true2d_attention_layer_index(size_t layer_count) {
  return layer_count / 2;
}

__host__ __device__ size_t true2d_matrix_layer_input_state(
    size_t layer_index,
    size_t attention_layer_index) {
  return layer_index < attention_layer_index ? layer_index : layer_index + 1;
}

__host__ __device__ size_t true2d_matrix_layer_output_state(
    size_t layer_index,
    size_t attention_layer_index) {
  return true2d_matrix_layer_input_state(layer_index, attention_layer_index) + 1;
}

__host__ __device__ size_t true2d_final_matrix_state(size_t layer_count) {
  return layer_count + 1;
}

size_t true2d_weight_count(OrbitWarsCudaTrue2DShape shape) {
  return true2d_output_scalar_weights_offset(shape.row_count, shape.layer_count, shape.head_count) +
         TRUE2D_OUTPUT_SCALAR_WEIGHTS * TRUE2D_ACTION_TARGETS_PER_SOURCE;
}

struct True2DManyWorkspace {
  float* device_input = nullptr;
  size_t* device_model_indices = nullptr;
  float* device_weights = nullptr;
  float* device_matrix = nullptr;
  float* device_next_matrix = nullptr;
  float* device_attention_queries = nullptr;
  float* device_attention_keys = nullptr;
  float* device_attention_values = nullptr;
  float* device_output = nullptr;
  size_t request_capacity = 0;
  size_t model_capacity = 0;
  size_t row_count = 0;
  size_t layer_count = 0;
  size_t head_count = 0;
  size_t resident_model_count = 0;
  size_t resident_weight_count = 0;
  bool resident_weights_loaded = false;
};

True2DManyWorkspace true2d_many_workspace;

cudaError_t free_true2d_many_workspace() {
  cudaError_t first_error = cudaSuccess;
  auto release_float = [&first_error](float*& pointer) {
    if (pointer == nullptr) {
      return;
    }
    const cudaError_t error = cudaFree(pointer);
    if (first_error == cudaSuccess && error != cudaSuccess) {
      first_error = error;
    }
    pointer = nullptr;
  };
  auto release_size = [&first_error](size_t*& pointer) {
    if (pointer == nullptr) {
      return;
    }
    const cudaError_t error = cudaFree(pointer);
    if (first_error == cudaSuccess && error != cudaSuccess) {
      first_error = error;
    }
    pointer = nullptr;
  };
  release_float(true2d_many_workspace.device_output);
  release_float(true2d_many_workspace.device_attention_values);
  release_float(true2d_many_workspace.device_attention_keys);
  release_float(true2d_many_workspace.device_attention_queries);
  release_float(true2d_many_workspace.device_next_matrix);
  release_float(true2d_many_workspace.device_matrix);
  release_float(true2d_many_workspace.device_weights);
  release_size(true2d_many_workspace.device_model_indices);
  release_float(true2d_many_workspace.device_input);
  true2d_many_workspace.request_capacity = 0;
  true2d_many_workspace.model_capacity = 0;
  true2d_many_workspace.row_count = 0;
  true2d_many_workspace.layer_count = 0;
  true2d_many_workspace.head_count = 0;
  true2d_many_workspace.resident_model_count = 0;
  true2d_many_workspace.resident_weight_count = 0;
  true2d_many_workspace.resident_weights_loaded = false;
  return first_error;
}

bool true2d_many_workspace_matches(
    size_t request_count,
    size_t model_count,
    OrbitWarsCudaTrue2DShape shape) {
  return true2d_many_workspace.request_capacity >= request_count &&
         true2d_many_workspace.model_capacity >= model_count &&
         true2d_many_workspace.row_count == shape.row_count &&
         true2d_many_workspace.layer_count == shape.layer_count &&
         true2d_many_workspace.head_count == shape.head_count &&
         true2d_many_workspace.device_input != nullptr &&
         true2d_many_workspace.device_model_indices != nullptr &&
         true2d_many_workspace.device_weights != nullptr &&
         true2d_many_workspace.device_matrix != nullptr &&
         true2d_many_workspace.device_next_matrix != nullptr &&
         true2d_many_workspace.device_attention_queries != nullptr &&
         true2d_many_workspace.device_attention_keys != nullptr &&
         true2d_many_workspace.device_attention_values != nullptr &&
         true2d_many_workspace.device_output != nullptr;
}

OrbitWarsCudaStatus ensure_true2d_many_workspace(
    size_t request_count,
    size_t model_count,
    OrbitWarsCudaTrue2DShape shape) {
  if (true2d_many_workspace_matches(request_count, model_count, shape)) {
    return {CUDA_STATUS_OK, "ok"};
  }

  cudaError_t error = free_true2d_many_workspace();
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }

  const size_t input_count = request_count * shape.row_count * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t matrix_count = request_count * cell_count;
  const size_t attention_projection_count = matrix_count * shape.head_count;
  const size_t output_count = request_count * shape.row_count * TRUE2D_OUTPUT_FEATURE_COUNT;
  const size_t weight_count = true2d_weight_count(shape);
  const size_t all_weight_count = model_count * weight_count;

  if (error == cudaSuccess) {
    error = cudaMalloc(&true2d_many_workspace.device_input, input_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &true2d_many_workspace.device_model_indices,
        request_count * sizeof(size_t));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&true2d_many_workspace.device_weights, all_weight_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&true2d_many_workspace.device_matrix, matrix_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&true2d_many_workspace.device_next_matrix, matrix_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &true2d_many_workspace.device_attention_queries,
        attention_projection_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &true2d_many_workspace.device_attention_keys,
        attention_projection_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &true2d_many_workspace.device_attention_values,
        attention_projection_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&true2d_many_workspace.device_output, output_count * sizeof(float));
  }

  if (error != cudaSuccess) {
    free_true2d_many_workspace();
    return status_from_cuda(error);
  }

  true2d_many_workspace.request_capacity = request_count;
  true2d_many_workspace.model_capacity = model_count;
  true2d_many_workspace.row_count = shape.row_count;
  true2d_many_workspace.layer_count = shape.layer_count;
  true2d_many_workspace.head_count = shape.head_count;
  return {CUDA_STATUS_OK, "ok"};
}

bool true2d_many_batch_is_valid(
    const float* input_rows,
    const size_t* model_indices,
    const float* all_weights,
    float* output_rows,
    size_t request_count,
    size_t model_count,
    OrbitWarsCudaTrue2DShape shape) {
  if (input_rows == nullptr || model_indices == nullptr || all_weights == nullptr ||
      output_rows == nullptr || request_count == 0 || model_count == 0 ||
      !true2d_shape_is_valid(shape)) {
    return false;
  }
  for (size_t request_index = 0; request_index < request_count; ++request_index) {
    if (model_indices[request_index] >= model_count) {
      return false;
    }
  }
  return true;
}

bool true2d_many_request_is_valid(
    const float* input_rows,
    const size_t* model_indices,
    float* output_rows,
    size_t request_count,
    size_t model_count,
    OrbitWarsCudaTrue2DShape shape) {
  if (input_rows == nullptr || model_indices == nullptr || output_rows == nullptr ||
      request_count == 0 || model_count == 0 || !true2d_shape_is_valid(shape)) {
    return false;
  }
  for (size_t request_index = 0; request_index < request_count; ++request_index) {
    if (model_indices[request_index] >= model_count) {
      return false;
    }
  }
  return true;
}

bool true2d_training_config_is_valid(OrbitWarsCudaTrue2DTrainingConfig config) {
  return config.batch_sample_count > 0 && config.learning_rate > 0.0f &&
         config.reward_normalizer > 0.0f && config.send_threshold >= 0.0f &&
         config.send_threshold <= 1.0f;
}

bool true2d_resident_population_matches(OrbitWarsCudaTrue2DShape shape) {
  return true2d_many_workspace.resident_weights_loaded &&
         true2d_many_workspace.resident_model_count > 0 &&
         true2d_many_workspace.resident_weight_count == true2d_weight_count(shape) &&
         true2d_many_workspace.row_count == shape.row_count &&
         true2d_many_workspace.layer_count == shape.layer_count &&
         true2d_many_workspace.head_count == shape.head_count &&
         true2d_many_workspace.device_weights != nullptr;
}

__device__ const float* true2d_many_weights_for_game(
    const float* all_weights,
    const size_t* model_indices,
    size_t game,
    size_t model_count,
    size_t weight_count) {
  const size_t model_index = model_indices[game];
  if (model_index >= model_count) {
    return nullptr;
  }
  return all_weights + model_index * weight_count;
}

__global__ void true2d_expand_kernel(
    const float* input_rows,
    const float* weights,
    float* matrix,
    size_t game_count,
    size_t row_count) {
  const size_t cell_count = true2d_cell_count(row_count);
  const size_t value_count = game_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t cell_index = index % cell_count;
  const size_t game = index / cell_count;
  const size_t source_row = cell_index / row_count;
  const size_t target_row = cell_index % row_count;
  const size_t source_input = (game * row_count + source_row) * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t target_input = (game * row_count + target_row) * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t source_embeddings = true2d_source_row_embeddings_offset();
  const size_t target_embeddings = true2d_target_row_embeddings_offset(row_count);
  const size_t pair_embeddings = true2d_pair_embeddings_offset(row_count);
  const float value =
      input_rows[source_input] * weights[TRUE2D_SOURCE_OWNER_WEIGHT_INDEX] +
      input_rows[source_input + 1] * weights[TRUE2D_SOURCE_SHIP_WEIGHT_INDEX] +
      input_rows[source_input + 2] * weights[TRUE2D_SOURCE_X_WEIGHT_INDEX] +
      input_rows[source_input + 3] * weights[TRUE2D_SOURCE_Y_WEIGHT_INDEX] +
      input_rows[source_input + 4] * weights[TRUE2D_SOURCE_PRODUCTION_WEIGHT_INDEX] +
      input_rows[source_input + 5] * weights[TRUE2D_SOURCE_VELOCITY_X_WEIGHT_INDEX] +
      input_rows[source_input + 6] * weights[TRUE2D_SOURCE_VELOCITY_Y_WEIGHT_INDEX] +
      input_rows[target_input] * weights[TRUE2D_TARGET_OWNER_WEIGHT_INDEX] +
      input_rows[target_input + 1] * weights[TRUE2D_TARGET_SHIP_WEIGHT_INDEX] +
      input_rows[target_input + 2] * weights[TRUE2D_TARGET_X_WEIGHT_INDEX] +
      input_rows[target_input + 3] * weights[TRUE2D_TARGET_Y_WEIGHT_INDEX] +
      input_rows[target_input + 4] * weights[TRUE2D_TARGET_PRODUCTION_WEIGHT_INDEX] +
      input_rows[target_input + 5] * weights[TRUE2D_TARGET_VELOCITY_X_WEIGHT_INDEX] +
      input_rows[target_input + 6] * weights[TRUE2D_TARGET_VELOCITY_Y_WEIGHT_INDEX] +
      weights[source_embeddings + source_row] + weights[target_embeddings + target_row] +
      weights[pair_embeddings + cell_index] + weights[true2d_expand_bias_offset(row_count)];
  matrix[index] = tanhf(value);
}

__global__ void true2d_output_kernel(
    const float* matrix,
    const float* weights,
    float* output_rows,
    size_t game_count,
    OrbitWarsCudaTrue2DShape shape) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row_value_count = game_count * shape.row_count;
  if (index >= row_value_count) {
    return;
  }

  const size_t game = index / shape.row_count;
  const size_t source_row = index % shape.row_count;
  const size_t row_count = shape.row_count;
  const size_t matrix_row_offset =
      game * true2d_cell_count(row_count) + source_row * row_count;
  const size_t target_bias_offset =
      true2d_target_output_bias_offset(row_count, shape.layer_count, shape.head_count);
  const size_t send_bias_offset =
      true2d_send_output_bias_offset(row_count, shape.layer_count, shape.head_count);
  const size_t output_weights_offset =
      true2d_output_scalar_weights_offset(row_count, shape.layer_count, shape.head_count);
  const float target_fraction_denominator = static_cast<float>(row_count - 1);
  const size_t row_output_offset = index * TRUE2D_OUTPUT_FEATURE_COUNT;
  float send_fractions[TRUE2D_ACTION_TARGETS_PER_SOURCE];
  float send_fraction_sum = 0.0f;
  for (size_t action_index = 0; action_index < TRUE2D_ACTION_TARGETS_PER_SOURCE; ++action_index) {
    const size_t action_bias_offset = action_index * row_count;
    const size_t action_weights_offset =
        output_weights_offset + action_index * TRUE2D_OUTPUT_SCALAR_WEIGHTS;
    const float target_scale =
        weights[action_weights_offset + TRUE2D_TARGET_OUTPUT_SCALE_INDEX];
    const float send_key_scale = weights[action_weights_offset + TRUE2D_SEND_KEY_SCALE_INDEX];
    const float send_value_scale = weights[action_weights_offset + TRUE2D_SEND_VALUE_SCALE_INDEX];
    const float send_output_scale =
        weights[action_weights_offset + TRUE2D_SEND_OUTPUT_SCALE_INDEX];

    float target_max_score = -FLT_MAX;
    for (size_t target_row = 0; target_row < row_count; ++target_row) {
      const float score =
          matrix[matrix_row_offset + target_row] * target_scale +
          weights[target_bias_offset + action_bias_offset + target_row];
      target_max_score = fmaxf(target_max_score, score);
    }
    float target_denominator = 0.0f;
    float target_fraction = 0.0f;
    for (size_t target_row = 0; target_row < row_count; ++target_row) {
      const float score =
          matrix[matrix_row_offset + target_row] * target_scale +
          weights[target_bias_offset + action_bias_offset + target_row];
      const float attention = expf(score - target_max_score);
      target_denominator += attention;
      target_fraction += attention * (static_cast<float>(target_row) / target_fraction_denominator);
    }
    target_fraction /= target_denominator;

    float send_max_score = -FLT_MAX;
    for (size_t target_row = 0; target_row < row_count; ++target_row) {
      const float score =
          matrix[matrix_row_offset + target_row] * send_key_scale +
          weights[send_bias_offset + action_bias_offset + target_row];
      send_max_score = fmaxf(send_max_score, score);
    }
    float send_denominator = 0.0f;
    float send_weighted_value = 0.0f;
    for (size_t target_row = 0; target_row < row_count; ++target_row) {
      const float value = matrix[matrix_row_offset + target_row];
      const float score =
          value * send_key_scale + weights[send_bias_offset + action_bias_offset + target_row];
      const float attention = expf(score - send_max_score);
      send_denominator += attention;
      send_weighted_value += attention * value * send_value_scale;
    }
    const float send_logit =
        weights[action_weights_offset + TRUE2D_SEND_OUTPUT_BIAS_INDEX] +
        send_weighted_value / send_denominator;
    const size_t target_output_offset = row_output_offset + action_index * TRUE2D_ACTION_TARGET_FEATURES;
    output_rows[target_output_offset] = fminf(fmaxf(target_fraction, 0.0f), 1.0f);
    send_fractions[action_index] = sigmoid_device(send_logit * send_output_scale);
    send_fraction_sum += send_fractions[action_index];
  }
  const float send_normalizer = send_fraction_sum > 1.0f ? send_fraction_sum : 1.0f;
  for (size_t action_index = 0; action_index < TRUE2D_ACTION_TARGETS_PER_SOURCE; ++action_index) {
    const size_t target_output_offset =
        row_output_offset + action_index * TRUE2D_ACTION_TARGET_FEATURES;
    output_rows[target_output_offset + 1] = send_fractions[action_index] / send_normalizer;
  }
}

__global__ void true2d_expand_many_kernel(
    const float* input_rows,
    const size_t* model_indices,
    const float* all_weights,
    float* matrix,
    size_t request_count,
    size_t model_count,
    size_t weight_count,
    size_t row_count) {
  const size_t cell_count = true2d_cell_count(row_count);
  const size_t value_count = request_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t cell_index = index % cell_count;
  const size_t game = index / cell_count;
  const float* weights =
      true2d_many_weights_for_game(all_weights, model_indices, game, model_count, weight_count);
  if (weights == nullptr) {
    return;
  }
  const size_t source_row = cell_index / row_count;
  const size_t target_row = cell_index % row_count;
  const size_t source_input = (game * row_count + source_row) * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t target_input = (game * row_count + target_row) * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t source_embeddings = true2d_source_row_embeddings_offset();
  const size_t target_embeddings = true2d_target_row_embeddings_offset(row_count);
  const size_t pair_embeddings = true2d_pair_embeddings_offset(row_count);
  const float value =
      input_rows[source_input] * weights[TRUE2D_SOURCE_OWNER_WEIGHT_INDEX] +
      input_rows[source_input + 1] * weights[TRUE2D_SOURCE_SHIP_WEIGHT_INDEX] +
      input_rows[source_input + 2] * weights[TRUE2D_SOURCE_X_WEIGHT_INDEX] +
      input_rows[source_input + 3] * weights[TRUE2D_SOURCE_Y_WEIGHT_INDEX] +
      input_rows[source_input + 4] * weights[TRUE2D_SOURCE_PRODUCTION_WEIGHT_INDEX] +
      input_rows[source_input + 5] * weights[TRUE2D_SOURCE_VELOCITY_X_WEIGHT_INDEX] +
      input_rows[source_input + 6] * weights[TRUE2D_SOURCE_VELOCITY_Y_WEIGHT_INDEX] +
      input_rows[target_input] * weights[TRUE2D_TARGET_OWNER_WEIGHT_INDEX] +
      input_rows[target_input + 1] * weights[TRUE2D_TARGET_SHIP_WEIGHT_INDEX] +
      input_rows[target_input + 2] * weights[TRUE2D_TARGET_X_WEIGHT_INDEX] +
      input_rows[target_input + 3] * weights[TRUE2D_TARGET_Y_WEIGHT_INDEX] +
      input_rows[target_input + 4] * weights[TRUE2D_TARGET_PRODUCTION_WEIGHT_INDEX] +
      input_rows[target_input + 5] * weights[TRUE2D_TARGET_VELOCITY_X_WEIGHT_INDEX] +
      input_rows[target_input + 6] * weights[TRUE2D_TARGET_VELOCITY_Y_WEIGHT_INDEX] +
      weights[source_embeddings + source_row] + weights[target_embeddings + target_row] +
      weights[pair_embeddings + cell_index] + weights[true2d_expand_bias_offset(row_count)];
  matrix[index] = tanhf(value);
}

__global__ void true2d_output_many_kernel(
    const float* matrix,
    const size_t* model_indices,
    const float* all_weights,
    float* output_rows,
    size_t request_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row_value_count = request_count * shape.row_count;
  if (index >= row_value_count) {
    return;
  }

  const size_t game = index / shape.row_count;
  const float* weights =
      true2d_many_weights_for_game(all_weights, model_indices, game, model_count, weight_count);
  if (weights == nullptr) {
    return;
  }
  const size_t source_row = index % shape.row_count;
  const size_t row_count = shape.row_count;
  const size_t matrix_row_offset =
      game * true2d_cell_count(row_count) + source_row * row_count;
  const size_t target_bias_offset =
      true2d_target_output_bias_offset(row_count, shape.layer_count, shape.head_count);
  const size_t send_bias_offset =
      true2d_send_output_bias_offset(row_count, shape.layer_count, shape.head_count);
  const size_t output_weights_offset =
      true2d_output_scalar_weights_offset(row_count, shape.layer_count, shape.head_count);
  const float target_fraction_denominator = static_cast<float>(row_count - 1);
  const size_t row_output_offset = index * TRUE2D_OUTPUT_FEATURE_COUNT;
  float send_fractions[TRUE2D_ACTION_TARGETS_PER_SOURCE];
  float send_fraction_sum = 0.0f;
  for (size_t action_index = 0; action_index < TRUE2D_ACTION_TARGETS_PER_SOURCE; ++action_index) {
    const size_t action_bias_offset = action_index * row_count;
    const size_t action_weights_offset =
        output_weights_offset + action_index * TRUE2D_OUTPUT_SCALAR_WEIGHTS;
    const float target_scale =
        weights[action_weights_offset + TRUE2D_TARGET_OUTPUT_SCALE_INDEX];
    const float send_key_scale = weights[action_weights_offset + TRUE2D_SEND_KEY_SCALE_INDEX];
    const float send_value_scale = weights[action_weights_offset + TRUE2D_SEND_VALUE_SCALE_INDEX];
    const float send_output_scale =
        weights[action_weights_offset + TRUE2D_SEND_OUTPUT_SCALE_INDEX];

    float target_max_score = -FLT_MAX;
    for (size_t target_row = 0; target_row < row_count; ++target_row) {
      const float score =
          matrix[matrix_row_offset + target_row] * target_scale +
          weights[target_bias_offset + action_bias_offset + target_row];
      target_max_score = fmaxf(target_max_score, score);
    }
    float target_denominator = 0.0f;
    float target_fraction = 0.0f;
    for (size_t target_row = 0; target_row < row_count; ++target_row) {
      const float score =
          matrix[matrix_row_offset + target_row] * target_scale +
          weights[target_bias_offset + action_bias_offset + target_row];
      const float attention = expf(score - target_max_score);
      target_denominator += attention;
      target_fraction += attention * (static_cast<float>(target_row) / target_fraction_denominator);
    }
    target_fraction /= target_denominator;

    float send_max_score = -FLT_MAX;
    for (size_t target_row = 0; target_row < row_count; ++target_row) {
      const float score =
          matrix[matrix_row_offset + target_row] * send_key_scale +
          weights[send_bias_offset + action_bias_offset + target_row];
      send_max_score = fmaxf(send_max_score, score);
    }
    float send_denominator = 0.0f;
    float send_weighted_value = 0.0f;
    for (size_t target_row = 0; target_row < row_count; ++target_row) {
      const float value = matrix[matrix_row_offset + target_row];
      const float score =
          value * send_key_scale + weights[send_bias_offset + action_bias_offset + target_row];
      const float attention = expf(score - send_max_score);
      send_denominator += attention;
      send_weighted_value += attention * value * send_value_scale;
    }
    const float send_logit =
        weights[action_weights_offset + TRUE2D_SEND_OUTPUT_BIAS_INDEX] +
        send_weighted_value / send_denominator;
    const size_t target_output_offset = row_output_offset + action_index * TRUE2D_ACTION_TARGET_FEATURES;
    output_rows[target_output_offset] = fminf(fmaxf(target_fraction, 0.0f), 1.0f);
    send_fractions[action_index] = sigmoid_device(send_logit * send_output_scale);
    send_fraction_sum += send_fractions[action_index];
  }
  const float send_normalizer = send_fraction_sum > 1.0f ? send_fraction_sum : 1.0f;
  for (size_t action_index = 0; action_index < TRUE2D_ACTION_TARGETS_PER_SOURCE; ++action_index) {
    const size_t target_output_offset =
        row_output_offset + action_index * TRUE2D_ACTION_TARGET_FEATURES;
    output_rows[target_output_offset + 1] = send_fractions[action_index] / send_normalizer;
  }
}

__global__ void true2d_attention_qkv_kernel(
    const float* matrix,
    const float* weights,
    float* attention_queries,
    float* attention_keys,
    float* attention_values,
    size_t game_count,
    OrbitWarsCudaTrue2DShape shape) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = game_count * shape.head_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t cell_index = index % cell_count;
  const size_t head_index = (index / cell_count) % shape.head_count;
  const size_t game = index / (cell_count * shape.head_count);
  const float* layer_weights =
      weights + true2d_attention_weights_offset(shape.row_count, shape.layer_count);
  const float* head_weights =
      layer_weights + head_index * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  const float cell_value = bf16_round_float(matrix[game * cell_count + cell_index]);
  attention_queries[index] = bf16_round_float(
      cell_value * bf16_round_float(head_weights[TRUE2D_HEAD_QUERY_SCALE_INDEX]) +
      bf16_round_float(head_weights[TRUE2D_HEAD_QUERY_BIAS_INDEX]));
  attention_keys[index] = bf16_round_float(
      cell_value * bf16_round_float(head_weights[TRUE2D_HEAD_KEY_SCALE_INDEX]) +
      bf16_round_float(head_weights[TRUE2D_HEAD_KEY_BIAS_INDEX]));
  attention_values[index] = bf16_round_float(
      cell_value * bf16_round_float(head_weights[TRUE2D_HEAD_VALUE_SCALE_INDEX]) +
      bf16_round_float(head_weights[TRUE2D_HEAD_VALUE_BIAS_INDEX]));
}

__global__ void true2d_attention_qkv_many_kernel(
    const float* matrix,
    const size_t* model_indices,
    const float* all_weights,
    float* attention_queries,
    float* attention_keys,
    float* attention_values,
    size_t request_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = request_count * shape.head_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t cell_index = index % cell_count;
  const size_t head_index = (index / cell_count) % shape.head_count;
  const size_t game = index / (cell_count * shape.head_count);
  const float* weights =
      true2d_many_weights_for_game(all_weights, model_indices, game, model_count, weight_count);
  if (weights == nullptr) {
    return;
  }
  const float* layer_weights =
      weights + true2d_attention_weights_offset(shape.row_count, shape.layer_count);
  const float* head_weights =
      layer_weights + head_index * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  const float cell_value = bf16_round_float(matrix[game * cell_count + cell_index]);
  attention_queries[index] = bf16_round_float(
      cell_value * bf16_round_float(head_weights[TRUE2D_HEAD_QUERY_SCALE_INDEX]) +
      bf16_round_float(head_weights[TRUE2D_HEAD_QUERY_BIAS_INDEX]));
  attention_keys[index] = bf16_round_float(
      cell_value * bf16_round_float(head_weights[TRUE2D_HEAD_KEY_SCALE_INDEX]) +
      bf16_round_float(head_weights[TRUE2D_HEAD_KEY_BIAS_INDEX]));
  attention_values[index] = bf16_round_float(
      cell_value * bf16_round_float(head_weights[TRUE2D_HEAD_VALUE_SCALE_INDEX]) +
      bf16_round_float(head_weights[TRUE2D_HEAD_VALUE_BIAS_INDEX]));
}

__global__ void true2d_full_attention_kernel(
    const float* matrix,
    const float* weights,
    const float* attention_queries,
    const float* attention_keys,
    const float* attention_values,
    float* next_matrix,
    size_t game_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = game_count * cell_count;
  const size_t warp_index =
      blockIdx.x * TRUE2D_ATTENTION_WARPS_PER_BLOCK +
      threadIdx.x / TRUE2D_ATTENTION_WARP_SIZE;
  const int lane_index = threadIdx.x % TRUE2D_ATTENTION_WARP_SIZE;
  if (warp_index >= value_count) {
    return;
  }

  const size_t game = warp_index / cell_count;
  const size_t query_cell = warp_index % cell_count;
  const size_t game_offset = game * cell_count;
  const float* layer_weights =
      weights + true2d_attention_weights_offset(shape.row_count, shape.layer_count);
  const float* shared_weights =
      layer_weights + shape.head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  const float query_cell_value = bf16_round_float(matrix[game_offset + query_cell]);
  float combined_context = 0.0f;
  for (size_t head_index = 0; head_index < shape.head_count; ++head_index) {
    const float* head_weights =
        layer_weights + head_index * TRUE2D_ATTENTION_HEAD_WEIGHTS;
    const size_t head_projection_offset = (game * shape.head_count + head_index) * cell_count;
    const float query = attention_queries[head_projection_offset + query_cell];
    float local_max_score = -FLT_MAX;
    for (size_t key_cell = lane_index; key_cell < cell_count;
         key_cell += TRUE2D_ATTENTION_WARP_SIZE) {
      const float key = attention_keys[head_projection_offset + key_cell];
      local_max_score = fmaxf(local_max_score, bf16_round_float(query * key));
    }
    const float max_score = warp_reduce_max(local_max_score);
    float local_denominator = 0.0f;
    float local_weighted_value = 0.0f;
    for (size_t key_cell = lane_index; key_cell < cell_count;
         key_cell += TRUE2D_ATTENTION_WARP_SIZE) {
      const float key = attention_keys[head_projection_offset + key_cell];
      const float value = attention_values[head_projection_offset + key_cell];
      const float attention = expf(bf16_round_float(query * key) - max_score);
      local_denominator += attention;
      local_weighted_value += attention * value;
    }
    const float denominator = warp_reduce_sum(local_denominator);
    const float weighted_value = warp_reduce_sum(local_weighted_value);
    if (lane_index == 0) {
      combined_context += weighted_value / denominator *
                          bf16_round_float(head_weights[TRUE2D_HEAD_OUTPUT_SCALE_INDEX]);
    }
  }
  if (lane_index == 0) {
    const float residual_attention =
        query_cell_value * shared_weights[TRUE2D_LAYER_RESIDUAL_SCALE_INDEX] +
        combined_context + shared_weights[TRUE2D_LAYER_ATTENTION_BIAS_INDEX];
    const float ffn_hidden =
        tanhf(residual_attention * shared_weights[TRUE2D_LAYER_FFN_INPUT_SCALE_INDEX] +
              shared_weights[TRUE2D_LAYER_FFN_INPUT_BIAS_INDEX]);
    const float value =
        residual_attention +
        ffn_hidden * shared_weights[TRUE2D_LAYER_FFN_OUTPUT_SCALE_INDEX] +
        shared_weights[TRUE2D_LAYER_FFN_OUTPUT_BIAS_INDEX];
    next_matrix[warp_index] = tanhf(value);
  }
}

__global__ void true2d_full_attention_many_kernel(
    const float* matrix,
    const size_t* model_indices,
    const float* all_weights,
    const float* attention_queries,
    const float* attention_keys,
    const float* attention_values,
    float* next_matrix,
    size_t request_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = request_count * cell_count;
  const size_t warp_index =
      blockIdx.x * TRUE2D_ATTENTION_WARPS_PER_BLOCK +
      threadIdx.x / TRUE2D_ATTENTION_WARP_SIZE;
  const int lane_index = threadIdx.x % TRUE2D_ATTENTION_WARP_SIZE;
  if (warp_index >= value_count) {
    return;
  }

  const size_t game = warp_index / cell_count;
  const float* weights =
      true2d_many_weights_for_game(all_weights, model_indices, game, model_count, weight_count);
  if (weights == nullptr) {
    return;
  }
  const size_t query_cell = warp_index % cell_count;
  const size_t game_offset = game * cell_count;
  const float* layer_weights =
      weights + true2d_attention_weights_offset(shape.row_count, shape.layer_count);
  const float* shared_weights =
      layer_weights + shape.head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  const float query_cell_value = bf16_round_float(matrix[game_offset + query_cell]);
  float combined_context = 0.0f;
  for (size_t head_index = 0; head_index < shape.head_count; ++head_index) {
    const float* head_weights =
        layer_weights + head_index * TRUE2D_ATTENTION_HEAD_WEIGHTS;
    const size_t head_projection_offset = (game * shape.head_count + head_index) * cell_count;
    const float query = attention_queries[head_projection_offset + query_cell];
    float local_max_score = -FLT_MAX;
    for (size_t key_cell = lane_index; key_cell < cell_count;
         key_cell += TRUE2D_ATTENTION_WARP_SIZE) {
      const float key = attention_keys[head_projection_offset + key_cell];
      local_max_score = fmaxf(local_max_score, bf16_round_float(query * key));
    }
    const float max_score = warp_reduce_max(local_max_score);
    float local_denominator = 0.0f;
    float local_weighted_value = 0.0f;
    for (size_t key_cell = lane_index; key_cell < cell_count;
         key_cell += TRUE2D_ATTENTION_WARP_SIZE) {
      const float key = attention_keys[head_projection_offset + key_cell];
      const float value = attention_values[head_projection_offset + key_cell];
      const float attention = expf(bf16_round_float(query * key) - max_score);
      local_denominator += attention;
      local_weighted_value += attention * value;
    }
    const float denominator = warp_reduce_sum(local_denominator);
    const float weighted_value = warp_reduce_sum(local_weighted_value);
    if (lane_index == 0) {
      combined_context += weighted_value / denominator *
                          bf16_round_float(head_weights[TRUE2D_HEAD_OUTPUT_SCALE_INDEX]);
    }
  }
  if (lane_index == 0) {
    const float residual_attention =
        query_cell_value * shared_weights[TRUE2D_LAYER_RESIDUAL_SCALE_INDEX] +
        combined_context + shared_weights[TRUE2D_LAYER_ATTENTION_BIAS_INDEX];
    const float ffn_hidden =
        tanhf(residual_attention * shared_weights[TRUE2D_LAYER_FFN_INPUT_SCALE_INDEX] +
              shared_weights[TRUE2D_LAYER_FFN_INPUT_BIAS_INDEX]);
    const float value =
        residual_attention +
        ffn_hidden * shared_weights[TRUE2D_LAYER_FFN_OUTPUT_SCALE_INDEX] +
        shared_weights[TRUE2D_LAYER_FFN_OUTPUT_BIAS_INDEX];
    next_matrix[warp_index] = tanhf(value);
  }
}

__global__ void true2d_matrix_layer_kernel(
    const float* matrix,
    float* next_matrix,
    const float* weights,
    size_t game_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = game_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t cell_index = index % cell_count;
  const size_t layer_offset =
      true2d_matrix_layer_weights_offset(shape.row_count) +
      layer_index * true2d_matrix_layer_weight_count(shape.row_count);
  const float* cell_weights =
      weights + layer_offset + cell_index * TRUE2D_MATRIX_LAYER_CELL_WEIGHTS;
  next_matrix[index] = true2d_matrix_layer_forward_value(matrix[index], cell_weights);
}

__global__ void true2d_matrix_layer_many_kernel(
    const float* matrix,
    float* next_matrix,
    const size_t* model_indices,
    const float* all_weights,
    size_t request_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = request_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t game = index / cell_count;
  const size_t cell_index = index % cell_count;
  const float* weights =
      true2d_many_weights_for_game(all_weights, model_indices, game, model_count, weight_count);
  if (weights == nullptr) {
    return;
  }
  const size_t layer_offset =
      true2d_matrix_layer_weights_offset(shape.row_count) +
      layer_index * true2d_matrix_layer_weight_count(shape.row_count);
  const float* cell_weights =
      weights + layer_offset + cell_index * TRUE2D_MATRIX_LAYER_CELL_WEIGHTS;
  next_matrix[index] = true2d_matrix_layer_forward_value(matrix[index], cell_weights);
}

__global__ void true2d_train_expand_kernel(
    const float* input_rows,
    const size_t* model_indices,
    const float* all_weights,
    float* matrices,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = sample_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t sample = index / cell_count;
  const size_t cell_index = index % cell_count;
  const float* weights =
      true2d_many_weights_for_game(all_weights, model_indices, sample, model_count, weight_count);
  if (weights == nullptr) {
    return;
  }
  const size_t source_row = cell_index / shape.row_count;
  const size_t target_row = cell_index % shape.row_count;
  const size_t source_input =
      (sample * shape.row_count + source_row) * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t target_input =
      (sample * shape.row_count + target_row) * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t source_embeddings = true2d_source_row_embeddings_offset();
  const size_t target_embeddings = true2d_target_row_embeddings_offset(shape.row_count);
  const size_t pair_embeddings = true2d_pair_embeddings_offset(shape.row_count);
  const float value =
      input_rows[source_input] * weights[TRUE2D_SOURCE_OWNER_WEIGHT_INDEX] +
      input_rows[source_input + 1] * weights[TRUE2D_SOURCE_SHIP_WEIGHT_INDEX] +
      input_rows[source_input + 2] * weights[TRUE2D_SOURCE_X_WEIGHT_INDEX] +
      input_rows[source_input + 3] * weights[TRUE2D_SOURCE_Y_WEIGHT_INDEX] +
      input_rows[source_input + 4] * weights[TRUE2D_SOURCE_PRODUCTION_WEIGHT_INDEX] +
      input_rows[source_input + 5] * weights[TRUE2D_SOURCE_VELOCITY_X_WEIGHT_INDEX] +
      input_rows[source_input + 6] * weights[TRUE2D_SOURCE_VELOCITY_Y_WEIGHT_INDEX] +
      input_rows[target_input] * weights[TRUE2D_TARGET_OWNER_WEIGHT_INDEX] +
      input_rows[target_input + 1] * weights[TRUE2D_TARGET_SHIP_WEIGHT_INDEX] +
      input_rows[target_input + 2] * weights[TRUE2D_TARGET_X_WEIGHT_INDEX] +
      input_rows[target_input + 3] * weights[TRUE2D_TARGET_Y_WEIGHT_INDEX] +
      input_rows[target_input + 4] * weights[TRUE2D_TARGET_PRODUCTION_WEIGHT_INDEX] +
      input_rows[target_input + 5] * weights[TRUE2D_TARGET_VELOCITY_X_WEIGHT_INDEX] +
      input_rows[target_input + 6] * weights[TRUE2D_TARGET_VELOCITY_Y_WEIGHT_INDEX] +
      weights[source_embeddings + source_row] + weights[target_embeddings + target_row] +
      weights[pair_embeddings + cell_index] +
      weights[true2d_expand_bias_offset(shape.row_count)];
  matrices[true2d_matrix_history_offset(sample, 0, cell_count, shape.layer_count) + cell_index] =
      tanhf(value);
}

__global__ void true2d_train_attention_forward_kernel(
    const float* matrices,
    const size_t* model_indices,
    const float* all_weights,
    float* attention_values,
    float* attention_max_values,
    float* attention_denominators,
    float* attention_weighted_averages,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = sample_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t sample = index / cell_count;
  const size_t query_cell = index % cell_count;
  const float* weights =
      true2d_many_weights_for_game(all_weights, model_indices, sample, model_count, weight_count);
  if (weights == nullptr) {
    return;
  }
  const size_t matrix_offset =
      true2d_matrix_history_offset(sample, layer_index, cell_count, shape.layer_count);
  const float* layer_weights =
      weights + true2d_attention_weights_offset(shape.row_count, shape.layer_count);
  const float query_cell_value = matrices[matrix_offset + query_cell];
  float combined_context = 0.0f;
  for (size_t head_index = 0; head_index < shape.head_count; ++head_index) {
    const float* head_weights =
        layer_weights + head_index * TRUE2D_ATTENTION_HEAD_WEIGHTS;
    const float query = query_cell_value * head_weights[TRUE2D_HEAD_QUERY_SCALE_INDEX] +
                        head_weights[TRUE2D_HEAD_QUERY_BIAS_INDEX];
    float max_score = -FLT_MAX;
    for (size_t key_cell = 0; key_cell < cell_count; ++key_cell) {
      const float key_cell_value = matrices[matrix_offset + key_cell];
      const float key = key_cell_value * head_weights[TRUE2D_HEAD_KEY_SCALE_INDEX] +
                        head_weights[TRUE2D_HEAD_KEY_BIAS_INDEX];
      max_score = fmaxf(max_score, query * key);
    }
    float denominator = 0.0f;
    float weighted_value = 0.0f;
    for (size_t key_cell = 0; key_cell < cell_count; ++key_cell) {
      const float key_cell_value = matrices[matrix_offset + key_cell];
      const float key = key_cell_value * head_weights[TRUE2D_HEAD_KEY_SCALE_INDEX] +
                        head_weights[TRUE2D_HEAD_KEY_BIAS_INDEX];
      const float value = key_cell_value * head_weights[TRUE2D_HEAD_VALUE_SCALE_INDEX] +
                          head_weights[TRUE2D_HEAD_VALUE_BIAS_INDEX];
      const float attention = expf(query * key - max_score);
      denominator += attention;
      weighted_value += attention * value;
    }
    const float weighted_average = weighted_value / denominator;
    const size_t stats_offset = true2d_attention_head_history_offset(
                                    sample,
                                    head_index,
                                    cell_count,
                                    shape.head_count) +
                                query_cell;
    attention_max_values[stats_offset] = max_score;
    attention_denominators[stats_offset] = denominator;
    attention_weighted_averages[stats_offset] = weighted_average;
    combined_context += weighted_average * head_weights[TRUE2D_HEAD_OUTPUT_SCALE_INDEX];
  }
  attention_values
      [true2d_attention_history_offset(sample, cell_count) + query_cell] = combined_context;
}

__global__ void true2d_train_layer_apply_forward_kernel(
    const float* matrices,
    const float* attention_values,
    const size_t* model_indices,
    const float* all_weights,
    float* next_matrices,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = sample_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t sample = index / cell_count;
  const size_t cell_index = index % cell_count;
  const float* weights =
      true2d_many_weights_for_game(all_weights, model_indices, sample, model_count, weight_count);
  if (weights == nullptr) {
    return;
  }
  const float* layer_weights =
      weights + true2d_attention_weights_offset(shape.row_count, shape.layer_count);
  const float* shared_weights =
      layer_weights + shape.head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  const size_t matrix_offset =
      true2d_matrix_history_offset(sample, layer_index, cell_count, shape.layer_count);
  const size_t attention_offset =
      true2d_attention_history_offset(sample, cell_count);
  const float cell_value = matrices[matrix_offset + cell_index];
  const float residual_attention =
      cell_value * shared_weights[TRUE2D_LAYER_RESIDUAL_SCALE_INDEX] +
      attention_values[attention_offset + cell_index] +
      shared_weights[TRUE2D_LAYER_ATTENTION_BIAS_INDEX];
  const float ffn_hidden =
      tanhf(residual_attention * shared_weights[TRUE2D_LAYER_FFN_INPUT_SCALE_INDEX] +
            shared_weights[TRUE2D_LAYER_FFN_INPUT_BIAS_INDEX]);
  const float value =
      residual_attention +
      ffn_hidden * shared_weights[TRUE2D_LAYER_FFN_OUTPUT_SCALE_INDEX] +
      shared_weights[TRUE2D_LAYER_FFN_OUTPUT_BIAS_INDEX];
  next_matrices
      [true2d_matrix_history_offset(sample, layer_index + 1, cell_count, shape.layer_count) +
       cell_index] = tanhf(value);
}

__global__ void true2d_train_matrix_layer_forward_kernel(
    const float* matrices,
    const size_t* model_indices,
    const float* all_weights,
    float* next_matrices,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = sample_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t sample = index / cell_count;
  const size_t cell_index = index % cell_count;
  const float* weights =
      true2d_many_weights_for_game(all_weights, model_indices, sample, model_count, weight_count);
  if (weights == nullptr) {
    return;
  }
  const size_t attention_index = true2d_attention_layer_index(shape.layer_count);
  const size_t input_state = true2d_matrix_layer_input_state(layer_index, attention_index);
  const size_t output_state = true2d_matrix_layer_output_state(layer_index, attention_index);
  const size_t input_offset =
      true2d_matrix_history_offset(sample, input_state, cell_count, shape.layer_count);
  const size_t output_offset =
      true2d_matrix_history_offset(sample, output_state, cell_count, shape.layer_count);
  const size_t layer_offset =
      true2d_matrix_layer_weights_offset(shape.row_count) +
      layer_index * true2d_matrix_layer_weight_count(shape.row_count);
  const float* cell_weights =
      weights + layer_offset + cell_index * TRUE2D_MATRIX_LAYER_CELL_WEIGHTS;
  next_matrices[output_offset + cell_index] =
      true2d_matrix_layer_forward_value(matrices[input_offset + cell_index], cell_weights);
}

__global__ void true2d_train_output_loss_backward_kernel(
    const float* input_rows,
    const float* output_rows,
    const int* rewards,
    const float* target_action_rewards,
    const size_t* model_indices,
    const float* matrices,
    const float* all_weights,
    float* grad_matrices,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    OrbitWarsCudaTrue2DTrainingConfig training_config) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t value_count = sample_count * shape.row_count;
  if (index >= value_count) {
    return;
  }

  const size_t sample = index / shape.row_count;
  const size_t source_row = index % shape.row_count;
  const size_t model_index = model_indices[sample];
  if (model_index >= model_count) {
    return;
  }
  const size_t source_input =
      (sample * shape.row_count + source_row) * TRUE2D_INPUT_FEATURE_COUNT;
  const float source_owner = input_rows[source_input];
  if (!owner_class_matches(source_owner, training_config.owner_class_own)) {
    return;
  }
  const float reward_scale =
      static_cast<float>(rewards[sample]) / training_config.reward_normalizer;

  size_t active_count = 0;
  for (size_t row = 0; row < shape.row_count; ++row) {
    const float owner = input_rows[(sample * shape.row_count + row) * TRUE2D_INPUT_FEATURE_COUNT];
    if (!owner_class_matches(owner, training_config.owner_class_empty)) {
      ++active_count;
    }
  }
  if (active_count == 0) {
    return;
  }
  const size_t output_offset =
      (sample * shape.row_count + source_row) * TRUE2D_OUTPUT_FEATURE_COUNT;
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t final_matrix_offset =
      true2d_matrix_history_offset(
          sample, true2d_final_matrix_state(shape.layer_count), cell_count, shape.layer_count);
  const size_t row_offset = final_matrix_offset + source_row * shape.row_count;
  const float* weights = all_weights + model_index * weight_count;
  const size_t target_bias_offset =
      true2d_target_output_bias_offset(shape.row_count, shape.layer_count, shape.head_count);
  const size_t send_bias_offset =
      true2d_send_output_bias_offset(shape.row_count, shape.layer_count, shape.head_count);
  const size_t output_weights_offset =
      true2d_output_scalar_weights_offset(shape.row_count, shape.layer_count, shape.head_count);

  for (size_t action_index = 0; action_index < TRUE2D_ACTION_TARGETS_PER_SOURCE; ++action_index) {
    const size_t action_output_offset =
        output_offset + action_index * TRUE2D_ACTION_TARGET_FEATURES;
    const size_t target_action_reward_offset =
        (sample * shape.row_count + source_row) * TRUE2D_ACTION_TARGETS_PER_SOURCE +
        action_index;
    const float target_reward_scale =
        reward_scale + target_action_rewards[target_action_reward_offset];
    const float stored_target_fraction = clamp_probability_device(
        output_rows[action_output_offset + TRUE2D_OUTPUT_TARGET_FEATURE_INDEX]);
    size_t selected_active_index =
        static_cast<size_t>(floorf(stored_target_fraction * static_cast<float>(active_count)));
    if (selected_active_index >= active_count) {
      selected_active_index = active_count - 1;
    }
    size_t selected_target_row = source_row;
    size_t active_index = 0;
    for (size_t row = 0; row < shape.row_count; ++row) {
      const float owner =
          input_rows[(sample * shape.row_count + row) * TRUE2D_INPUT_FEATURE_COUNT];
      if (owner_class_matches(owner, training_config.owner_class_empty)) {
        continue;
      }
      if (active_index == selected_active_index) {
        selected_target_row = row;
        break;
      }
      ++active_index;
    }

    const size_t action_bias_offset = action_index * shape.row_count;
    const size_t action_weights_offset =
        output_weights_offset + action_index * TRUE2D_OUTPUT_SCALAR_WEIGHTS;
    const float target_scale =
        weights[action_weights_offset + TRUE2D_TARGET_OUTPUT_SCALE_INDEX];
    const float send_key_scale = weights[action_weights_offset + TRUE2D_SEND_KEY_SCALE_INDEX];
    const float send_value_scale = weights[action_weights_offset + TRUE2D_SEND_VALUE_SCALE_INDEX];
    const float send_output_scale =
        weights[action_weights_offset + TRUE2D_SEND_OUTPUT_SCALE_INDEX];

    float target_max_score = -FLT_MAX;
    for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
      const float score = matrices[row_offset + target_row] * target_scale +
                          weights[target_bias_offset + action_bias_offset + target_row];
      target_max_score = fmaxf(target_max_score, score);
    }
    float target_denominator = 0.0f;
    for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
      const float score = matrices[row_offset + target_row] * target_scale +
                          weights[target_bias_offset + action_bias_offset + target_row];
      target_denominator += expf(score - target_max_score);
    }
    if (target_reward_scale != 0.0f) {
      for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
        const float cell_value = matrices[row_offset + target_row];
        const float score =
            cell_value * target_scale +
            weights[target_bias_offset + action_bias_offset + target_row];
        const float probability = expf(score - target_max_score) / target_denominator;
        const float expected = target_row == selected_target_row ? 1.0f : 0.0f;
        const float d_score = target_reward_scale * (probability - expected);
        grad_matrices[row_offset + target_row] += d_score * target_scale;
      }
    }

    const float selected_target_owner =
        input_rows[(sample * shape.row_count + selected_target_row) *
                   TRUE2D_INPUT_FEATURE_COUNT];
    if (reward_scale == 0.0f || selected_target_row == source_row ||
        owner_class_matches(selected_target_owner, training_config.owner_class_absent_on_map)) {
      continue;
    }

    float send_max_score = -FLT_MAX;
    for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
      const float score = matrices[row_offset + target_row] * send_key_scale +
                          weights[send_bias_offset + action_bias_offset + target_row];
      send_max_score = fmaxf(send_max_score, score);
    }
    float send_denominator = 0.0f;
    float send_weighted_value = 0.0f;
    for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
      const float cell_value = matrices[row_offset + target_row];
      const float score =
          cell_value * send_key_scale + weights[send_bias_offset + action_bias_offset + target_row];
      const float attention = expf(score - send_max_score);
      send_denominator += attention;
      send_weighted_value += attention * cell_value;
    }
    const float send_average = send_weighted_value / send_denominator;
    const float send_logit =
        weights[action_weights_offset + TRUE2D_SEND_OUTPUT_BIAS_INDEX] +
        send_average * send_value_scale;
    const float send_probability = sigmoid_device(send_logit * send_output_scale);
    const float stored_send_fraction = clamp_probability_device(
        output_rows[action_output_offset + TRUE2D_OUTPUT_SEND_FEATURE_INDEX]);
    const float send_action =
        stored_send_fraction >= training_config.send_threshold ? 1.0f : 0.0f;
    const float d_send_scaled_logit = reward_scale * (send_probability - send_action);
    const float d_send_logit = d_send_scaled_logit * send_output_scale;
    const float d_send_average = d_send_logit * send_value_scale;
    for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
      const float cell_value = matrices[row_offset + target_row];
      const float score =
          cell_value * send_key_scale + weights[send_bias_offset + action_bias_offset + target_row];
      const float attention = expf(score - send_max_score) / send_denominator;
      const float d_cell_from_average = d_send_average * attention;
      const float d_send_score = d_send_average * attention * (cell_value - send_average);
      grad_matrices[row_offset + target_row] +=
          d_cell_from_average + d_send_score * send_key_scale;
    }
  }
}

__global__ void true2d_train_output_gradients_kernel(
    const float* input_rows,
    const float* output_rows,
    const int* rewards,
    const float* target_action_rewards,
    const size_t* model_indices,
    const float* matrices,
    const float* all_weights,
    float* gradients,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    OrbitWarsCudaTrue2DTrainingConfig training_config) {
  const size_t output_gradient_count =
      shape.row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE +
      shape.row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE +
      TRUE2D_OUTPUT_SCALAR_WEIGHTS * TRUE2D_ACTION_TARGETS_PER_SOURCE;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t value_count = model_count * output_gradient_count;
  if (index >= value_count) {
    return;
  }

  const size_t model_index = index / output_gradient_count;
  const size_t output_gradient_index = index % output_gradient_count;
  const size_t target_bias_gradient_count =
      shape.row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE;
  const size_t send_bias_gradient_count =
      shape.row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE;
  size_t action_index = 0;
  size_t action_parameter_index = 0;
  size_t scalar_parameter_index = TRUE2D_OUTPUT_SCALAR_WEIGHTS;
  bool is_target_bias_gradient = false;
  bool is_send_bias_gradient = false;
  bool is_scalar_gradient = false;
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t target_bias_offset =
      true2d_target_output_bias_offset(shape.row_count, shape.layer_count, shape.head_count);
  const size_t send_bias_offset =
      true2d_send_output_bias_offset(shape.row_count, shape.layer_count, shape.head_count);
  const size_t output_weights_offset =
      true2d_output_scalar_weights_offset(shape.row_count, shape.layer_count, shape.head_count);
  size_t target_weight_offset = output_weights_offset;
  if (output_gradient_index < target_bias_gradient_count) {
    action_index = output_gradient_index / shape.row_count;
    action_parameter_index = output_gradient_index % shape.row_count;
    target_weight_offset = target_bias_offset + output_gradient_index;
    is_target_bias_gradient = true;
  } else if (output_gradient_index < target_bias_gradient_count + send_bias_gradient_count) {
    const size_t send_index = output_gradient_index - target_bias_gradient_count;
    action_index = send_index / shape.row_count;
    action_parameter_index = send_index % shape.row_count;
    target_weight_offset = send_bias_offset + send_index;
    is_send_bias_gradient = true;
  } else {
    const size_t scalar_index =
        output_gradient_index - target_bias_gradient_count - send_bias_gradient_count;
    action_index = scalar_index / TRUE2D_OUTPUT_SCALAR_WEIGHTS;
    scalar_parameter_index = scalar_index % TRUE2D_OUTPUT_SCALAR_WEIGHTS;
    target_weight_offset = output_weights_offset + scalar_index;
    is_scalar_gradient = true;
  }

  const float* weights = all_weights + model_index * weight_count;
  const size_t action_bias_offset = action_index * shape.row_count;
  const size_t action_weights_offset =
      output_weights_offset + action_index * TRUE2D_OUTPUT_SCALAR_WEIGHTS;
  const float target_scale =
      weights[action_weights_offset + TRUE2D_TARGET_OUTPUT_SCALE_INDEX];
  const float send_key_scale = weights[action_weights_offset + TRUE2D_SEND_KEY_SCALE_INDEX];
  const float send_value_scale = weights[action_weights_offset + TRUE2D_SEND_VALUE_SCALE_INDEX];
  const float send_output_scale =
      weights[action_weights_offset + TRUE2D_SEND_OUTPUT_SCALE_INDEX];
  float gradient = 0.0f;

  for (size_t sample = 0; sample < sample_count; ++sample) {
    if (model_indices[sample] != model_index) {
      continue;
    }
    const float reward_scale =
        static_cast<float>(rewards[sample]) / training_config.reward_normalizer;
    size_t active_count = 0;
    for (size_t row = 0; row < shape.row_count; ++row) {
      const float owner = input_rows[(sample * shape.row_count + row) * TRUE2D_INPUT_FEATURE_COUNT];
      if (!owner_class_matches(owner, training_config.owner_class_empty)) {
        ++active_count;
      }
    }
    if (active_count == 0) {
      continue;
    }
    for (size_t source_row = 0; source_row < shape.row_count; ++source_row) {
      const size_t source_input =
          (sample * shape.row_count + source_row) * TRUE2D_INPUT_FEATURE_COUNT;
      const float source_owner = input_rows[source_input];
      if (!owner_class_matches(source_owner, training_config.owner_class_own)) {
        continue;
      }

      const size_t output_offset =
          (sample * shape.row_count + source_row) * TRUE2D_OUTPUT_FEATURE_COUNT;
      const size_t action_output_offset =
          output_offset + action_index * TRUE2D_ACTION_TARGET_FEATURES;
      const size_t target_action_reward_offset =
          (sample * shape.row_count + source_row) * TRUE2D_ACTION_TARGETS_PER_SOURCE +
          action_index;
      const float target_reward_scale =
          reward_scale + target_action_rewards[target_action_reward_offset];
      const float stored_target_fraction = clamp_probability_device(
          output_rows[action_output_offset + TRUE2D_OUTPUT_TARGET_FEATURE_INDEX]);
      size_t selected_active_index =
          static_cast<size_t>(floorf(stored_target_fraction * static_cast<float>(active_count)));
      if (selected_active_index >= active_count) {
        selected_active_index = active_count - 1;
      }
      size_t selected_target_row = source_row;
      size_t active_index = 0;
      for (size_t row = 0; row < shape.row_count; ++row) {
        const float owner =
            input_rows[(sample * shape.row_count + row) * TRUE2D_INPUT_FEATURE_COUNT];
        if (owner_class_matches(owner, training_config.owner_class_empty)) {
          continue;
        }
        if (active_index == selected_active_index) {
          selected_target_row = row;
          break;
        }
        ++active_index;
      }

      const size_t final_matrix_offset =
          true2d_matrix_history_offset(
              sample, true2d_final_matrix_state(shape.layer_count), cell_count, shape.layer_count);
      const size_t row_offset = final_matrix_offset + source_row * shape.row_count;
      float target_max_score = -FLT_MAX;
      for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
        const float score = matrices[row_offset + target_row] * target_scale +
                            weights[target_bias_offset + action_bias_offset + target_row];
        target_max_score = fmaxf(target_max_score, score);
      }
      float target_denominator = 0.0f;
      for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
        const float score = matrices[row_offset + target_row] * target_scale +
                            weights[target_bias_offset + action_bias_offset + target_row];
        target_denominator += expf(score - target_max_score);
      }
      if (target_reward_scale != 0.0f) {
        for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
          const float cell_value = matrices[row_offset + target_row];
          const float score =
              cell_value * target_scale +
              weights[target_bias_offset + action_bias_offset + target_row];
          const float probability = expf(score - target_max_score) / target_denominator;
          const float expected = target_row == selected_target_row ? 1.0f : 0.0f;
          const float d_score = target_reward_scale * (probability - expected);
          if (is_target_bias_gradient && action_parameter_index == target_row) {
            gradient += d_score;
          } else if (is_scalar_gradient &&
                     scalar_parameter_index == TRUE2D_TARGET_OUTPUT_SCALE_INDEX) {
            gradient += d_score * cell_value;
          }
        }
      }

      const float selected_target_owner =
          input_rows[(sample * shape.row_count + selected_target_row) *
                     TRUE2D_INPUT_FEATURE_COUNT];
      if (reward_scale == 0.0f || selected_target_row == source_row ||
          owner_class_matches(selected_target_owner, training_config.owner_class_absent_on_map)) {
        continue;
      }

      float send_max_score = -FLT_MAX;
      for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
        const float score = matrices[row_offset + target_row] * send_key_scale +
                            weights[send_bias_offset + action_bias_offset + target_row];
        send_max_score = fmaxf(send_max_score, score);
      }
      float send_denominator = 0.0f;
      float send_weighted_value = 0.0f;
      for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
        const float cell_value = matrices[row_offset + target_row];
        const float score =
            cell_value * send_key_scale + weights[send_bias_offset + action_bias_offset + target_row];
        const float attention = expf(score - send_max_score);
        send_denominator += attention;
        send_weighted_value += attention * cell_value;
      }
      const float send_average = send_weighted_value / send_denominator;
      const float send_logit =
          weights[action_weights_offset + TRUE2D_SEND_OUTPUT_BIAS_INDEX] +
          send_average * send_value_scale;
      const float send_probability = sigmoid_device(send_logit * send_output_scale);
      const float stored_send_fraction = clamp_probability_device(
          output_rows[action_output_offset + TRUE2D_OUTPUT_SEND_FEATURE_INDEX]);
      const float send_action =
          stored_send_fraction >= training_config.send_threshold ? 1.0f : 0.0f;
      const float d_send_scaled_logit = reward_scale * (send_probability - send_action);
      if (is_scalar_gradient && scalar_parameter_index == TRUE2D_SEND_OUTPUT_SCALE_INDEX) {
        gradient += d_send_scaled_logit * send_logit;
      }
      const float d_send_logit = d_send_scaled_logit * send_output_scale;
      if (is_scalar_gradient && scalar_parameter_index == TRUE2D_SEND_OUTPUT_BIAS_INDEX) {
        gradient += d_send_logit;
      } else if (is_scalar_gradient &&
                 scalar_parameter_index == TRUE2D_SEND_VALUE_SCALE_INDEX) {
        gradient += d_send_logit * send_average;
      }
      const float d_send_average = d_send_logit * send_value_scale;
      for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
        const float cell_value = matrices[row_offset + target_row];
        const float score =
            cell_value * send_key_scale + weights[send_bias_offset + action_bias_offset + target_row];
        const float attention = expf(score - send_max_score) / send_denominator;
        const float d_send_score = d_send_average * attention * (cell_value - send_average);
        if (is_send_bias_gradient && action_parameter_index == target_row) {
          gradient += d_send_score;
        } else if (is_scalar_gradient &&
                   scalar_parameter_index == TRUE2D_SEND_KEY_SCALE_INDEX) {
          gradient += d_send_score * cell_value;
        }
      }
    }
  }

  gradients[model_index * weight_count + target_weight_offset] += gradient;
}

__global__ void true2d_train_layer_apply_backward_kernel(
    const float* matrices,
    const float* attention_values,
    const size_t* model_indices,
    const float* all_weights,
    float* grad_matrices,
    float* grad_attention_values,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = sample_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t sample = index / cell_count;
  const size_t cell_index = index % cell_count;
  const size_t model_index = model_indices[sample];
  if (model_index >= model_count) {
    return;
  }
  const float* weights = all_weights + model_index * weight_count;
  const size_t layer_offset =
      true2d_attention_weights_offset(shape.row_count, shape.layer_count);
  const float* layer_weights = weights + layer_offset;
  const float* shared_weights =
      layer_weights + shape.head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  const size_t input_offset =
      true2d_matrix_history_offset(sample, layer_index, cell_count, shape.layer_count);
  const size_t next_offset =
      true2d_matrix_history_offset(sample, layer_index + 1, cell_count, shape.layer_count);
  const size_t attention_offset =
      true2d_attention_history_offset(sample, cell_count);
  const float cell_value = matrices[input_offset + cell_index];
  const float residual_attention =
      cell_value * shared_weights[TRUE2D_LAYER_RESIDUAL_SCALE_INDEX] +
      attention_values[attention_offset + cell_index] +
      shared_weights[TRUE2D_LAYER_ATTENTION_BIAS_INDEX];
  const float ffn_hidden =
      tanhf(residual_attention * shared_weights[TRUE2D_LAYER_FFN_INPUT_SCALE_INDEX] +
            shared_weights[TRUE2D_LAYER_FFN_INPUT_BIAS_INDEX]);
  const float next_value = matrices[next_offset + cell_index];
  const float d_next = grad_matrices[next_offset + cell_index];
  const float d_value = d_next * (1.0f - next_value * next_value);
  const float d_ffn_hidden = d_value * shared_weights[TRUE2D_LAYER_FFN_OUTPUT_SCALE_INDEX];
  const float d_ffn_input = d_ffn_hidden * (1.0f - ffn_hidden * ffn_hidden);
  const float d_residual_attention =
      d_value + d_ffn_input * shared_weights[TRUE2D_LAYER_FFN_INPUT_SCALE_INDEX];
  grad_matrices[input_offset + cell_index] +=
      d_residual_attention * shared_weights[TRUE2D_LAYER_RESIDUAL_SCALE_INDEX];
  grad_attention_values[attention_offset + cell_index] = d_residual_attention;
}

__global__ void true2d_train_layer_shared_gradients_kernel(
    const float* matrices,
    const float* attention_values,
    const float* grad_matrices,
    const size_t* model_indices,
    const float* all_weights,
    float* gradients,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t value_count = model_count * TRUE2D_LAYER_SHARED_WEIGHTS;
  if (index >= value_count) {
    return;
  }

  const size_t model_index = index / TRUE2D_LAYER_SHARED_WEIGHTS;
  const size_t shared_index = index % TRUE2D_LAYER_SHARED_WEIGHTS;
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const float* weights = all_weights + model_index * weight_count;
  const size_t layer_offset =
      true2d_attention_weights_offset(shape.row_count, shape.layer_count);
  const size_t shared_offset = layer_offset + shape.head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  const float* layer_weights = weights + layer_offset;
  const float* shared_weights =
      layer_weights + shape.head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  float gradient = 0.0f;

  for (size_t sample = 0; sample < sample_count; ++sample) {
    if (model_indices[sample] != model_index) {
      continue;
    }
    const size_t input_offset =
        true2d_matrix_history_offset(sample, layer_index, cell_count, shape.layer_count);
    const size_t next_offset =
        true2d_matrix_history_offset(sample, layer_index + 1, cell_count, shape.layer_count);
    const size_t attention_offset =
        true2d_attention_history_offset(sample, cell_count);
    for (size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
      const float cell_value = matrices[input_offset + cell_index];
      const float residual_attention =
          cell_value * shared_weights[TRUE2D_LAYER_RESIDUAL_SCALE_INDEX] +
          attention_values[attention_offset + cell_index] +
          shared_weights[TRUE2D_LAYER_ATTENTION_BIAS_INDEX];
      const float ffn_hidden =
          tanhf(residual_attention * shared_weights[TRUE2D_LAYER_FFN_INPUT_SCALE_INDEX] +
                shared_weights[TRUE2D_LAYER_FFN_INPUT_BIAS_INDEX]);
      const float next_value = matrices[next_offset + cell_index];
      const float d_next = grad_matrices[next_offset + cell_index];
      const float d_value = d_next * (1.0f - next_value * next_value);
      const float d_ffn_hidden = d_value * shared_weights[TRUE2D_LAYER_FFN_OUTPUT_SCALE_INDEX];
      const float d_ffn_input = d_ffn_hidden * (1.0f - ffn_hidden * ffn_hidden);
      const float d_residual_attention =
          d_value + d_ffn_input * shared_weights[TRUE2D_LAYER_FFN_INPUT_SCALE_INDEX];
      if (shared_index == TRUE2D_LAYER_RESIDUAL_SCALE_INDEX) {
        gradient += d_residual_attention * cell_value;
      } else if (shared_index == TRUE2D_LAYER_ATTENTION_BIAS_INDEX) {
        gradient += d_residual_attention;
      } else if (shared_index == TRUE2D_LAYER_FFN_INPUT_SCALE_INDEX) {
        gradient += d_ffn_input * residual_attention;
      } else if (shared_index == TRUE2D_LAYER_FFN_INPUT_BIAS_INDEX) {
        gradient += d_ffn_input;
      } else if (shared_index == TRUE2D_LAYER_FFN_OUTPUT_SCALE_INDEX) {
        gradient += d_value * ffn_hidden;
      } else if (shared_index == TRUE2D_LAYER_FFN_OUTPUT_BIAS_INDEX) {
        gradient += d_value;
      }
    }
  }

  gradients[model_index * weight_count + shared_offset + shared_index] += gradient;
}

__global__ void true2d_train_matrix_layer_backward_kernel(
    const float* matrices,
    const size_t* model_indices,
    const float* all_weights,
    float* grad_matrices,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = sample_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t sample = index / cell_count;
  const size_t cell_index = index % cell_count;
  const size_t model_index = model_indices[sample];
  if (model_index >= model_count) {
    return;
  }

  const size_t attention_index = true2d_attention_layer_index(shape.layer_count);
  const size_t input_state = true2d_matrix_layer_input_state(layer_index, attention_index);
  const size_t output_state = true2d_matrix_layer_output_state(layer_index, attention_index);
  const size_t input_offset =
      true2d_matrix_history_offset(sample, input_state, cell_count, shape.layer_count);
  const size_t output_offset =
      true2d_matrix_history_offset(sample, output_state, cell_count, shape.layer_count);
  const float* weights = all_weights + model_index * weight_count;
  const size_t layer_offset =
      true2d_matrix_layer_weights_offset(shape.row_count) +
      layer_index * true2d_matrix_layer_weight_count(shape.row_count);
  const float* cell_weights =
      weights + layer_offset + cell_index * TRUE2D_MATRIX_LAYER_CELL_WEIGHTS;
  const float cell_value = matrices[input_offset + cell_index];
  const float hidden_a =
      tanhf(cell_value * cell_weights[TRUE2D_MATRIX_HIDDEN_A_SCALE_INDEX] +
            cell_weights[TRUE2D_MATRIX_HIDDEN_A_BIAS_INDEX]);
  const float hidden_b =
      tanhf(cell_value * cell_weights[TRUE2D_MATRIX_HIDDEN_B_CELL_SCALE_INDEX] +
            hidden_a * cell_weights[TRUE2D_MATRIX_HIDDEN_B_HIDDEN_SCALE_INDEX] +
            cell_weights[TRUE2D_MATRIX_HIDDEN_B_BIAS_INDEX]);
  const float gate =
      sigmoid_device(cell_value * cell_weights[TRUE2D_MATRIX_GATE_SCALE_INDEX] +
                     cell_weights[TRUE2D_MATRIX_GATE_BIAS_INDEX]);
  const float update =
      hidden_b * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_B_SCALE_INDEX] +
      hidden_a * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_A_SCALE_INDEX] +
      cell_weights[TRUE2D_MATRIX_UPDATE_BIAS_INDEX];
  const float next_value = matrices[output_offset + cell_index];
  const float d_next = grad_matrices[output_offset + cell_index];
  const float d_pre_activation = d_next * (1.0f - next_value * next_value);
  float d_cell = d_pre_activation * (1.0f + cell_weights[TRUE2D_MATRIX_RESIDUAL_SCALE_INDEX]);
  const float d_gate = d_pre_activation * update;
  const float d_gate_input = d_gate * gate * (1.0f - gate);
  d_cell += d_gate_input * cell_weights[TRUE2D_MATRIX_GATE_SCALE_INDEX];
  const float d_update = d_pre_activation * gate;
  float d_hidden_b = d_update * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_B_SCALE_INDEX];
  float d_hidden_a = d_update * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_A_SCALE_INDEX];
  const float d_hidden_b_input = d_hidden_b * (1.0f - hidden_b * hidden_b);
  d_cell += d_hidden_b_input * cell_weights[TRUE2D_MATRIX_HIDDEN_B_CELL_SCALE_INDEX];
  d_hidden_a += d_hidden_b_input * cell_weights[TRUE2D_MATRIX_HIDDEN_B_HIDDEN_SCALE_INDEX];
  const float d_hidden_a_input = d_hidden_a * (1.0f - hidden_a * hidden_a);
  d_cell += d_hidden_a_input * cell_weights[TRUE2D_MATRIX_HIDDEN_A_SCALE_INDEX];
  grad_matrices[input_offset + cell_index] += d_cell;
}

__global__ void true2d_train_matrix_layer_gradients_kernel(
    const float* matrices,
    const float* grad_matrices,
    const size_t* model_indices,
    const float* all_weights,
    float* gradients,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t matrix_layer_weight_count = true2d_matrix_layer_weight_count(shape.row_count);
  const size_t value_count =
      model_count * shape.layer_count * matrix_layer_weight_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t parameter_index = index % TRUE2D_MATRIX_LAYER_CELL_WEIGHTS;
  const size_t cell_index =
      (index / TRUE2D_MATRIX_LAYER_CELL_WEIGHTS) % cell_count;
  const size_t layer_index =
      (index / matrix_layer_weight_count) % shape.layer_count;
  const size_t model_index =
      index / (shape.layer_count * matrix_layer_weight_count);
  const size_t attention_index = true2d_attention_layer_index(shape.layer_count);
  const size_t input_state = true2d_matrix_layer_input_state(layer_index, attention_index);
  const size_t output_state = true2d_matrix_layer_output_state(layer_index, attention_index);
  const float* weights = all_weights + model_index * weight_count;
  const size_t layer_offset =
      true2d_matrix_layer_weights_offset(shape.row_count) +
      layer_index * matrix_layer_weight_count;
  const float* cell_weights =
      weights + layer_offset + cell_index * TRUE2D_MATRIX_LAYER_CELL_WEIGHTS;
  float gradient = 0.0f;

  for (size_t sample = 0; sample < sample_count; ++sample) {
    if (model_indices[sample] != model_index) {
      continue;
    }
    const size_t input_offset =
        true2d_matrix_history_offset(sample, input_state, cell_count, shape.layer_count);
    const size_t output_offset =
        true2d_matrix_history_offset(sample, output_state, cell_count, shape.layer_count);
    const float cell_value = matrices[input_offset + cell_index];
    const float hidden_a =
        tanhf(cell_value * cell_weights[TRUE2D_MATRIX_HIDDEN_A_SCALE_INDEX] +
              cell_weights[TRUE2D_MATRIX_HIDDEN_A_BIAS_INDEX]);
    const float hidden_b =
        tanhf(cell_value * cell_weights[TRUE2D_MATRIX_HIDDEN_B_CELL_SCALE_INDEX] +
              hidden_a * cell_weights[TRUE2D_MATRIX_HIDDEN_B_HIDDEN_SCALE_INDEX] +
              cell_weights[TRUE2D_MATRIX_HIDDEN_B_BIAS_INDEX]);
    const float gate =
        sigmoid_device(cell_value * cell_weights[TRUE2D_MATRIX_GATE_SCALE_INDEX] +
                       cell_weights[TRUE2D_MATRIX_GATE_BIAS_INDEX]);
    const float update =
        hidden_b * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_B_SCALE_INDEX] +
        hidden_a * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_A_SCALE_INDEX] +
        cell_weights[TRUE2D_MATRIX_UPDATE_BIAS_INDEX];
    const float next_value = matrices[output_offset + cell_index];
    const float d_next = grad_matrices[output_offset + cell_index];
    const float d_pre_activation = d_next * (1.0f - next_value * next_value);
    const float d_gate = d_pre_activation * update;
    const float d_gate_input = d_gate * gate * (1.0f - gate);
    const float d_update = d_pre_activation * gate;
    float d_hidden_b = d_update * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_B_SCALE_INDEX];
    float d_hidden_a = d_update * cell_weights[TRUE2D_MATRIX_UPDATE_HIDDEN_A_SCALE_INDEX];
    const float d_hidden_b_input = d_hidden_b * (1.0f - hidden_b * hidden_b);
    d_hidden_a += d_hidden_b_input * cell_weights[TRUE2D_MATRIX_HIDDEN_B_HIDDEN_SCALE_INDEX];
    const float d_hidden_a_input = d_hidden_a * (1.0f - hidden_a * hidden_a);

    if (parameter_index == TRUE2D_MATRIX_RESIDUAL_SCALE_INDEX) {
      gradient += d_pre_activation * cell_value;
    } else if (parameter_index == TRUE2D_MATRIX_RESIDUAL_BIAS_INDEX) {
      gradient += d_pre_activation;
    } else if (parameter_index == TRUE2D_MATRIX_HIDDEN_A_SCALE_INDEX) {
      gradient += d_hidden_a_input * cell_value;
    } else if (parameter_index == TRUE2D_MATRIX_HIDDEN_A_BIAS_INDEX) {
      gradient += d_hidden_a_input;
    } else if (parameter_index == TRUE2D_MATRIX_HIDDEN_B_CELL_SCALE_INDEX) {
      gradient += d_hidden_b_input * cell_value;
    } else if (parameter_index == TRUE2D_MATRIX_HIDDEN_B_HIDDEN_SCALE_INDEX) {
      gradient += d_hidden_b_input * hidden_a;
    } else if (parameter_index == TRUE2D_MATRIX_HIDDEN_B_BIAS_INDEX) {
      gradient += d_hidden_b_input;
    } else if (parameter_index == TRUE2D_MATRIX_GATE_SCALE_INDEX) {
      gradient += d_gate_input * cell_value;
    } else if (parameter_index == TRUE2D_MATRIX_GATE_BIAS_INDEX) {
      gradient += d_gate_input;
    } else if (parameter_index == TRUE2D_MATRIX_UPDATE_HIDDEN_B_SCALE_INDEX) {
      gradient += d_update * hidden_b;
    } else if (parameter_index == TRUE2D_MATRIX_UPDATE_HIDDEN_A_SCALE_INDEX) {
      gradient += d_update * hidden_a;
    } else if (parameter_index == TRUE2D_MATRIX_UPDATE_BIAS_INDEX) {
      gradient += d_update;
    }
  }

  gradients[model_index * weight_count + layer_offset +
            cell_index * TRUE2D_MATRIX_LAYER_CELL_WEIGHTS + parameter_index] += gradient;
}

__global__ void true2d_train_attention_backward_kernel(
    const float* matrices,
    const float* grad_attention_values,
    const float* attention_max_values,
    const float* attention_denominators,
    const float* attention_weighted_averages,
    const size_t* model_indices,
    const float* all_weights,
    float* attention_head_cell_gradients,
    float* grad_matrices,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t value_count = sample_count * cell_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t sample = index / cell_count;
  const size_t cell_index = index % cell_count;
  const size_t model_index = model_indices[sample];
  if (model_index >= model_count) {
    return;
  }
  const size_t input_offset =
      true2d_matrix_history_offset(sample, layer_index, cell_count, shape.layer_count);
  const size_t attention_offset =
      true2d_attention_history_offset(sample, cell_count);

  const float* weights = all_weights + model_index * weight_count;
  const size_t layer_offset =
      true2d_attention_weights_offset(shape.row_count, shape.layer_count);
  const float cell_value = matrices[input_offset + cell_index];
  float input_gradient = 0.0f;

  for (size_t head_index = 0; head_index < shape.head_count; ++head_index) {
    const size_t head_offset = layer_offset + head_index * TRUE2D_ATTENTION_HEAD_WEIGHTS;
    const float* head_weights = weights + head_offset;
    const float query_scale = head_weights[TRUE2D_HEAD_QUERY_SCALE_INDEX];
    const float query_bias = head_weights[TRUE2D_HEAD_QUERY_BIAS_INDEX];
    const float key_scale = head_weights[TRUE2D_HEAD_KEY_SCALE_INDEX];
    const float key_bias = head_weights[TRUE2D_HEAD_KEY_BIAS_INDEX];
    const float value_scale = head_weights[TRUE2D_HEAD_VALUE_SCALE_INDEX];
    const float value_bias = head_weights[TRUE2D_HEAD_VALUE_BIAS_INDEX];
    const float output_scale = head_weights[TRUE2D_HEAD_OUTPUT_SCALE_INDEX];
    const size_t stats_base = true2d_attention_head_history_offset(
        sample, head_index, cell_count, shape.head_count);
    float output_scale_gradient = 0.0f;
    float query_scale_gradient = 0.0f;
    float query_bias_gradient = 0.0f;

    const float query_for_cell = cell_value * query_scale + query_bias;
    const float d_attention_for_query = grad_attention_values[attention_offset + cell_index];
    float d_query = 0.0f;
    if (d_attention_for_query != 0.0f) {
      const float max_score = attention_max_values[stats_base + cell_index];
      const float denominator = attention_denominators[stats_base + cell_index];
      const float weighted_average = attention_weighted_averages[stats_base + cell_index];
      const float d_weighted_average = d_attention_for_query * output_scale;
      for (size_t key_cell = 0; key_cell < cell_count; ++key_cell) {
        const float key_cell_value = matrices[input_offset + key_cell];
        const float key = key_cell_value * key_scale + key_bias;
        const float value = key_cell_value * value_scale + value_bias;
        const float probability = expf(query_for_cell * key - max_score) / denominator;
        const float d_score =
            d_weighted_average * probability * (value - weighted_average);
        d_query += d_score * key;
      }
      output_scale_gradient = d_attention_for_query * weighted_average;
      query_scale_gradient = d_query * cell_value;
      query_bias_gradient = d_query;
      input_gradient += d_query * query_scale;
    }

    const float key_for_cell = cell_value * key_scale + key_bias;
    const float value_for_cell = cell_value * value_scale + value_bias;
    float key_scale_gradient = 0.0f;
    float key_bias_gradient = 0.0f;
    float value_scale_gradient = 0.0f;
    float value_bias_gradient = 0.0f;
    for (size_t query_cell = 0; query_cell < cell_count; ++query_cell) {
      const float d_attention = grad_attention_values[attention_offset + query_cell];
      if (d_attention == 0.0f) {
        continue;
      }
      const float query_cell_value = matrices[input_offset + query_cell];
      const float query = query_cell_value * query_scale + query_bias;
      const float max_score = attention_max_values[stats_base + query_cell];
      const float denominator = attention_denominators[stats_base + query_cell];
      const float weighted_average = attention_weighted_averages[stats_base + query_cell];
      const float probability = expf(query * key_for_cell - max_score) / denominator;
      const float d_weighted_average = d_attention * output_scale;
      const float d_value = d_weighted_average * probability;
      const float d_score = d_weighted_average * probability * (value_for_cell - weighted_average);
      const float d_key = d_score * query;
      value_scale_gradient += d_value * cell_value;
      value_bias_gradient += d_value;
      key_scale_gradient += d_key * cell_value;
      key_bias_gradient += d_key;
      input_gradient += d_value * value_scale + d_key * key_scale;
    }
    const size_t cell_gradient_base = true2d_attention_head_cell_gradient_offset(
        sample,
        head_index,
        cell_index,
        0,
        cell_count,
        shape.head_count);
    attention_head_cell_gradients[cell_gradient_base + TRUE2D_HEAD_QUERY_SCALE_INDEX] =
        query_scale_gradient;
    attention_head_cell_gradients[cell_gradient_base + TRUE2D_HEAD_QUERY_BIAS_INDEX] =
        query_bias_gradient;
    attention_head_cell_gradients[cell_gradient_base + TRUE2D_HEAD_KEY_SCALE_INDEX] =
        key_scale_gradient;
    attention_head_cell_gradients[cell_gradient_base + TRUE2D_HEAD_KEY_BIAS_INDEX] =
        key_bias_gradient;
    attention_head_cell_gradients[cell_gradient_base + TRUE2D_HEAD_VALUE_SCALE_INDEX] =
        value_scale_gradient;
    attention_head_cell_gradients[cell_gradient_base + TRUE2D_HEAD_VALUE_BIAS_INDEX] =
        value_bias_gradient;
    attention_head_cell_gradients[cell_gradient_base + TRUE2D_HEAD_OUTPUT_SCALE_INDEX] =
        output_scale_gradient;
  }

  grad_matrices[input_offset + cell_index] += input_gradient;
}

__global__ void true2d_train_attention_head_gradients_kernel(
    const float* attention_head_cell_gradients,
    const size_t* model_indices,
    float* gradients,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape,
    size_t layer_index) {
  const size_t value_count =
      model_count * shape.head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t parameter_index = index % TRUE2D_ATTENTION_HEAD_WEIGHTS;
  const size_t head_index = (index / TRUE2D_ATTENTION_HEAD_WEIGHTS) % shape.head_count;
  const size_t model_index = index / (shape.head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS);
  const size_t cell_count = true2d_cell_count(shape.row_count);
  float gradient = 0.0f;
  for (size_t sample = 0; sample < sample_count; ++sample) {
    if (model_indices[sample] != model_index) {
      continue;
    }
    for (size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
      gradient += attention_head_cell_gradients[true2d_attention_head_cell_gradient_offset(
          sample,
          head_index,
          cell_index,
          parameter_index,
          cell_count,
          shape.head_count)];
    }
  }

  const size_t head_offset =
      true2d_attention_weights_offset(shape.row_count, shape.layer_count) +
      head_index * TRUE2D_ATTENTION_HEAD_WEIGHTS;
  gradients[model_index * weight_count + head_offset + parameter_index] += gradient;
}

__global__ void true2d_train_expand_gradients_kernel(
    const float* input_rows,
    const size_t* model_indices,
    const float* matrices,
    const float* grad_matrices,
    float* gradients,
    size_t sample_count,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DShape shape) {
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t expand_weight_count = true2d_matrix_layer_weights_offset(shape.row_count);
  const size_t value_count = model_count * expand_weight_count;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const size_t model_index = index / expand_weight_count;
  const size_t weight_index = index % expand_weight_count;
  const size_t source_embeddings_offset = true2d_source_row_embeddings_offset();
  const size_t target_embeddings_offset = true2d_target_row_embeddings_offset(shape.row_count);
  const size_t pair_embeddings_offset = true2d_pair_embeddings_offset(shape.row_count);
  const size_t expand_bias_offset = true2d_expand_bias_offset(shape.row_count);
  float gradient = 0.0f;

  for (size_t sample = 0; sample < sample_count; ++sample) {
    if (model_indices[sample] != model_index) {
      continue;
    }
    const size_t matrix_offset =
        true2d_matrix_history_offset(sample, 0, cell_count, shape.layer_count);
    if (weight_index < TRUE2D_INPUT_PAIR_FEATURE_WEIGHTS) {
      for (size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
        const size_t source_row = cell_index / shape.row_count;
        const size_t target_row = cell_index % shape.row_count;
        const size_t source_input =
            (sample * shape.row_count + source_row) * TRUE2D_INPUT_FEATURE_COUNT;
        const size_t target_input =
            (sample * shape.row_count + target_row) * TRUE2D_INPUT_FEATURE_COUNT;
        const float matrix_value = matrices[matrix_offset + cell_index];
        const float d_pre_activation =
            grad_matrices[matrix_offset + cell_index] * (1.0f - matrix_value * matrix_value);
        if (weight_index == TRUE2D_SOURCE_OWNER_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[source_input];
        } else if (weight_index == TRUE2D_SOURCE_SHIP_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[source_input + 1];
        } else if (weight_index == TRUE2D_SOURCE_X_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[source_input + 2];
        } else if (weight_index == TRUE2D_SOURCE_Y_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[source_input + 3];
        } else if (weight_index == TRUE2D_SOURCE_PRODUCTION_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[source_input + 4];
        } else if (weight_index == TRUE2D_SOURCE_VELOCITY_X_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[source_input + 5];
        } else if (weight_index == TRUE2D_SOURCE_VELOCITY_Y_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[source_input + 6];
        } else if (weight_index == TRUE2D_TARGET_OWNER_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[target_input];
        } else if (weight_index == TRUE2D_TARGET_SHIP_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[target_input + 1];
        } else if (weight_index == TRUE2D_TARGET_X_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[target_input + 2];
        } else if (weight_index == TRUE2D_TARGET_Y_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[target_input + 3];
        } else if (weight_index == TRUE2D_TARGET_PRODUCTION_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[target_input + 4];
        } else if (weight_index == TRUE2D_TARGET_VELOCITY_X_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[target_input + 5];
        } else if (weight_index == TRUE2D_TARGET_VELOCITY_Y_WEIGHT_INDEX) {
          gradient += d_pre_activation * input_rows[target_input + 6];
        }
      }
    } else if (
        weight_index >= source_embeddings_offset && weight_index < target_embeddings_offset) {
      const size_t source_row = weight_index - source_embeddings_offset;
      for (size_t target_row = 0; target_row < shape.row_count; ++target_row) {
        const size_t cell_index = source_row * shape.row_count + target_row;
        const float matrix_value = matrices[matrix_offset + cell_index];
        gradient +=
            grad_matrices[matrix_offset + cell_index] * (1.0f - matrix_value * matrix_value);
      }
    } else if (weight_index >= target_embeddings_offset && weight_index < pair_embeddings_offset) {
      const size_t target_row = weight_index - target_embeddings_offset;
      for (size_t source_row = 0; source_row < shape.row_count; ++source_row) {
        const size_t cell_index = source_row * shape.row_count + target_row;
        const float matrix_value = matrices[matrix_offset + cell_index];
        gradient +=
            grad_matrices[matrix_offset + cell_index] * (1.0f - matrix_value * matrix_value);
      }
    } else if (weight_index >= pair_embeddings_offset && weight_index < expand_bias_offset) {
      const size_t cell_index = weight_index - pair_embeddings_offset;
      const float matrix_value = matrices[matrix_offset + cell_index];
      gradient +=
          grad_matrices[matrix_offset + cell_index] * (1.0f - matrix_value * matrix_value);
    } else if (weight_index == expand_bias_offset) {
      for (size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
        const float matrix_value = matrices[matrix_offset + cell_index];
        gradient +=
            grad_matrices[matrix_offset + cell_index] * (1.0f - matrix_value * matrix_value);
      }
    }
  }

  gradients[model_index * weight_count + weight_index] += gradient;
}

__global__ void true2d_apply_gradients_kernel(
    float* all_weights,
    float* gradients,
    const size_t* model_sample_counts,
    size_t model_count,
    size_t weight_count,
    OrbitWarsCudaTrue2DTrainingConfig training_config) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t value_count = model_count * weight_count;
  if (index >= value_count) {
    return;
  }
  const size_t model_index = index / weight_count;
  const size_t sample_count = model_sample_counts[model_index];
  if (sample_count == 0) {
    return;
  }
  const float delta =
      training_config.learning_rate * gradients[index] / static_cast<float>(sample_count);
  all_weights[index] -= delta;
}

}  // namespace

extern "C" OrbitWarsCudaStatus orbit_wars_cuda_status(void) {
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  if (device_count <= 0) {
    return {CUDA_STATUS_RUNTIME_ERROR, "no cuda devices"};
  }
  return {CUDA_STATUS_OK, "cuda device available"};
}

extern "C" OrbitWarsCudaStatus orbit_wars_cuda_encode_identity_batch(
    const float* input_rows,
    float* output_rows,
    size_t game_count,
    size_t row_count,
    size_t feature_count) {
  if (input_rows == nullptr || output_rows == nullptr || game_count == 0 || row_count == 0 ||
      feature_count == 0) {
    return {CUDA_STATUS_BAD_ARGUMENT, "invalid batch buffer"};
  }

  const size_t value_count = game_count * row_count * feature_count;
  const int block_count =
      static_cast<int>((value_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  copy_features_kernel<<<block_count, THREADS_PER_BLOCK>>>(input_rows, output_rows, value_count);
  return status_from_cuda(cudaDeviceSynchronize());
}

extern "C" OrbitWarsCudaStatus orbit_wars_cuda_trainable_forward(
    const float* input_rows,
    const float* weights,
    float* output_rows,
    size_t game_count,
    OrbitWarsCudaTransformerShape shape) {
  if (input_rows == nullptr || weights == nullptr || output_rows == nullptr || game_count == 0 ||
      !shape_is_valid(shape)) {
    return {CUDA_STATUS_BAD_ARGUMENT, "invalid transformer batch"};
  }

  const size_t input_count = game_count * shape.row_count * CUDA_INPUT_FEATURE_COUNT;
  const size_t hidden_count = game_count * shape.row_count * shape.d_model;
  const size_t output_count = game_count * shape.row_count * TRUE2D_OUTPUT_FEATURE_COUNT;
  const size_t weight_count = transformer_weight_count(shape);
  const size_t input_bytes = input_count * sizeof(float);
  const size_t hidden_bytes = hidden_count * sizeof(float);
  const size_t output_bytes = output_count * sizeof(float);
  const size_t weight_bytes = weight_count * sizeof(float);

  float* device_input = nullptr;
  float* device_weights = nullptr;
  float* device_hidden = nullptr;
  float* device_next = nullptr;
  float* device_output = nullptr;

  cudaError_t error = cudaMalloc(&device_input, input_bytes);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  error = cudaMalloc(&device_weights, weight_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_input);
    return status_from_cuda(error);
  }
  error = cudaMalloc(&device_hidden, hidden_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_weights);
    cudaFree(device_input);
    return status_from_cuda(error);
  }
  error = cudaMalloc(&device_next, hidden_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_hidden);
    cudaFree(device_weights);
    cudaFree(device_input);
    return status_from_cuda(error);
  }
  error = cudaMalloc(&device_output, output_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_next);
    cudaFree(device_hidden);
    cudaFree(device_weights);
    cudaFree(device_input);
    return status_from_cuda(error);
  }

  error = cudaMemcpy(device_input, input_rows, input_bytes, cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_weights, weights, weight_bytes, cudaMemcpyHostToDevice);
  }

  const int hidden_blocks =
      static_cast<int>((hidden_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  const int output_blocks =
      static_cast<int>((game_count * shape.row_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

  if (error == cudaSuccess) {
    input_projection_kernel<<<hidden_blocks, THREADS_PER_BLOCK>>>(
        device_input, device_weights, device_hidden, game_count, shape.row_count, shape.d_model);
    error = cudaGetLastError();
  }
  for (size_t layer_index = 0; error == cudaSuccess && layer_index < shape.layer_count;
       ++layer_index) {
    attention_kernel<<<hidden_blocks, THREADS_PER_BLOCK>>>(
        device_hidden, device_next, game_count, shape.row_count, shape.d_model, shape.head_count);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
      layer_apply_kernel<<<hidden_blocks, THREADS_PER_BLOCK>>>(
          device_hidden,
          device_next,
          device_weights,
          game_count,
          shape.row_count,
          shape.d_model,
          shape.layer_count,
          layer_index);
      error = cudaGetLastError();
    }
  }
  if (error == cudaSuccess) {
    output_projection_kernel<<<output_blocks, THREADS_PER_BLOCK>>>(
        device_hidden,
        device_weights,
        device_output,
        game_count,
        shape.row_count,
        shape.d_model,
        shape.layer_count);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(output_rows, device_output, output_bytes, cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }

  cudaFree(device_output);
  cudaFree(device_next);
  cudaFree(device_hidden);
  cudaFree(device_weights);
  cudaFree(device_input);
  return status_from_cuda(error);
}

OrbitWarsCudaStatus run_true2d_many_workspace_forward(
    const float* input_rows,
    const size_t* model_indices,
    float* output_rows,
    size_t request_count,
    size_t model_count,
    OrbitWarsCudaTrue2DShape shape) {
  const size_t input_count = request_count * shape.row_count * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t matrix_count = request_count * cell_count;
  const size_t attention_projection_count = matrix_count * shape.head_count;
  const size_t output_count = request_count * shape.row_count * TRUE2D_OUTPUT_FEATURE_COUNT;
  const size_t weight_count = true2d_weight_count(shape);
  const size_t input_bytes = input_count * sizeof(float);
  const size_t model_index_bytes = request_count * sizeof(size_t);
  const size_t output_bytes = output_count * sizeof(float);

  cudaError_t error = cudaMemcpy(
      true2d_many_workspace.device_input, input_rows, input_bytes, cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        true2d_many_workspace.device_model_indices,
        model_indices,
        model_index_bytes,
        cudaMemcpyHostToDevice);
  }

  float* device_matrix = true2d_many_workspace.device_matrix;
  float* device_next_matrix = true2d_many_workspace.device_next_matrix;
  const int matrix_blocks =
      static_cast<int>((matrix_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  const int attention_warp_blocks =
      static_cast<int>((matrix_count + TRUE2D_ATTENTION_WARPS_PER_BLOCK - 1) /
                       TRUE2D_ATTENTION_WARPS_PER_BLOCK);
  const int attention_projection_blocks = static_cast<int>(
      (attention_projection_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  const int context_blocks =
      static_cast<int>((output_count / TRUE2D_OUTPUT_FEATURE_COUNT + THREADS_PER_BLOCK - 1) /
                       THREADS_PER_BLOCK);

  if (error == cudaSuccess) {
    true2d_expand_many_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
        true2d_many_workspace.device_input,
        true2d_many_workspace.device_model_indices,
        true2d_many_workspace.device_weights,
        device_matrix,
        request_count,
        model_count,
        weight_count,
        shape.row_count);
    error = cudaGetLastError();
  }

  const size_t attention_index = true2d_attention_layer_index(shape.layer_count);
  for (size_t layer_index = 0; error == cudaSuccess && layer_index < attention_index;
       ++layer_index) {
    true2d_matrix_layer_many_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
        device_matrix,
        device_next_matrix,
        true2d_many_workspace.device_model_indices,
        true2d_many_workspace.device_weights,
        request_count,
        model_count,
        weight_count,
        shape,
        layer_index);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
      float* old_matrix = device_matrix;
      device_matrix = device_next_matrix;
      device_next_matrix = old_matrix;
    }
  }
  if (error == cudaSuccess) {
    true2d_attention_qkv_many_kernel<<<attention_projection_blocks, THREADS_PER_BLOCK>>>(
        device_matrix,
        true2d_many_workspace.device_model_indices,
        true2d_many_workspace.device_weights,
        true2d_many_workspace.device_attention_queries,
        true2d_many_workspace.device_attention_keys,
        true2d_many_workspace.device_attention_values,
        request_count,
        model_count,
        weight_count,
        shape);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    true2d_full_attention_many_kernel<<<attention_warp_blocks, THREADS_PER_BLOCK>>>(
        device_matrix,
        true2d_many_workspace.device_model_indices,
        true2d_many_workspace.device_weights,
        true2d_many_workspace.device_attention_queries,
        true2d_many_workspace.device_attention_keys,
        true2d_many_workspace.device_attention_values,
        device_next_matrix,
        request_count,
        model_count,
        weight_count,
        shape,
        attention_index);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    float* old_matrix = device_matrix;
    device_matrix = device_next_matrix;
    device_next_matrix = old_matrix;
  }
  for (size_t layer_index = attention_index; error == cudaSuccess && layer_index < shape.layer_count;
       ++layer_index) {
    true2d_matrix_layer_many_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
        device_matrix,
        device_next_matrix,
        true2d_many_workspace.device_model_indices,
        true2d_many_workspace.device_weights,
        request_count,
        model_count,
        weight_count,
        shape,
        layer_index);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
      float* old_matrix = device_matrix;
      device_matrix = device_next_matrix;
      device_next_matrix = old_matrix;
    }
  }

  if (error == cudaSuccess) {
    true2d_output_many_kernel<<<context_blocks, THREADS_PER_BLOCK>>>(
        device_matrix,
        true2d_many_workspace.device_model_indices,
        true2d_many_workspace.device_weights,
        true2d_many_workspace.device_output,
        request_count,
        model_count,
        weight_count,
        shape);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        output_rows,
        true2d_many_workspace.device_output,
        output_bytes,
        cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }

  return status_from_cuda(error);
}

extern "C" OrbitWarsCudaStatus orbit_wars_cuda_true2d_forward_many(
    const float* input_rows,
    const size_t* model_indices,
    const float* all_weights,
    float* output_rows,
    size_t request_count,
    size_t model_count,
    OrbitWarsCudaTrue2DShape shape) {
  if (!true2d_many_batch_is_valid(
          input_rows, model_indices, all_weights, output_rows, request_count, model_count, shape)) {
    return {CUDA_STATUS_BAD_ARGUMENT, "invalid true2d many batch"};
  }

  OrbitWarsCudaStatus status = ensure_true2d_many_workspace(request_count, model_count, shape);
  if (status.code != CUDA_STATUS_OK) {
    return status;
  }

  const size_t weight_count = true2d_weight_count(shape);
  const size_t all_weight_count = model_count * weight_count;
  const size_t weight_bytes = all_weight_count * sizeof(float);
  cudaError_t error = cudaMemcpy(
      true2d_many_workspace.device_weights, all_weights, weight_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    true2d_many_workspace.resident_weights_loaded = false;
    return status_from_cuda(error);
  }
  true2d_many_workspace.resident_model_count = model_count;
  true2d_many_workspace.resident_weight_count = weight_count;
  true2d_many_workspace.resident_weights_loaded = true;
  return run_true2d_many_workspace_forward(
      input_rows, model_indices, output_rows, request_count, model_count, shape);
}

extern "C" OrbitWarsCudaStatus orbit_wars_cuda_true2d_upload_population(
    const float* all_weights,
    size_t model_count,
    size_t request_capacity,
    OrbitWarsCudaTrue2DShape shape) {
  if (all_weights == nullptr || model_count == 0 || request_capacity == 0 ||
      !true2d_shape_is_valid(shape)) {
    return {CUDA_STATUS_BAD_ARGUMENT, "invalid true2d population upload"};
  }

  OrbitWarsCudaStatus status = ensure_true2d_many_workspace(request_capacity, model_count, shape);
  if (status.code != CUDA_STATUS_OK) {
    return status;
  }

  const size_t weight_count = true2d_weight_count(shape);
  const size_t all_weight_count = model_count * weight_count;
  const size_t weight_bytes = all_weight_count * sizeof(float);
  const cudaError_t error = cudaMemcpy(
      true2d_many_workspace.device_weights, all_weights, weight_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    true2d_many_workspace.resident_weights_loaded = false;
    return status_from_cuda(error);
  }
  true2d_many_workspace.resident_model_count = model_count;
  true2d_many_workspace.resident_weight_count = weight_count;
  true2d_many_workspace.resident_weights_loaded = true;
  return {CUDA_STATUS_OK, "ok"};
}

extern "C" OrbitWarsCudaStatus orbit_wars_cuda_true2d_forward_many_resident(
    const float* input_rows,
    const size_t* model_indices,
    float* output_rows,
    size_t request_count,
    OrbitWarsCudaTrue2DShape shape) {
  if (!true2d_resident_population_matches(shape)) {
    return {CUDA_STATUS_BAD_ARGUMENT, "true2d resident population not loaded"};
  }
  if (request_count > true2d_many_workspace.request_capacity) {
    return {CUDA_STATUS_BAD_ARGUMENT, "true2d resident request capacity exceeded"};
  }
  const size_t model_count = true2d_many_workspace.resident_model_count;
  if (!true2d_many_request_is_valid(
          input_rows, model_indices, output_rows, request_count, model_count, shape)) {
    return {CUDA_STATUS_BAD_ARGUMENT, "invalid true2d resident batch"};
  }
  return run_true2d_many_workspace_forward(
      input_rows, model_indices, output_rows, request_count, model_count, shape);
}

extern "C" OrbitWarsCudaStatus orbit_wars_cuda_true2d_train_full_attention(
    const float* input_rows,
    const float* output_rows,
    const int* rewards,
    const float* target_action_rewards,
    const size_t* model_indices,
    float* all_weights,
    size_t sample_count,
    size_t model_count,
    OrbitWarsCudaTrue2DShape shape,
    OrbitWarsCudaTrue2DTrainingConfig training_config) {
  if (input_rows == nullptr || output_rows == nullptr || rewards == nullptr ||
      target_action_rewards == nullptr || model_indices == nullptr || all_weights == nullptr ||
      sample_count == 0 || model_count == 0 || !true2d_shape_is_valid(shape) ||
      !true2d_training_config_is_valid(training_config)) {
    return {CUDA_STATUS_BAD_ARGUMENT, "invalid true2d training batch"};
  }

  const size_t weight_count = true2d_weight_count(shape);
  const size_t all_weight_count = model_count * weight_count;
  std::vector<size_t> model_sample_counts(model_count, 0);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const size_t model_index = model_indices[sample_index];
    if (model_index >= model_count) {
      return {CUDA_STATUS_BAD_ARGUMENT, "true2d training model index out of range"};
    }
    ++model_sample_counts[model_index];
  }

  const size_t batch_capacity = std::min(training_config.batch_sample_count, sample_count);
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t input_batch_count =
      batch_capacity * shape.row_count * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t output_batch_count = batch_capacity * shape.row_count * TRUE2D_OUTPUT_FEATURE_COUNT;
  const size_t target_action_reward_batch_count =
      batch_capacity * shape.row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE;
  const size_t matrix_history_count = batch_capacity * (shape.layer_count + 2) * cell_count;
  const size_t attention_history_count = batch_capacity * cell_count;
  const size_t attention_head_history_count =
      batch_capacity * shape.head_count * cell_count;
  const size_t attention_head_cell_gradient_count =
      attention_head_history_count * TRUE2D_ATTENTION_HEAD_WEIGHTS;

  float* device_input = nullptr;
  float* device_output = nullptr;
  int* device_rewards = nullptr;
  float* device_target_action_rewards = nullptr;
  size_t* device_model_indices = nullptr;
  size_t* device_model_sample_counts = nullptr;
  float* device_weights = nullptr;
  float* device_gradients = nullptr;
  float* device_matrices = nullptr;
  float* device_attention_values = nullptr;
  float* device_attention_max_values = nullptr;
  float* device_attention_denominators = nullptr;
  float* device_attention_weighted_averages = nullptr;
  float* device_attention_head_cell_gradients = nullptr;
  float* device_grad_matrices = nullptr;
  float* device_grad_attention_values = nullptr;

  cudaError_t error = cudaSuccess;
  auto release_float = [&error](float*& pointer) {
    if (pointer == nullptr) {
      return;
    }
    const cudaError_t release_error = cudaFree(pointer);
    if (error == cudaSuccess && release_error != cudaSuccess) {
      error = release_error;
    }
    pointer = nullptr;
  };
  auto release_int = [&error](int*& pointer) {
    if (pointer == nullptr) {
      return;
    }
    const cudaError_t release_error = cudaFree(pointer);
    if (error == cudaSuccess && release_error != cudaSuccess) {
      error = release_error;
    }
    pointer = nullptr;
  };
  auto release_size = [&error](size_t*& pointer) {
    if (pointer == nullptr) {
      return;
    }
    const cudaError_t release_error = cudaFree(pointer);
    if (error == cudaSuccess && release_error != cudaSuccess) {
      error = release_error;
    }
    pointer = nullptr;
  };

  if (error == cudaSuccess) {
    error = cudaMalloc(&device_input, input_batch_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output, output_batch_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_rewards, batch_capacity * sizeof(int));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &device_target_action_rewards, target_action_reward_batch_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_model_indices, batch_capacity * sizeof(size_t));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_model_sample_counts, model_count * sizeof(size_t));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_weights, all_weight_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_gradients, all_weight_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_matrices, matrix_history_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_attention_values, attention_history_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_attention_max_values, attention_head_history_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error =
        cudaMalloc(&device_attention_denominators, attention_head_history_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &device_attention_weighted_averages, attention_head_history_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &device_attention_head_cell_gradients,
        attention_head_cell_gradient_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_grad_matrices, matrix_history_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_grad_attention_values, attention_history_count * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        device_weights, all_weights, all_weight_count * sizeof(float), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        device_model_sample_counts,
        model_sample_counts.data(),
        model_count * sizeof(size_t),
        cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemset(device_gradients, 0, all_weight_count * sizeof(float));
  }

  for (size_t sample_offset = 0; error == cudaSuccess && sample_offset < sample_count;
       sample_offset += batch_capacity) {
    const size_t chunk_sample_count = std::min(batch_capacity, sample_count - sample_offset);
    const size_t chunk_input_count =
        chunk_sample_count * shape.row_count * TRUE2D_INPUT_FEATURE_COUNT;
    const size_t chunk_output_count =
        chunk_sample_count * shape.row_count * TRUE2D_OUTPUT_FEATURE_COUNT;
    const size_t chunk_target_action_reward_count =
        chunk_sample_count * shape.row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE;
    const size_t chunk_matrix_history_count =
        chunk_sample_count * (shape.layer_count + 2) * cell_count;
    const size_t chunk_attention_history_count = chunk_sample_count * cell_count;
    const int matrix_blocks =
        static_cast<int>((chunk_sample_count * cell_count + THREADS_PER_BLOCK - 1) /
                         THREADS_PER_BLOCK);
    const int output_blocks =
        static_cast<int>((chunk_sample_count * shape.row_count + THREADS_PER_BLOCK - 1) /
                         THREADS_PER_BLOCK);
    const int output_gradient_blocks = static_cast<int>(
        (model_count *
             (shape.row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE +
              shape.row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE +
              TRUE2D_OUTPUT_SCALAR_WEIGHTS * TRUE2D_ACTION_TARGETS_PER_SOURCE) +
         THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK);
    const int layer_shared_gradient_blocks = static_cast<int>(
        (model_count * TRUE2D_LAYER_SHARED_WEIGHTS + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK);
    const int attention_head_gradient_blocks = static_cast<int>(
        (model_count * shape.head_count * TRUE2D_ATTENTION_HEAD_WEIGHTS + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK);
    const int matrix_layer_gradient_blocks = static_cast<int>(
        (model_count * shape.layer_count * true2d_matrix_layer_weight_count(shape.row_count) +
         THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK);
    const int expand_gradient_blocks = static_cast<int>(
        (model_count * true2d_matrix_layer_weights_offset(shape.row_count) +
         THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK);

    error = cudaMemcpy(
        device_input,
        input_rows + sample_offset * shape.row_count * TRUE2D_INPUT_FEATURE_COUNT,
        chunk_input_count * sizeof(float),
        cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(
          device_output,
          output_rows + sample_offset * shape.row_count * TRUE2D_OUTPUT_FEATURE_COUNT,
          chunk_output_count * sizeof(float),
          cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(
          device_rewards, rewards + sample_offset, chunk_sample_count * sizeof(int),
          cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(
          device_target_action_rewards,
          target_action_rewards +
              sample_offset * shape.row_count * TRUE2D_ACTION_TARGETS_PER_SOURCE,
          chunk_target_action_reward_count * sizeof(float),
          cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(
          device_model_indices,
          model_indices + sample_offset,
          chunk_sample_count * sizeof(size_t),
          cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemset(device_grad_matrices, 0, chunk_matrix_history_count * sizeof(float));
    }
    if (error == cudaSuccess && chunk_attention_history_count > 0) {
      error =
          cudaMemset(device_grad_attention_values, 0, chunk_attention_history_count * sizeof(float));
    }

    if (error == cudaSuccess) {
      true2d_train_expand_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
          device_input,
          device_model_indices,
          device_weights,
          device_matrices,
          chunk_sample_count,
          model_count,
          weight_count,
          shape);
      error = cudaGetLastError();
    }
    const size_t attention_index = true2d_attention_layer_index(shape.layer_count);
    for (size_t layer_index = 0; error == cudaSuccess && layer_index < attention_index;
         ++layer_index) {
      true2d_train_matrix_layer_forward_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
          device_matrices,
          device_model_indices,
          device_weights,
          device_matrices,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          layer_index);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_attention_forward_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
          device_matrices,
          device_model_indices,
          device_weights,
          device_attention_values,
          device_attention_max_values,
          device_attention_denominators,
          device_attention_weighted_averages,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          attention_index);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_layer_apply_forward_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
          device_matrices,
          device_attention_values,
          device_model_indices,
          device_weights,
          device_matrices,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          attention_index);
      error = cudaGetLastError();
    }
    for (size_t layer_index = attention_index; error == cudaSuccess && layer_index < shape.layer_count;
         ++layer_index) {
      true2d_train_matrix_layer_forward_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
          device_matrices,
          device_model_indices,
          device_weights,
          device_matrices,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          layer_index);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_output_loss_backward_kernel<<<output_blocks, THREADS_PER_BLOCK>>>(
          device_input,
          device_output,
          device_rewards,
          device_target_action_rewards,
          device_model_indices,
          device_matrices,
          device_weights,
          device_grad_matrices,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          training_config);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_output_gradients_kernel<<<output_gradient_blocks, THREADS_PER_BLOCK>>>(
          device_input,
          device_output,
          device_rewards,
          device_target_action_rewards,
          device_model_indices,
          device_matrices,
          device_weights,
          device_gradients,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          training_config);
      error = cudaGetLastError();
    }
    for (size_t reverse_layer = shape.layer_count;
         error == cudaSuccess && reverse_layer > attention_index;
         --reverse_layer) {
      const size_t layer_index = reverse_layer - 1;
      true2d_train_matrix_layer_backward_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
          device_matrices,
          device_model_indices,
          device_weights,
          device_grad_matrices,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          layer_index);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_layer_apply_backward_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
          device_matrices,
          device_attention_values,
          device_model_indices,
          device_weights,
          device_grad_matrices,
          device_grad_attention_values,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          attention_index);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_layer_shared_gradients_kernel<<<
          layer_shared_gradient_blocks,
          THREADS_PER_BLOCK>>>(
          device_matrices,
          device_attention_values,
          device_grad_matrices,
          device_model_indices,
          device_weights,
          device_gradients,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          attention_index);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_attention_backward_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
          device_matrices,
          device_grad_attention_values,
          device_attention_max_values,
          device_attention_denominators,
          device_attention_weighted_averages,
          device_model_indices,
          device_weights,
          device_attention_head_cell_gradients,
          device_grad_matrices,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          attention_index);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_attention_head_gradients_kernel<<<
          attention_head_gradient_blocks,
          THREADS_PER_BLOCK>>>(
          device_attention_head_cell_gradients,
          device_model_indices,
          device_gradients,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          attention_index);
      error = cudaGetLastError();
    }
    for (size_t reverse_layer = attention_index; error == cudaSuccess && reverse_layer > 0;
         --reverse_layer) {
      const size_t layer_index = reverse_layer - 1;
      true2d_train_matrix_layer_backward_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
          device_matrices,
          device_model_indices,
          device_weights,
          device_grad_matrices,
          chunk_sample_count,
          model_count,
          weight_count,
          shape,
          layer_index);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_matrix_layer_gradients_kernel<<<
          matrix_layer_gradient_blocks,
          THREADS_PER_BLOCK>>>(
          device_matrices,
          device_grad_matrices,
          device_model_indices,
          device_weights,
          device_gradients,
          chunk_sample_count,
          model_count,
          weight_count,
          shape);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      true2d_train_expand_gradients_kernel<<<expand_gradient_blocks, THREADS_PER_BLOCK>>>(
          device_input,
          device_model_indices,
          device_matrices,
          device_grad_matrices,
          device_gradients,
          chunk_sample_count,
          model_count,
          weight_count,
          shape);
      error = cudaGetLastError();
    }
  }

  if (error == cudaSuccess) {
    const int weight_blocks =
        static_cast<int>((all_weight_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    true2d_apply_gradients_kernel<<<weight_blocks, THREADS_PER_BLOCK>>>(
        device_weights,
        device_gradients,
        device_model_sample_counts,
        model_count,
        weight_count,
        training_config);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    error =
        cudaMemcpy(all_weights, device_weights, all_weight_count * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }

  release_float(device_grad_attention_values);
  release_float(device_grad_matrices);
  release_float(device_attention_head_cell_gradients);
  release_float(device_attention_weighted_averages);
  release_float(device_attention_denominators);
  release_float(device_attention_max_values);
  release_float(device_attention_values);
  release_float(device_matrices);
  release_float(device_gradients);
  release_float(device_weights);
  release_float(device_target_action_rewards);
  release_size(device_model_sample_counts);
  release_size(device_model_indices);
  release_int(device_rewards);
  release_float(device_output);
  release_float(device_input);
  return status_from_cuda(error);
}

extern "C" OrbitWarsCudaStatus orbit_wars_cuda_true2d_forward(
    const float* input_rows,
    const float* weights,
    float* output_rows,
    size_t game_count,
    OrbitWarsCudaTrue2DShape shape) {
  if (input_rows == nullptr || weights == nullptr || output_rows == nullptr || game_count == 0 ||
      !true2d_shape_is_valid(shape)) {
    return {CUDA_STATUS_BAD_ARGUMENT, "invalid true2d batch"};
  }

  const size_t input_count = game_count * shape.row_count * TRUE2D_INPUT_FEATURE_COUNT;
  const size_t cell_count = true2d_cell_count(shape.row_count);
  const size_t matrix_count = game_count * cell_count;
  const size_t output_count = game_count * shape.row_count * TRUE2D_OUTPUT_FEATURE_COUNT;
  const size_t weight_count = true2d_weight_count(shape);
  const size_t input_bytes = input_count * sizeof(float);
  const size_t matrix_bytes = matrix_count * sizeof(float);
  const size_t output_bytes = output_count * sizeof(float);
  const size_t weight_bytes = weight_count * sizeof(float);

  float* device_input = nullptr;
  float* device_weights = nullptr;
  float* device_matrix = nullptr;
  float* device_next_matrix = nullptr;
  float* device_attention_queries = nullptr;
  float* device_attention_keys = nullptr;
  float* device_attention_values = nullptr;
  float* device_output = nullptr;

  cudaError_t error = cudaMalloc(&device_input, input_bytes);
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_weights, weight_bytes);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_matrix, matrix_bytes);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_next_matrix, matrix_bytes);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_attention_queries, matrix_bytes * shape.head_count);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_attention_keys, matrix_bytes * shape.head_count);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_attention_values, matrix_bytes * shape.head_count);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output, output_bytes);
  }

  if (error == cudaSuccess) {
    error = cudaMemcpy(device_input, input_rows, input_bytes, cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_weights, weights, weight_bytes, cudaMemcpyHostToDevice);
  }

  const int matrix_blocks =
      static_cast<int>((matrix_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  const int attention_warp_blocks =
      static_cast<int>((matrix_count + TRUE2D_ATTENTION_WARPS_PER_BLOCK - 1) /
                       TRUE2D_ATTENTION_WARPS_PER_BLOCK);
  const int attention_projection_blocks = static_cast<int>(
      (matrix_count * shape.head_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  const int context_blocks =
      static_cast<int>((game_count * shape.row_count + THREADS_PER_BLOCK - 1) /
                       THREADS_PER_BLOCK);

  if (error == cudaSuccess) {
    true2d_expand_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
        device_input, device_weights, device_matrix, game_count, shape.row_count);
    error = cudaGetLastError();
  }

  const size_t attention_index = true2d_attention_layer_index(shape.layer_count);
  for (size_t layer_index = 0; error == cudaSuccess && layer_index < attention_index;
       ++layer_index) {
    true2d_matrix_layer_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
        device_matrix, device_next_matrix, device_weights, game_count, shape, layer_index);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
      float* old_matrix = device_matrix;
      device_matrix = device_next_matrix;
      device_next_matrix = old_matrix;
    }
  }
  if (error == cudaSuccess) {
    true2d_attention_qkv_kernel<<<attention_projection_blocks, THREADS_PER_BLOCK>>>(
        device_matrix,
        device_weights,
        device_attention_queries,
        device_attention_keys,
        device_attention_values,
        game_count,
        shape);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    true2d_full_attention_kernel<<<attention_warp_blocks, THREADS_PER_BLOCK>>>(
        device_matrix,
        device_weights,
        device_attention_queries,
        device_attention_keys,
        device_attention_values,
        device_next_matrix,
        game_count,
        shape,
        attention_index);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    float* old_matrix = device_matrix;
    device_matrix = device_next_matrix;
    device_next_matrix = old_matrix;
  }
  for (size_t layer_index = attention_index; error == cudaSuccess && layer_index < shape.layer_count;
       ++layer_index) {
    true2d_matrix_layer_kernel<<<matrix_blocks, THREADS_PER_BLOCK>>>(
        device_matrix, device_next_matrix, device_weights, game_count, shape, layer_index);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
      float* old_matrix = device_matrix;
      device_matrix = device_next_matrix;
      device_next_matrix = old_matrix;
    }
  }

  if (error == cudaSuccess) {
    true2d_output_kernel<<<context_blocks, THREADS_PER_BLOCK>>>(
        device_matrix, device_weights, device_output, game_count, shape);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(output_rows, device_output, output_bytes, cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }

  cudaFree(device_output);
  cudaFree(device_attention_values);
  cudaFree(device_attention_keys);
  cudaFree(device_attention_queries);
  cudaFree(device_next_matrix);
  cudaFree(device_matrix);
  cudaFree(device_weights);
  cudaFree(device_input);
  return status_from_cuda(error);
}
