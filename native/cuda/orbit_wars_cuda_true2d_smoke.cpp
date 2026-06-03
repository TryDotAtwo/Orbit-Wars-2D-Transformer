#include "orbit_wars_cuda.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr size_t kGameCount = 2;
constexpr size_t kModelCount = 2;
constexpr size_t kRowCount = 64;
constexpr size_t kInputFeatureCount = 7;
constexpr size_t kActionTargetsPerSource = 12;
constexpr size_t kActionTargetFeatures = 2;
constexpr size_t kOutputFeatureCount = kActionTargetsPerSource * kActionTargetFeatures;
constexpr size_t kLayerCount = 61;
constexpr size_t kHeadCount = 1;
constexpr size_t kInputPairWeights = kInputFeatureCount * 2;
constexpr size_t kAttentionHeadWeights = 7;
constexpr size_t kAttentionSharedWeights = 6;
constexpr size_t kAttentionWeights = kHeadCount * kAttentionHeadWeights + kAttentionSharedWeights;
constexpr size_t kMatrixLayerCellWeights = 12;
constexpr size_t kOutputScalarWeights = 5;
constexpr float kMaxAbsDiff = 0.0005f;

size_t cell_count() {
  return kRowCount * kRowCount;
}

size_t source_embeddings_offset() {
  return kInputPairWeights;
}

size_t target_embeddings_offset() {
  return source_embeddings_offset() + kRowCount;
}

size_t pair_embeddings_offset() {
  return target_embeddings_offset() + kRowCount;
}

size_t expand_bias_offset() {
  return pair_embeddings_offset() + cell_count();
}

size_t matrix_layer_weights_offset() {
  return expand_bias_offset() + 1;
}

size_t matrix_layer_weights() {
  return cell_count() * kMatrixLayerCellWeights;
}

size_t attention_weights_offset() {
  return matrix_layer_weights_offset() + kLayerCount * matrix_layer_weights();
}

size_t target_output_bias_offset() {
  return attention_weights_offset() + kAttentionWeights;
}

size_t send_output_bias_offset() {
  return target_output_bias_offset() + kRowCount * kActionTargetsPerSource;
}

size_t output_scalar_weights_offset() {
  return send_output_bias_offset() + kRowCount * kActionTargetsPerSource;
}

size_t weight_count() {
  return output_scalar_weights_offset() + kOutputScalarWeights * kActionTargetsPerSource;
}

float sigmoid(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

float row_value(const std::vector<float>& matrix, size_t game, size_t source, size_t target) {
  return matrix[game * cell_count() + source * kRowCount + target];
}

float matrix_layer_value(float cell, const float* weights) {
  const float hidden_a = std::tanh(cell * weights[2] + weights[3]);
  const float hidden_b = std::tanh(cell * weights[4] + hidden_a * weights[5] + weights[6]);
  const float gate = sigmoid(cell * weights[7] + weights[8]);
  const float update = hidden_b * weights[9] + hidden_a * weights[10] + weights[11];
  const float residual = cell * (1.0f + weights[0]) + weights[1];
  return std::tanh(residual + gate * update);
}

std::vector<float> full_attention_values(
    const std::vector<float>& matrix,
    const float* layer) {
  std::vector<float> attention_values(kGameCount * cell_count());
  for (size_t game = 0; game < kGameCount; ++game) {
    const size_t game_offset = game * cell_count();
    for (size_t query_cell = 0; query_cell < cell_count(); ++query_cell) {
      const float query_cell_value = matrix[game_offset + query_cell];
      float combined_context = 0.0f;
      for (size_t head = 0; head < kHeadCount; ++head) {
        const float* head_weights = layer + head * kAttentionHeadWeights;
        const float query = query_cell_value * head_weights[0] + head_weights[1];
        float max_score = -INFINITY;
        for (size_t key_cell = 0; key_cell < cell_count(); ++key_cell) {
          const float key_cell_value = matrix[game_offset + key_cell];
          const float key = key_cell_value * head_weights[2] + head_weights[3];
          max_score = std::fmax(max_score, query * key);
        }
        float denominator = 0.0f;
        float weighted_value = 0.0f;
        for (size_t key_cell = 0; key_cell < cell_count(); ++key_cell) {
          const float key_cell_value = matrix[game_offset + key_cell];
          const float key = key_cell_value * head_weights[2] + head_weights[3];
          const float value = key_cell_value * head_weights[4] + head_weights[5];
          const float attention = std::exp(query * key - max_score);
          denominator += attention;
          weighted_value += attention * value;
        }
        combined_context += weighted_value / denominator * head_weights[6];
      }
      attention_values[game_offset + query_cell] = combined_context;
    }
  }
  return attention_values;
}

std::vector<float> cpu_reference(
    const std::vector<float>& input,
    const std::vector<float>& weights) {
  std::vector<float> matrix(kGameCount * cell_count());
  for (size_t game = 0; game < kGameCount; ++game) {
    for (size_t source = 0; source < kRowCount; ++source) {
      for (size_t target = 0; target < kRowCount; ++target) {
        const size_t cell = source * kRowCount + target;
        const size_t source_input = (game * kRowCount + source) * kInputFeatureCount;
        const size_t target_input = (game * kRowCount + target) * kInputFeatureCount;
        const float value =
            input[source_input] * weights[0] + input[source_input + 1] * weights[1] +
            input[source_input + 2] * weights[2] + input[source_input + 3] * weights[3] +
            input[source_input + 4] * weights[4] + input[source_input + 5] * weights[5] +
            input[source_input + 6] * weights[6] + input[target_input] * weights[7] +
            input[target_input + 1] * weights[8] + input[target_input + 2] * weights[9] +
            input[target_input + 3] * weights[10] + input[target_input + 4] * weights[11] +
            input[target_input + 5] * weights[12] + input[target_input + 6] * weights[13] +
            weights[source_embeddings_offset() + source] +
            weights[target_embeddings_offset() + target] + weights[pair_embeddings_offset() + cell] +
            weights[expand_bias_offset()];
        matrix[game * cell_count() + cell] = std::tanh(value);
      }
    }
  }

  const size_t attention_index = kLayerCount / 2;
  for (size_t layer_index = 0; layer_index < attention_index; ++layer_index) {
    const float* layer = weights.data() + matrix_layer_weights_offset() +
                         layer_index * matrix_layer_weights();
    std::vector<float> next(matrix.size());
    for (size_t game = 0; game < kGameCount; ++game) {
      for (size_t source = 0; source < kRowCount; ++source) {
        for (size_t target = 0; target < kRowCount; ++target) {
          const size_t index = game * cell_count() + source * kRowCount + target;
          const size_t cell = source * kRowCount + target;
          next[index] =
              matrix_layer_value(matrix[index], layer + cell * kMatrixLayerCellWeights);
        }
      }
    }
    matrix = next;
  }
  const float* attention_layer = weights.data() + attention_weights_offset();
  const std::vector<float> attention_values = full_attention_values(matrix, attention_layer);
  const float* shared = attention_layer + kHeadCount * kAttentionHeadWeights;
  std::vector<float> attention_next(matrix.size());
  for (size_t game = 0; game < kGameCount; ++game) {
    for (size_t source = 0; source < kRowCount; ++source) {
      for (size_t target = 0; target < kRowCount; ++target) {
        const size_t index = game * cell_count() + source * kRowCount + target;
        const float cell = matrix[index];
        const float residual_attention = cell * shared[0] + attention_values[index] + shared[1];
        const float hidden = std::tanh(residual_attention * shared[2] + shared[3]);
        attention_next[index] = std::tanh(residual_attention + hidden * shared[4] + shared[5]);
      }
    }
  }
  matrix = attention_next;
  for (size_t layer_index = attention_index; layer_index < kLayerCount; ++layer_index) {
    const float* layer = weights.data() + matrix_layer_weights_offset() +
                         layer_index * matrix_layer_weights();
    std::vector<float> next(matrix.size());
    for (size_t game = 0; game < kGameCount; ++game) {
      for (size_t source = 0; source < kRowCount; ++source) {
        for (size_t target = 0; target < kRowCount; ++target) {
          const size_t index = game * cell_count() + source * kRowCount + target;
          const size_t cell = source * kRowCount + target;
          next[index] =
              matrix_layer_value(matrix[index], layer + cell * kMatrixLayerCellWeights);
        }
      }
    }
    matrix = next;
  }

  std::vector<float> output(kGameCount * kRowCount * kOutputFeatureCount);
  for (size_t game = 0; game < kGameCount; ++game) {
    for (size_t source = 0; source < kRowCount; ++source) {
      const size_t output_index = (game * kRowCount + source) * kOutputFeatureCount;
      float send_fractions[kActionTargetsPerSource] = {};
      float send_fraction_sum = 0.0f;
      for (size_t action = 0; action < kActionTargetsPerSource; ++action) {
        const float* output_weights =
            weights.data() + output_scalar_weights_offset() + action * kOutputScalarWeights;
        const size_t bias_offset = action * kRowCount;
        float target_max = -INFINITY;
        for (size_t target = 0; target < kRowCount; ++target) {
          const float score = row_value(matrix, game, source, target) * output_weights[0] +
                              weights[target_output_bias_offset() + bias_offset + target];
          target_max = std::fmax(target_max, score);
        }
        float target_denominator = 0.0f;
        float target_fraction = 0.0f;
        for (size_t target = 0; target < kRowCount; ++target) {
          const float score = row_value(matrix, game, source, target) * output_weights[0] +
                              weights[target_output_bias_offset() + bias_offset + target];
          const float attention = std::exp(score - target_max);
          target_denominator += attention;
          target_fraction += attention * static_cast<float>(target) /
                             static_cast<float>(kRowCount - 1);
        }

        float send_max = -INFINITY;
        for (size_t target = 0; target < kRowCount; ++target) {
          const float score = row_value(matrix, game, source, target) * output_weights[1] +
                              weights[send_output_bias_offset() + bias_offset + target];
          send_max = std::fmax(send_max, score);
        }
        float send_denominator = 0.0f;
        float send_weighted = 0.0f;
        for (size_t target = 0; target < kRowCount; ++target) {
          const float value = row_value(matrix, game, source, target);
          const float score =
              value * output_weights[1] + weights[send_output_bias_offset() + bias_offset + target];
          const float attention = std::exp(score - send_max);
          send_denominator += attention;
          send_weighted += attention * value * output_weights[2];
        }
        const size_t action_output_index = output_index + action * kActionTargetFeatures;
        output[action_output_index] = target_fraction / target_denominator;
        send_fractions[action] =
            sigmoid((output_weights[4] + send_weighted / send_denominator) * output_weights[3]);
        send_fraction_sum += send_fractions[action];
      }
      const float send_normalizer = send_fraction_sum > 1.0f ? send_fraction_sum : 1.0f;
      for (size_t action = 0; action < kActionTargetsPerSource; ++action) {
        const size_t action_output_index = output_index + action * kActionTargetFeatures;
        output[action_output_index + 1] = send_fractions[action] / send_normalizer;
      }
    }
  }
  return output;
}

int require_ok(OrbitWarsCudaStatus status, const char* operation) {
  if (status.code == 0) {
    return 0;
  }
  std::cerr << "operation=" << operation << "; code=" << status.code
            << "; message=" << status.message << "\n";
  return status.code;
}

}  // namespace

int main() {
  int status_code = require_ok(orbit_wars_cuda_status(), "status");
  if (status_code != 0) {
    return status_code;
  }

  std::vector<float> input(kGameCount * kRowCount * kInputFeatureCount);
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = std::sin(static_cast<float>(index + 1) * 0.11f);
  }

  std::vector<float> weights(weight_count());
  for (size_t index = 0; index < weights.size(); ++index) {
    weights[index] = std::sin(static_cast<float>(index + 1) * 0.17f) * 0.02f;
  }
  std::vector<float> second_weights(weight_count());
  for (size_t index = 0; index < second_weights.size(); ++index) {
    second_weights[index] = std::sin(static_cast<float>(index + 1) * 0.23f) * 0.02f;
  }

  std::vector<float> gpu_output(kGameCount * kRowCount * kOutputFeatureCount);
  OrbitWarsCudaTrue2DShape shape{kLayerCount, kRowCount, kHeadCount};
  status_code = require_ok(
      orbit_wars_cuda_true2d_forward(
          input.data(), weights.data(), gpu_output.data(), kGameCount, shape),
      "true2d_forward");
  if (status_code != 0) {
    return status_code;
  }

  const std::vector<float> cpu_output = cpu_reference(input, weights);
  float max_abs_diff = 0.0f;
  for (size_t index = 0; index < gpu_output.size(); ++index) {
    if (!std::isfinite(gpu_output[index]) || gpu_output[index] < 0.0f || gpu_output[index] > 1.0f) {
      std::cerr << "operation=output_range_check; index=" << index << "; value=" << gpu_output[index]
                << "\n";
      return 3;
    }
    max_abs_diff = std::fmax(max_abs_diff, std::fabs(gpu_output[index] - cpu_output[index]));
  }
  if (max_abs_diff > kMaxAbsDiff) {
    std::cerr << "operation=cpu_gpu_compare; max_abs_diff=" << max_abs_diff << "\n";
    return 4;
  }

  std::vector<float> all_weights;
  all_weights.reserve(kModelCount * weight_count());
  all_weights.insert(all_weights.end(), weights.begin(), weights.end());
  all_weights.insert(all_weights.end(), second_weights.begin(), second_weights.end());
  std::vector<size_t> model_indices{kModelCount - 2, kModelCount - 1};
  std::vector<float> gpu_many_output(kGameCount * kRowCount * kOutputFeatureCount);
  status_code = require_ok(
      orbit_wars_cuda_true2d_forward_many(
          input.data(),
          model_indices.data(),
          all_weights.data(),
          gpu_many_output.data(),
          kGameCount,
          kModelCount,
          shape),
      "true2d_forward_many");
  if (status_code != 0) {
    return status_code;
  }
  const std::vector<float> first_reference = cpu_reference(input, weights);
  const std::vector<float> second_reference = cpu_reference(input, second_weights);
  float many_max_abs_diff = 0.0f;
  for (size_t index = 0; index < gpu_many_output.size(); ++index) {
    const size_t game = index / (kRowCount * kOutputFeatureCount);
    const float expected_value = game == 0 ? first_reference[index] : second_reference[index];
    if (!std::isfinite(gpu_many_output[index]) || gpu_many_output[index] < 0.0f ||
        gpu_many_output[index] > 1.0f) {
      std::cerr << "operation=many_output_range_check; index=" << index
                << "; value=" << gpu_many_output[index] << "\n";
      return 5;
    }
    many_max_abs_diff = std::fmax(
        many_max_abs_diff,
        std::fabs(gpu_many_output[index] - expected_value));
  }
  if (many_max_abs_diff > kMaxAbsDiff) {
    std::cerr << "operation=many_cpu_gpu_compare; max_abs_diff=" << many_max_abs_diff << "\n";
    return 6;
  }

  std::cout << "status=ok; cuda_true2d_forward_smoke=true; games=" << kGameCount
            << "; rows=" << kRowCount << "; max_abs_diff=" << max_abs_diff
            << "; many_max_abs_diff=" << many_max_abs_diff << "\n";
  return 0;
}
