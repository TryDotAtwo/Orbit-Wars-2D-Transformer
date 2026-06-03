#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OrbitWarsCudaStatus {
  int code;
  const char* message;
} OrbitWarsCudaStatus;

typedef struct OrbitWarsCudaTransformerShape {
  size_t layer_count;
  size_t d_model;
  size_t head_count;
  size_t row_count;
} OrbitWarsCudaTransformerShape;

typedef struct OrbitWarsCudaTrue2DShape {
  size_t layer_count;
  size_t row_count;
  size_t head_count;
} OrbitWarsCudaTrue2DShape;

typedef struct OrbitWarsCudaTrue2DTrainingConfig {
  size_t batch_sample_count;
  float learning_rate;
  float reward_normalizer;
  float send_threshold;
  float owner_class_own;
  float owner_class_empty;
  float owner_class_absent_on_map;
} OrbitWarsCudaTrue2DTrainingConfig;

OrbitWarsCudaStatus orbit_wars_cuda_status(void);

OrbitWarsCudaStatus orbit_wars_cuda_encode_identity_batch(
    const float* input_rows,
    float* output_rows,
    size_t game_count,
    size_t row_count,
    size_t feature_count);

OrbitWarsCudaStatus orbit_wars_cuda_trainable_forward(
    const float* input_rows,
    const float* weights,
    float* output_rows,
    size_t game_count,
    OrbitWarsCudaTransformerShape shape);

OrbitWarsCudaStatus orbit_wars_cuda_true2d_forward(
    const float* input_rows,
    const float* weights,
    float* output_rows,
    size_t game_count,
    OrbitWarsCudaTrue2DShape shape);

OrbitWarsCudaStatus orbit_wars_cuda_true2d_forward_many(
    const float* input_rows,
    const size_t* model_indices,
    const float* all_weights,
    float* output_rows,
    size_t request_count,
    size_t model_count,
    OrbitWarsCudaTrue2DShape shape);

OrbitWarsCudaStatus orbit_wars_cuda_true2d_upload_population(
    const float* all_weights,
    size_t model_count,
    size_t request_capacity,
    OrbitWarsCudaTrue2DShape shape);

OrbitWarsCudaStatus orbit_wars_cuda_true2d_forward_many_resident(
    const float* input_rows,
    const size_t* model_indices,
    float* output_rows,
    size_t request_count,
    OrbitWarsCudaTrue2DShape shape);

OrbitWarsCudaStatus orbit_wars_cuda_true2d_train_full_attention(
    const float* input_rows,
    const float* output_rows,
    const int* rewards,
    const float* target_action_rewards,
    const size_t* model_indices,
    float* all_weights,
    size_t sample_count,
    size_t model_count,
    OrbitWarsCudaTrue2DShape shape,
    OrbitWarsCudaTrue2DTrainingConfig training_config);

#ifdef __cplusplus
}
#endif
