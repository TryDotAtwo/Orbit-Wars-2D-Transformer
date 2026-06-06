#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OrbitWarsV8CudaStatus {
  int code;
  const char* message;
} OrbitWarsV8CudaStatus;

typedef struct OrbitWarsV8CudaShape {
  size_t batch_count;
  size_t token_count;
  size_t token_features;
  size_t d_model;
  size_t head_count;
  size_t encoder_layer_count;
  size_t decoder_layer_count;
  size_t action_slot_count;
  size_t planet_count;
  size_t amount_class_count;
} OrbitWarsV8CudaShape;

typedef struct OrbitWarsV8CudaTensor {
  const char* name;
  const float* data;
  size_t len;
} OrbitWarsV8CudaTensor;

typedef struct OrbitWarsV8CudaModel OrbitWarsV8CudaModel;
typedef struct OrbitWarsCudaSimState OrbitWarsCudaSimState;

typedef struct OrbitWarsCudaPlanet {
  int id;
  int owner;
  float x;
  float y;
  float radius;
  float ships;
  float production;
  float velocity_x;
  float velocity_y;
} OrbitWarsCudaPlanet;

typedef struct OrbitWarsCudaFleet {
  int id;
  int owner;
  float x;
  float y;
  float angle;
  int from_planet_id;
  float ships;
  unsigned char alive;
} OrbitWarsCudaFleet;

typedef struct OrbitWarsCudaAction {
  int from_planet_id;
  float direction_angle;
  int ship_count;
} OrbitWarsCudaAction;

typedef struct OrbitWarsCudaActionLabel {
  int fire[8];
  int source_row[8];
  int target_row[8];
  int amount_class[8];
  int source_planet_id[8];
  int target_planet_id[8];
  int ship_count[8];
  float confidence[8];
} OrbitWarsCudaActionLabel;

typedef struct OrbitWarsV8CudaDeviceBatchView {
  float* tokens;
  long long* token_type_ids;
  long long* owner_ids;
  unsigned char* padding_mask;
  unsigned char* planet_mask;
  OrbitWarsCudaActionLabel* labels;
  int* labels_fire;
  int* labels_source;
  int* labels_target;
  int* labels_amount;
  float* labels_confidence;
  size_t request_count;
  size_t token_count;
  size_t token_features;
  size_t planet_count;
  size_t action_slots;
} OrbitWarsV8CudaDeviceBatchView;

typedef struct OrbitWarsCudaSimConfig {
  size_t game_count;
  size_t planet_count;
  size_t max_fleets_per_game;
  size_t max_players;
  size_t max_actions_per_player;
  int step;
  int episode_steps;
  float angular_velocity;
  float board_size;
  float board_center;
  float sun_radius;
  float rotation_radius_limit;
  float fleet_speed_max;
  float fleet_speed_reference_ships;
  float fleet_speed_curve_power;
  float fleet_spawn_offset;
} OrbitWarsCudaSimConfig;

typedef struct OrbitWarsCudaSimStats {
  int launched_fleet_count;
  int launched_ship_count;
  int hit_fleet_count;
  int hit_ship_count;
  int sun_destroyed_fleet_count;
  int sun_destroyed_ship_count;
  int out_of_bounds_destroyed_fleet_count;
  int out_of_bounds_destroyed_ship_count;
  int captured_planet_count;
  int overflow_fleet_count;
} OrbitWarsCudaSimStats;

typedef struct OrbitWarsCudaGameStatus {
  int done;
  int winner;
  int step;
} OrbitWarsCudaGameStatus;

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_status(void);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_step(
    OrbitWarsCudaPlanet* planets,
    const OrbitWarsCudaPlanet* initial_planets,
    OrbitWarsCudaFleet* fleets,
    int* next_fleet_ids,
    const OrbitWarsCudaAction* actions,
    const int* action_counts,
    OrbitWarsCudaSimStats* stats,
    OrbitWarsCudaSimConfig config);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_create(
    OrbitWarsCudaSimConfig config,
    OrbitWarsCudaSimState** out_state);

void orbit_wars_cuda_sim_destroy(OrbitWarsCudaSimState* state);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_load(
    OrbitWarsCudaSimState* state,
    const OrbitWarsCudaPlanet* planets,
    const OrbitWarsCudaPlanet* initial_planets,
    const OrbitWarsCudaFleet* fleets,
    const int* next_fleet_ids);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_load_with_angular_velocities(
    OrbitWarsCudaSimState* state,
    const OrbitWarsCudaPlanet* planets,
    const OrbitWarsCudaPlanet* initial_planets,
    const OrbitWarsCudaFleet* fleets,
    const int* next_fleet_ids,
    const float* angular_velocities);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_step_persistent(
    OrbitWarsCudaSimState* state,
    const OrbitWarsCudaAction* actions,
    const int* action_counts,
    int step);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_clear_actions(
    OrbitWarsCudaSimState* state);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_load_request_plan(
    OrbitWarsCudaSimState* state,
    const int* request_game_indices,
    const int* request_player_ids,
    size_t request_count);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_step_device_actions(
    OrbitWarsCudaSimState* state,
    int step);

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_resident_model_decode(
    OrbitWarsV8CudaModel* model,
    OrbitWarsCudaSimState* state,
    const int* request_game_indices,
    const int* request_player_ids,
    size_t request_count,
    int step);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_read(
    OrbitWarsCudaSimState* state,
    OrbitWarsCudaPlanet* planets,
    OrbitWarsCudaFleet* fleets,
    int* next_fleet_ids,
    OrbitWarsCudaSimStats* stats);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_read_planets_stats(
    OrbitWarsCudaSimState* state,
    OrbitWarsCudaPlanet* planets,
    int* next_fleet_ids,
    OrbitWarsCudaSimStats* stats);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_read_status_stats(
    OrbitWarsCudaSimState* state,
    OrbitWarsCudaGameStatus* statuses,
    OrbitWarsCudaSimStats* stats);

OrbitWarsV8CudaStatus orbit_wars_cuda_sim_read_actions(
    OrbitWarsCudaSimState* state,
    OrbitWarsCudaAction* actions,
    int* action_counts);

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_model_create(
    const OrbitWarsV8CudaTensor* tensors,
    size_t tensor_count,
    OrbitWarsV8CudaModel** out_model);

void orbit_wars_cuda_v8_model_destroy(OrbitWarsV8CudaModel* model);

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_forward_cached(
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
    OrbitWarsV8CudaShape shape);

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_forward(
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
    OrbitWarsV8CudaShape shape);

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_resident_models_decode(
    OrbitWarsV8CudaModel** models,
    size_t model_count,
    OrbitWarsCudaSimState* state,
    const int* request_offsets,
    const int* request_counts,
    const int* request_game_indices,
    const int* request_player_ids,
    size_t request_total,
    int step);

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_resident_models_decode_plan(
    OrbitWarsV8CudaModel** models,
    size_t model_count,
    OrbitWarsCudaSimState* state,
    const int* request_offsets,
    const int* request_counts,
    size_t request_total,
    int step);

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_resident_models_step_plan(
    OrbitWarsV8CudaModel** models,
    size_t model_count,
    OrbitWarsCudaSimState* state,
    const int* request_offsets,
    const int* request_counts,
    size_t request_total,
    int step);

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_read_last_batch(
    OrbitWarsV8CudaModel* model,
    float* tokens,
    long long* token_type_ids,
    long long* owner_ids,
    unsigned char* padding_mask,
    unsigned char* planet_mask,
    OrbitWarsCudaActionLabel* labels,
    size_t request_capacity,
    size_t* out_request_count);

OrbitWarsV8CudaStatus orbit_wars_cuda_v8_last_batch_device_view(
    OrbitWarsV8CudaModel* model,
    OrbitWarsV8CudaDeviceBatchView* out_view);

#ifdef __cplusplus
}
#endif
