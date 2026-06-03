# Orbit Wars CUDA Module

status=scaffold
submission_path=false
purpose=local_self_play_training_batch_inference

## Contract

- `orbit_wars_cuda_status` verifies CUDA device visibility.
- `orbit_wars_cuda_encode_identity_batch` is a buildable CUDA smoke kernel for batched row tensors.
- `orbit_wars_cuda_trainable_forward` runs host-buffer batched forward inference for the Rust `TrainableTransformer` weight layout.
- `orbit_wars_cuda_true2d_forward` runs host-buffer batched forward inference for the Rust `True2DTransformer` weight layout.
- `orbit_wars_cuda_true2d_forward_many` runs one CUDA batch with `model_indices[request]` selecting one model from contiguous `all_weights`.
- `orbit_wars_cuda_true2d_upload_population` uploads contiguous `all_weights` into the persistent true2d many-forward VRAM workspace.
- `orbit_wars_cuda_true2d_forward_many_resident` runs one CUDA batch with `model_indices[request]` selecting resident uploaded model weights.
- `orbit_wars_cuda_true2d_train_full_attention` runs selected-sample training on GPU: full forward/backward through learned expand, pre-attention 64x64 matrix layers, one middle full 4096x4096 Q/K/V attention, post-attention 64x64 matrix layers, and 64x24 output collapse, then writes updated fp32 weights back to the host buffer.
- `target_action_rewards` is a separate `sample_count x 64 x 12` fp32 buffer for action-local target penalties. Target logits use final reward plus this local value; send logits use only the final game reward.
- weight_layout=`input_projection[2*d_model] + position_embedding[d_model] + layer_scales[layer_count*d_model] + output_projection[2*d_model]`.
- true2d_input_layout=`game_count x 64 x 7` with `owner_class, ship_log_percent, x_position_normalized, y_position_normalized, production_normalized, velocity_x_normalized, velocity_y_normalized`.
- true2d_attention=`one middle full Q/K/V self-attention over 4096 relation cells; score shape=4096x4096 per head`.
- true2d_weight_layout=`input_pair_weights[10] + source_row_embedding[64] + target_row_embedding[64] + pair_embedding[64*64] + expand_bias[1] + matrix_layer_weights[layers*64*64*12] + attention_weights[heads*7+6] + target_output_bias[12*64] + send_output_bias[12*64] + output_scalar_weights[12*5]`.
- true2d_default_parameter_count=3004120.
- true2d_many_workspace=`persistent_device_buffers_and_resident_population_resize_on_capacity_or_shape_change`.
- true2d_backprop_batch_samples controls CUDA training chunk memory only; it is not a sample cap.
- trainer_gpu_binding_status=enabled_by_explicit_flag; `orbit-wars-trainer --cuda` requires compiled library and stops on CUDA errors.
- trainer_training_status=`--cuda` required when backprop is enabled; CPU inference remains the final submission path.
- precision_policy=fp32 stored/final weights for CPU inference; CUDA inference attention uses bf16-rounded operands with fp32 accumulation; CUDA backprop remains fp32 unless a separate training-precision contract is added.

## Build Smoke Command

```bash
nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so
```

## True2D Smoke Command

```bash
nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so
g++ -std=c++17 -O3 native/cuda/orbit_wars_cuda_true2d_smoke.cpp -Inative/cuda -Ltarget -lorbit_wars_cuda -Wl,-rpath,'$ORIGIN' -o target/orbit_wars_cuda_true2d_smoke
LD_LIBRARY_PATH=target ./target/orbit_wars_cuda_true2d_smoke
ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --cuda-true2d-smoke
```
