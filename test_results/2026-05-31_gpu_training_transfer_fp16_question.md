timestamp=2026-05-31T16:11:26+03:00
task_id=2026-05-31_gpu_training_transfer_fp16_question
branch=HEAD_unborn
commit=none
environment=cmz-native-dev:2026-05-26; docker_gpus=all; cuda=12.4
commands=[cargo fmt --all,cargo test --workspace,nvcc_shared_library_build,gpp_trainable_smoke_build,gpp_true2d_smoke_build,LD_LIBRARY_PATH=target ./target/orbit_wars_cuda_true2d_smoke,ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --cuda-true2d-smoke,ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --smoke --cuda,LD_LIBRARY_PATH=target ./target/orbit_wars_cuda_smoke]
result=pass
artifacts=[target/liborbit_wars_cuda.so,target/orbit_wars_cuda_true2d_smoke,target/orbit_wars_cuda_smoke,dashboard/public/telemetry/latest.json]
logs=[]
rust_format=passed
rust_tests=passed; count=26
cuda_build=passed
cuda_true2d_cpp_smoke=passed; output="status=ok; cuda_true2d_forward_smoke=true; games=2; rows=64; max_abs_diff=1.19209e-07"
cuda_true2d_rust_smoke=passed; output="status=ok; cuda_true2d_smoke=true; games=2; rows=64; max_abs_diff=0.00000018"
trainer_cuda_smoke=passed; output="status=ok; smoke=true; cuda=true; population=8; elite=2; best_model=4; best_reward=4"; wall_time_seconds=22.1
cuda_trainable_legacy_smoke=passed; output="status=ok; cuda_trainable_forward_smoke=true; games=4; rows=64; d_model=32"
fp16_policy=not_enabled; reason=final_cpu_inference_requires_fp32_export_and_fp32_cuda_reference_must_stay_verified_first
conclusion=true2d_cuda_forward_and_explicit_trainer_cuda_mode_verified
