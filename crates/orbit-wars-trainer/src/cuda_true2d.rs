use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

use orbit_wars_core::{
    ActionOutput, ActionTargetOutput, AgentConfig, RowFeature, True2DTransformer,
};

const CUDA_LIBRARY_PATH_ENV: &str = "ORBIT_WARS_CUDA_LIB_PATH";
const DEFAULT_CUDA_LIBRARY_PATH: &str = "target/liborbit_wars_cuda.so";
const TRUE2D_FORWARD_SYMBOL: &[u8] = b"orbit_wars_cuda_true2d_forward\0";
const TRUE2D_FORWARD_MANY_SYMBOL: &[u8] = b"orbit_wars_cuda_true2d_forward_many\0";
const TRUE2D_UPLOAD_POPULATION_SYMBOL: &[u8] = b"orbit_wars_cuda_true2d_upload_population\0";
const TRUE2D_FORWARD_MANY_RESIDENT_SYMBOL: &[u8] =
    b"orbit_wars_cuda_true2d_forward_many_resident\0";
const TRUE2D_TRAIN_FULL_ATTENTION_SYMBOL: &[u8] = b"orbit_wars_cuda_true2d_train_full_attention\0";
const CUDA_STATUS_OK: c_int = 0;
const RTLD_NOW: c_int = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct OrbitWarsCudaStatus {
    code: c_int,
    message: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OrbitWarsCudaTrue2DShape {
    layer_count: usize,
    row_count: usize,
    head_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OrbitWarsCudaTrue2DTrainingConfig {
    batch_sample_count: usize,
    learning_rate: f32,
    reward_normalizer: f32,
    send_threshold: f32,
    owner_class_own: f32,
    owner_class_empty: f32,
    owner_class_absent_on_map: f32,
}

type True2DForwardFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *mut f32,
    usize,
    OrbitWarsCudaTrue2DShape,
) -> OrbitWarsCudaStatus;

type True2DForwardManyFn = unsafe extern "C" fn(
    *const f32,
    *const usize,
    *const f32,
    *mut f32,
    usize,
    usize,
    OrbitWarsCudaTrue2DShape,
) -> OrbitWarsCudaStatus;

type True2DUploadPopulationFn =
    unsafe extern "C" fn(*const f32, usize, usize, OrbitWarsCudaTrue2DShape) -> OrbitWarsCudaStatus;

type True2DForwardManyResidentFn = unsafe extern "C" fn(
    *const f32,
    *const usize,
    *mut f32,
    usize,
    OrbitWarsCudaTrue2DShape,
) -> OrbitWarsCudaStatus;

type True2DTrainFullAttentionFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const i32,
    *const f32,
    *const usize,
    *mut f32,
    usize,
    usize,
    OrbitWarsCudaTrue2DShape,
    OrbitWarsCudaTrue2DTrainingConfig,
) -> OrbitWarsCudaStatus;

pub struct CudaTrue2D {
    library_handle: *mut c_void,
    true2d_forward: True2DForwardFn,
    true2d_forward_many: True2DForwardManyFn,
    true2d_upload_population: True2DUploadPopulationFn,
    true2d_forward_many_resident: True2DForwardManyResidentFn,
    true2d_train_full_attention: True2DTrainFullAttentionFn,
}

impl CudaTrue2D {
    pub fn open_from_environment() -> Result<Self, String> {
        let library_path = std::env::var(CUDA_LIBRARY_PATH_ENV)
            .unwrap_or_else(|_| DEFAULT_CUDA_LIBRARY_PATH.to_string());
        Self::open(&library_path)
    }

    pub fn forward_packed(
        &self,
        input_rows: &[f32],
        model: &True2DTransformer,
        game_count: usize,
        config: &AgentConfig,
    ) -> Result<Vec<ActionOutput>, String> {
        let expected_input_count = game_count
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_input_features))
            .ok_or_else(|| "cuda_true2d_input_count_overflow".to_string())?;
        if input_rows.len() != expected_input_count {
            return Err("cuda_true2d_input_count_mismatch".to_string());
        }
        let expected_weight_count = model
            .shape
            .parameter_count(config)
            .map_err(|error| format!("cuda_true2d_weight_count_failed={error:?}"))?;
        if model.weights.len() != expected_weight_count {
            return Err("cuda_true2d_weight_count_mismatch".to_string());
        }

        let expected_output_count = game_count
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_output_features))
            .ok_or_else(|| "cuda_true2d_output_count_overflow".to_string())?;
        let mut output_rows = vec![0.0f32; expected_output_count];
        let shape = OrbitWarsCudaTrue2DShape {
            layer_count: model.shape.layers,
            row_count: model.shape.row_count,
            head_count: config.true2d_attention_heads,
        };
        let status = unsafe {
            (self.true2d_forward)(
                input_rows.as_ptr(),
                model.weights.as_ptr(),
                output_rows.as_mut_ptr(),
                game_count,
                shape,
            )
        };
        require_ok(status, "cuda_true2d_forward")?;
        unpack_outputs(&output_rows, game_count, config)
    }

    pub fn upload_population(
        &self,
        population: &[True2DTransformer],
        request_capacity: usize,
        config: &AgentConfig,
    ) -> Result<(), String> {
        if request_capacity == 0 {
            return Err("cuda_true2d_upload_request_capacity_zero".to_string());
        }
        let (shape, expected_weight_count) = validate_population(population, config)?;
        let mut all_weights = Vec::with_capacity(population.len() * expected_weight_count);
        for model in population {
            all_weights.extend_from_slice(&model.weights);
        }
        let cuda_shape = cuda_shape(shape, config);
        let status = unsafe {
            (self.true2d_upload_population)(
                all_weights.as_ptr(),
                population.len(),
                request_capacity,
                cuda_shape,
            )
        };
        require_ok(status, "cuda_true2d_upload_population")
    }

    pub fn forward_population_packed(
        &self,
        input_rows: &[f32],
        model_indices: &[usize],
        population: &[True2DTransformer],
        config: &AgentConfig,
    ) -> Result<Vec<ActionOutput>, String> {
        let request_count = model_indices.len();
        if request_count == 0 {
            return Ok(Vec::new());
        }
        if population.is_empty() {
            return Err("cuda_true2d_population_empty".to_string());
        }
        let expected_input_count = request_count
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_input_features))
            .ok_or_else(|| "cuda_true2d_many_input_count_overflow".to_string())?;
        if input_rows.len() != expected_input_count {
            return Err("cuda_true2d_many_input_count_mismatch".to_string());
        }
        let (shape, expected_weight_count) = validate_population(population, config)?;
        let mut all_weights = Vec::with_capacity(population.len() * expected_weight_count);
        for model in population {
            all_weights.extend_from_slice(&model.weights);
        }
        for (request_index, model_index) in model_indices.iter().copied().enumerate() {
            if model_index >= population.len() {
                return Err(format!(
                    "cuda_true2d_many_model_index_out_of_range={request_index}:{model_index}"
                ));
            }
        }

        let expected_output_count = request_count
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_output_features))
            .ok_or_else(|| "cuda_true2d_many_output_count_overflow".to_string())?;
        let mut output_rows = vec![0.0f32; expected_output_count];
        let cuda_shape = cuda_shape(shape, config);
        let status = unsafe {
            (self.true2d_forward_many)(
                input_rows.as_ptr(),
                model_indices.as_ptr(),
                all_weights.as_ptr(),
                output_rows.as_mut_ptr(),
                request_count,
                population.len(),
                cuda_shape,
            )
        };
        require_ok(status, "cuda_true2d_forward_many")?;
        unpack_outputs(&output_rows, request_count, config)
    }

    pub fn forward_population_resident_packed(
        &self,
        input_rows: &[f32],
        model_indices: &[usize],
        population: &[True2DTransformer],
        config: &AgentConfig,
    ) -> Result<Vec<ActionOutput>, String> {
        let request_count = model_indices.len();
        if request_count == 0 {
            return Ok(Vec::new());
        }
        let (shape, _) = validate_population(population, config)?;
        let expected_input_count = request_count
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_input_features))
            .ok_or_else(|| "cuda_true2d_resident_input_count_overflow".to_string())?;
        if input_rows.len() != expected_input_count {
            return Err("cuda_true2d_resident_input_count_mismatch".to_string());
        }
        for (request_index, model_index) in model_indices.iter().copied().enumerate() {
            if model_index >= population.len() {
                return Err(format!(
                    "cuda_true2d_resident_model_index_out_of_range={request_index}:{model_index}"
                ));
            }
        }

        let expected_output_count = request_count
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_output_features))
            .ok_or_else(|| "cuda_true2d_resident_output_count_overflow".to_string())?;
        let mut output_rows = vec![0.0f32; expected_output_count];
        let status = unsafe {
            (self.true2d_forward_many_resident)(
                input_rows.as_ptr(),
                model_indices.as_ptr(),
                output_rows.as_mut_ptr(),
                request_count,
                cuda_shape(shape, config),
            )
        };
        require_ok(status, "cuda_true2d_forward_many_resident")?;
        unpack_outputs(&output_rows, request_count, config)
    }

    pub fn train_full_attention_population(
        &self,
        input_rows: &[f32],
        output_rows: &[f32],
        rewards: &[i32],
        sun_target_penalties: &[f32],
        model_indices: &[usize],
        population: &mut [True2DTransformer],
        config: &AgentConfig,
    ) -> Result<(), String> {
        let sample_count = rewards.len();
        if sample_count == 0 {
            return Ok(());
        }
        if model_indices.len() != sample_count {
            return Err("cuda_true2d_train_model_index_count_mismatch".to_string());
        }
        let expected_sun_target_penalty_count = sample_count
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_action_targets_per_source))
            .ok_or_else(|| "cuda_true2d_train_sun_target_penalty_count_overflow".to_string())?;
        if sun_target_penalties.len() != expected_sun_target_penalty_count {
            return Err("cuda_true2d_train_sun_target_penalty_count_mismatch".to_string());
        }
        let expected_input_count = sample_count
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_input_features))
            .ok_or_else(|| "cuda_true2d_train_input_count_overflow".to_string())?;
        if input_rows.len() != expected_input_count {
            return Err("cuda_true2d_train_input_count_mismatch".to_string());
        }
        let expected_output_count = sample_count
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_output_features))
            .ok_or_else(|| "cuda_true2d_train_output_count_overflow".to_string())?;
        if output_rows.len() != expected_output_count {
            return Err("cuda_true2d_train_output_count_mismatch".to_string());
        }
        let (shape, expected_weight_count) = validate_population(population, config)?;
        for (sample_index, model_index) in model_indices.iter().copied().enumerate() {
            if model_index >= population.len() {
                return Err(format!(
                    "cuda_true2d_train_model_index_out_of_range={sample_index}:{model_index}"
                ));
            }
        }
        let mut all_weights = Vec::with_capacity(population.len() * expected_weight_count);
        for model in population.iter() {
            all_weights.extend_from_slice(&model.weights);
        }
        let training_config = OrbitWarsCudaTrue2DTrainingConfig {
            batch_sample_count: config.training_backprop_batch_samples,
            learning_rate: config.training_backprop_learning_rate,
            reward_normalizer: config.training_backprop_reward_normalizer,
            send_threshold: config.training_backprop_send_threshold,
            owner_class_own: config.owner_class_own,
            owner_class_empty: config.owner_class_empty,
            owner_class_absent_on_map: config.owner_class_absent_on_map,
        };
        let status = unsafe {
            (self.true2d_train_full_attention)(
                input_rows.as_ptr(),
                output_rows.as_ptr(),
                rewards.as_ptr(),
                sun_target_penalties.as_ptr(),
                model_indices.as_ptr(),
                all_weights.as_mut_ptr(),
                sample_count,
                population.len(),
                cuda_shape(shape, config),
                training_config,
            )
        };
        require_ok(status, "cuda_true2d_train_full_attention")?;
        if all_weights.iter().any(|weight| !weight.is_finite()) {
            return Err("cuda_true2d_train_non_finite_weight".to_string());
        }
        for (model_index, model) in population.iter_mut().enumerate() {
            let start = model_index * expected_weight_count;
            let end = start + expected_weight_count;
            model.weights.copy_from_slice(&all_weights[start..end]);
        }
        Ok(())
    }

    fn open(library_path: &str) -> Result<Self, String> {
        let path =
            CString::new(library_path).map_err(|_| "cuda_library_path_contains_nul".to_string())?;
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(format!("cuda_library_open_failed={}", dl_error_message()));
        }
        let forward_symbol =
            unsafe { dlsym(handle, TRUE2D_FORWARD_SYMBOL.as_ptr().cast::<c_char>()) };
        if forward_symbol.is_null() {
            let message = dl_error_message();
            unsafe {
                dlclose(handle);
            }
            return Err(format!("cuda_symbol_lookup_failed={message}"));
        }
        let forward_many_symbol =
            unsafe { dlsym(handle, TRUE2D_FORWARD_MANY_SYMBOL.as_ptr().cast::<c_char>()) };
        if forward_many_symbol.is_null() {
            let message = dl_error_message();
            unsafe {
                dlclose(handle);
            }
            return Err(format!("cuda_symbol_lookup_failed={message}"));
        }
        let upload_population_symbol = unsafe {
            dlsym(
                handle,
                TRUE2D_UPLOAD_POPULATION_SYMBOL.as_ptr().cast::<c_char>(),
            )
        };
        if upload_population_symbol.is_null() {
            let message = dl_error_message();
            unsafe {
                dlclose(handle);
            }
            return Err(format!("cuda_symbol_lookup_failed={message}"));
        }
        let forward_many_resident_symbol = unsafe {
            dlsym(
                handle,
                TRUE2D_FORWARD_MANY_RESIDENT_SYMBOL
                    .as_ptr()
                    .cast::<c_char>(),
            )
        };
        if forward_many_resident_symbol.is_null() {
            let message = dl_error_message();
            unsafe {
                dlclose(handle);
            }
            return Err(format!("cuda_symbol_lookup_failed={message}"));
        }
        let train_full_attention_symbol = unsafe {
            dlsym(
                handle,
                TRUE2D_TRAIN_FULL_ATTENTION_SYMBOL.as_ptr().cast::<c_char>(),
            )
        };
        if train_full_attention_symbol.is_null() {
            let message = dl_error_message();
            unsafe {
                dlclose(handle);
            }
            return Err(format!("cuda_symbol_lookup_failed={message}"));
        }
        Ok(Self {
            library_handle: handle,
            true2d_forward: unsafe {
                std::mem::transmute::<*mut c_void, True2DForwardFn>(forward_symbol)
            },
            true2d_forward_many: unsafe {
                std::mem::transmute::<*mut c_void, True2DForwardManyFn>(forward_many_symbol)
            },
            true2d_upload_population: unsafe {
                std::mem::transmute::<*mut c_void, True2DUploadPopulationFn>(
                    upload_population_symbol,
                )
            },
            true2d_forward_many_resident: unsafe {
                std::mem::transmute::<*mut c_void, True2DForwardManyResidentFn>(
                    forward_many_resident_symbol,
                )
            },
            true2d_train_full_attention: unsafe {
                std::mem::transmute::<*mut c_void, True2DTrainFullAttentionFn>(
                    train_full_attention_symbol,
                )
            },
        })
    }
}

impl Drop for CudaTrue2D {
    fn drop(&mut self) {
        if !self.library_handle.is_null() {
            unsafe {
                dlclose(self.library_handle);
            }
        }
    }
}

pub fn pack_rows(rows: &[RowFeature], config: &AgentConfig) -> Result<Vec<f32>, String> {
    let mut packed = Vec::with_capacity(config.max_rows * config.true2d_input_features);
    pack_rows_into(rows, config, &mut packed)?;
    Ok(packed)
}

pub fn pack_rows_into(
    rows: &[RowFeature],
    config: &AgentConfig,
    packed: &mut Vec<f32>,
) -> Result<(), String> {
    if rows.len() != config.max_rows {
        return Err("cuda_true2d_row_count_mismatch".to_string());
    }
    for row in rows {
        packed.push(row.owner_class);
        packed.push(row.ship_log_percent);
        packed.push(row.x_position_normalized);
        packed.push(row.y_position_normalized);
        packed.push(row.production_normalized);
        packed.push(row.velocity_x_normalized);
        packed.push(row.velocity_y_normalized);
    }
    Ok(())
}

fn validate_population(
    population: &[True2DTransformer],
    config: &AgentConfig,
) -> Result<(orbit_wars_core::True2DTransformerShape, usize), String> {
    let first = population
        .first()
        .ok_or_else(|| "cuda_true2d_population_empty".to_string())?;
    let shape = first.shape;
    let expected_weight_count = shape
        .parameter_count(config)
        .map_err(|error| format!("cuda_true2d_weight_count_failed={error:?}"))?;
    for (model_index, model) in population.iter().enumerate() {
        if model.shape != shape {
            return Err(format!("cuda_true2d_shape_mismatch={model_index}"));
        }
        if model.weights.len() != expected_weight_count {
            return Err(format!("cuda_true2d_weight_count_mismatch={model_index}"));
        }
    }
    Ok((shape, expected_weight_count))
}

fn cuda_shape(
    shape: orbit_wars_core::True2DTransformerShape,
    config: &AgentConfig,
) -> OrbitWarsCudaTrue2DShape {
    OrbitWarsCudaTrue2DShape {
        layer_count: shape.layers,
        row_count: shape.row_count,
        head_count: config.true2d_attention_heads,
    }
}

fn unpack_outputs(
    output_rows: &[f32],
    game_count: usize,
    config: &AgentConfig,
) -> Result<Vec<ActionOutput>, String> {
    let expected = game_count
        .checked_mul(config.max_rows)
        .and_then(|value| value.checked_mul(config.true2d_output_features))
        .ok_or_else(|| "cuda_true2d_unpack_count_overflow".to_string())?;
    if output_rows.len() != expected {
        return Err("cuda_true2d_unpack_count_mismatch".to_string());
    }
    let mut outputs = Vec::with_capacity(game_count * config.max_rows);
    for row_index in 0..game_count * config.max_rows {
        let offset = row_index * config.true2d_output_features;
        let mut targets = [ActionTargetOutput {
            target_fraction: 0.0,
            send_fraction: 0.0,
        }; orbit_wars_core::config::TRUE2D_ACTION_TARGETS_PER_SOURCE];
        for (target_index, target_output) in targets.iter_mut().enumerate() {
            let target_offset =
                offset + target_index * orbit_wars_core::config::TRUE2D_ACTION_TARGET_FEATURES;
            let target_fraction = output_rows[target_offset];
            let send_fraction = output_rows[target_offset + 1];
            if !(target_fraction.is_finite() && send_fraction.is_finite()) {
                return Err("cuda_true2d_output_non_finite".to_string());
            }
            *target_output = ActionTargetOutput {
                target_fraction,
                send_fraction,
            };
        }
        outputs.push(ActionOutput { targets });
    }
    Ok(outputs)
}

fn require_ok(status: OrbitWarsCudaStatus, operation: &str) -> Result<(), String> {
    if status.code == CUDA_STATUS_OK {
        return Ok(());
    }
    let message = if status.message.is_null() {
        "null_status_message".to_string()
    } else {
        unsafe { CStr::from_ptr(status.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(format!("{operation}_failed={message}"))
}

fn dl_error_message() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() {
        "unknown_dl_error".to_string()
    } else {
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(target_os = "linux")]
#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

#[cfg(not(target_os = "linux"))]
unsafe fn dlopen(_filename: *const c_char, _flag: c_int) -> *mut c_void {
    std::ptr::null_mut()
}

#[cfg(not(target_os = "linux"))]
unsafe fn dlsym(_handle: *mut c_void, _symbol: *const c_char) -> *mut c_void {
    std::ptr::null_mut()
}

#[cfg(not(target_os = "linux"))]
unsafe fn dlclose(_handle: *mut c_void) -> c_int {
    CUDA_STATUS_OK
}

#[cfg(not(target_os = "linux"))]
unsafe fn dlerror() -> *const c_char {
    std::ptr::null()
}
