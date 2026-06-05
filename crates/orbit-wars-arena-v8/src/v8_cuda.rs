use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

use orbit_wars_core::{ActionSlotOutput, AgentConfig, Fleet, MoveCommand, Planet, V8Model, V8Tokens};

const CUDA_STATUS_OK: c_int = 0;
const RTLD_NOW: c_int = 2;
const STATUS_SYMBOL: &[u8] = b"orbit_wars_cuda_v8_status\0";
const FORWARD_SYMBOL: &[u8] = b"orbit_wars_cuda_v8_forward\0";
const MODEL_CREATE_SYMBOL: &[u8] = b"orbit_wars_cuda_v8_model_create\0";
const MODEL_DESTROY_SYMBOL: &[u8] = b"orbit_wars_cuda_v8_model_destroy\0";
const FORWARD_CACHED_SYMBOL: &[u8] = b"orbit_wars_cuda_v8_forward_cached\0";
const SIM_STEP_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_step\0";
const SIM_CREATE_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_create\0";
const SIM_DESTROY_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_destroy\0";
const SIM_LOAD_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_load\0";
const SIM_LOAD_WITH_ANGULAR_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_load_with_angular_velocities\0";
const SIM_STEP_PERSISTENT_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_step_persistent\0";
const SIM_CLEAR_ACTIONS_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_clear_actions\0";
const SIM_STEP_DEVICE_ACTIONS_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_step_device_actions\0";
const SIM_READ_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_read\0";
const SIM_READ_PLANETS_STATS_SYMBOL: &[u8] = b"orbit_wars_cuda_sim_read_planets_stats\0";
const RESIDENT_MODEL_DECODE_SYMBOL: &[u8] = b"orbit_wars_cuda_v8_resident_model_decode\0";
const DEFAULT_CUDA_V8_LIBRARY_PATH: &str = "target/liborbit_wars_v8_cuda.so";
const TOKEN_FEATURES: usize = 14;
const ACTION_SLOTS: usize = 8;
const PLANETS: usize = 64;
const AMOUNTS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy)]
struct OrbitWarsV8CudaStatus {
    code: c_int,
    message: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OrbitWarsV8CudaShape {
    batch_count: usize,
    token_count: usize,
    token_features: usize,
    d_model: usize,
    head_count: usize,
    encoder_layer_count: usize,
    decoder_layer_count: usize,
    action_slot_count: usize,
    planet_count: usize,
    amount_class_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OrbitWarsV8CudaTensor {
    name: *const c_char,
    data: *const f32,
    len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OrbitWarsCudaPlanet {
    pub id: i32,
    pub owner: i32,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub ships: f32,
    pub production: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OrbitWarsCudaFleet {
    pub id: i32,
    pub owner: i32,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub from_planet_id: i32,
    pub ships: f32,
    pub alive: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OrbitWarsCudaAction {
    pub from_planet_id: i32,
    pub direction_angle: f32,
    pub ship_count: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OrbitWarsCudaSimConfig {
    game_count: usize,
    planet_count: usize,
    max_fleets_per_game: usize,
    max_players: usize,
    max_actions_per_player: usize,
    step: i32,
    angular_velocity: f32,
    board_size: f32,
    board_center: f32,
    sun_radius: f32,
    rotation_radius_limit: f32,
    fleet_speed_max: f32,
    fleet_speed_reference_ships: f32,
    fleet_speed_curve_power: f32,
    fleet_spawn_offset: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OrbitWarsCudaSimStats {
    pub launched_fleet_count: i32,
    pub launched_ship_count: i32,
    pub hit_fleet_count: i32,
    pub hit_ship_count: i32,
    pub sun_destroyed_fleet_count: i32,
    pub sun_destroyed_ship_count: i32,
    pub out_of_bounds_destroyed_fleet_count: i32,
    pub out_of_bounds_destroyed_ship_count: i32,
    pub captured_planet_count: i32,
    pub overflow_fleet_count: i32,
}

impl From<Planet> for OrbitWarsCudaPlanet {
    fn from(planet: Planet) -> Self {
        Self {
            id: planet.id,
            owner: planet.owner,
            x: planet.x,
            y: planet.y,
            radius: planet.radius,
            ships: planet.ships,
            production: planet.production,
            velocity_x: planet.velocity_x,
            velocity_y: planet.velocity_y,
        }
    }
}

impl From<OrbitWarsCudaPlanet> for Planet {
    fn from(planet: OrbitWarsCudaPlanet) -> Self {
        Self {
            id: planet.id,
            owner: planet.owner,
            x: planet.x,
            y: planet.y,
            radius: planet.radius,
            ships: planet.ships,
            production: planet.production,
            velocity_x: planet.velocity_x,
            velocity_y: planet.velocity_y,
        }
    }
}

impl From<Fleet> for OrbitWarsCudaFleet {
    fn from(fleet: Fleet) -> Self {
        Self {
            id: fleet.id,
            owner: fleet.owner,
            x: fleet.x,
            y: fleet.y,
            angle: fleet.angle,
            from_planet_id: fleet.from_planet_id,
            ships: fleet.ships,
            alive: 1,
        }
    }
}

impl From<OrbitWarsCudaFleet> for Fleet {
    fn from(fleet: OrbitWarsCudaFleet) -> Self {
        Self {
            id: fleet.id,
            owner: fleet.owner,
            x: fleet.x,
            y: fleet.y,
            angle: fleet.angle,
            from_planet_id: fleet.from_planet_id,
            ships: fleet.ships,
        }
    }
}

impl From<MoveCommand> for OrbitWarsCudaAction {
    fn from(action: MoveCommand) -> Self {
        Self {
            from_planet_id: action.from_planet_id,
            direction_angle: action.direction_angle,
            ship_count: action.ship_count,
        }
    }
}

impl OrbitWarsCudaSimConfig {
    pub fn new(
        game_count: usize,
        planet_count: usize,
        max_fleets_per_game: usize,
        max_players: usize,
        max_actions_per_player: usize,
        step: usize,
        angular_velocity: f32,
        config: &AgentConfig,
    ) -> Self {
        Self {
            game_count,
            planet_count,
            max_fleets_per_game,
            max_players,
            max_actions_per_player,
            step: step as i32,
            angular_velocity,
            board_size: config.board_size,
            board_center: config.board_center,
            sun_radius: config.sun_radius,
            rotation_radius_limit: config.rotation_radius_limit,
            fleet_speed_max: config.fleet_speed_max,
            fleet_speed_reference_ships: config.fleet_speed_reference_ships,
            fleet_speed_curve_power: config.fleet_speed_curve_power,
            fleet_spawn_offset: config.fleet_spawn_offset,
        }
    }
}

type V8StatusFn = unsafe extern "C" fn() -> OrbitWarsV8CudaStatus;
type V8ForwardFn = unsafe extern "C" fn(
    tokens: *const f32,
    token_type_ids: *const i64,
    owner_ids: *const i64,
    padding_mask: *const u8,
    planet_mask: *const u8,
    tensors: *const OrbitWarsV8CudaTensor,
    tensor_count: usize,
    fire_logits: *mut f32,
    source_logits: *mut f32,
    target_logits: *mut f32,
    amount_logits: *mut f32,
    shape: OrbitWarsV8CudaShape,
) -> OrbitWarsV8CudaStatus;
type V8ModelCreateFn = unsafe extern "C" fn(
    tensors: *const OrbitWarsV8CudaTensor,
    tensor_count: usize,
    out_model: *mut *mut c_void,
) -> OrbitWarsV8CudaStatus;
type V8ModelDestroyFn = unsafe extern "C" fn(model: *mut c_void);
type V8ForwardCachedFn = unsafe extern "C" fn(
    model: *mut c_void,
    tokens: *const f32,
    token_type_ids: *const i64,
    owner_ids: *const i64,
    padding_mask: *const u8,
    planet_mask: *const u8,
    fire_logits: *mut f32,
    source_logits: *mut f32,
    target_logits: *mut f32,
    amount_logits: *mut f32,
    shape: OrbitWarsV8CudaShape,
) -> OrbitWarsV8CudaStatus;
type SimStepFn = unsafe extern "C" fn(
    planets: *mut OrbitWarsCudaPlanet,
    initial_planets: *const OrbitWarsCudaPlanet,
    fleets: *mut OrbitWarsCudaFleet,
    next_fleet_ids: *mut i32,
    actions: *const OrbitWarsCudaAction,
    action_counts: *const i32,
    stats: *mut OrbitWarsCudaSimStats,
    config: OrbitWarsCudaSimConfig,
) -> OrbitWarsV8CudaStatus;
type SimCreateFn = unsafe extern "C" fn(
    config: OrbitWarsCudaSimConfig,
    out_state: *mut *mut c_void,
) -> OrbitWarsV8CudaStatus;
type SimDestroyFn = unsafe extern "C" fn(state: *mut c_void);
type SimLoadFn = unsafe extern "C" fn(
    state: *mut c_void,
    planets: *const OrbitWarsCudaPlanet,
    initial_planets: *const OrbitWarsCudaPlanet,
    fleets: *const OrbitWarsCudaFleet,
    next_fleet_ids: *const i32,
) -> OrbitWarsV8CudaStatus;
type SimLoadWithAngularFn = unsafe extern "C" fn(
    state: *mut c_void,
    planets: *const OrbitWarsCudaPlanet,
    initial_planets: *const OrbitWarsCudaPlanet,
    fleets: *const OrbitWarsCudaFleet,
    next_fleet_ids: *const i32,
    angular_velocities: *const f32,
) -> OrbitWarsV8CudaStatus;
type SimStepPersistentFn = unsafe extern "C" fn(
    state: *mut c_void,
    actions: *const OrbitWarsCudaAction,
    action_counts: *const i32,
    step: i32,
) -> OrbitWarsV8CudaStatus;
type SimClearActionsFn = unsafe extern "C" fn(state: *mut c_void) -> OrbitWarsV8CudaStatus;
type SimStepDeviceActionsFn = unsafe extern "C" fn(
    state: *mut c_void,
    step: i32,
) -> OrbitWarsV8CudaStatus;
type SimReadFn = unsafe extern "C" fn(
    state: *mut c_void,
    planets: *mut OrbitWarsCudaPlanet,
    fleets: *mut OrbitWarsCudaFleet,
    next_fleet_ids: *mut i32,
    stats: *mut OrbitWarsCudaSimStats,
) -> OrbitWarsV8CudaStatus;
type SimReadPlanetsStatsFn = unsafe extern "C" fn(
    state: *mut c_void,
    planets: *mut OrbitWarsCudaPlanet,
    next_fleet_ids: *mut i32,
    stats: *mut OrbitWarsCudaSimStats,
) -> OrbitWarsV8CudaStatus;
type ResidentModelDecodeFn = unsafe extern "C" fn(
    model: *mut c_void,
    state: *mut c_void,
    request_game_indices: *const i32,
    request_player_ids: *const i32,
    request_count: usize,
    step: i32,
) -> OrbitWarsV8CudaStatus;

pub struct V8Cuda {
    library_handle: *mut c_void,
    status_fn: V8StatusFn,
    #[allow(dead_code)]
    forward_fn: V8ForwardFn,
    model_create_fn: V8ModelCreateFn,
    model_destroy_fn: V8ModelDestroyFn,
    forward_cached_fn: V8ForwardCachedFn,
    sim_step_fn: SimStepFn,
    sim_create_fn: SimCreateFn,
    sim_destroy_fn: SimDestroyFn,
    sim_load_fn: SimLoadFn,
    sim_load_with_angular_fn: SimLoadWithAngularFn,
    sim_step_persistent_fn: SimStepPersistentFn,
    sim_clear_actions_fn: SimClearActionsFn,
    sim_step_device_actions_fn: SimStepDeviceActionsFn,
    sim_read_fn: SimReadFn,
    sim_read_planets_stats_fn: SimReadPlanetsStatsFn,
    resident_model_decode_fn: ResidentModelDecodeFn,
}

impl V8Cuda {
    pub fn open_from_environment() -> Result<Self, String> {
        let path = std::env::var("ORBIT_WARS_V8_CUDA_LIB_PATH")
            .unwrap_or_else(|_| DEFAULT_CUDA_V8_LIBRARY_PATH.to_string());
        Self::open(&path)
    }

    pub fn open(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|_| "cuda_v8_path_contains_nul".to_string())?;
        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(format!("cuda_v8_dlopen_failed={}", dlerror_text()));
        }
        let status_fn = unsafe { dlsym(handle, STATUS_SYMBOL.as_ptr() as *const c_char) };
        if status_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_v8_status_symbol_missing={}", dlerror_text()));
        }
        let forward_fn = unsafe { dlsym(handle, FORWARD_SYMBOL.as_ptr() as *const c_char) };
        if forward_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_v8_forward_symbol_missing={}", dlerror_text()));
        }
        let model_create_fn = unsafe { dlsym(handle, MODEL_CREATE_SYMBOL.as_ptr() as *const c_char) };
        if model_create_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_v8_model_create_symbol_missing={}", dlerror_text()));
        }
        let model_destroy_fn = unsafe { dlsym(handle, MODEL_DESTROY_SYMBOL.as_ptr() as *const c_char) };
        if model_destroy_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_v8_model_destroy_symbol_missing={}", dlerror_text()));
        }
        let forward_cached_fn = unsafe { dlsym(handle, FORWARD_CACHED_SYMBOL.as_ptr() as *const c_char) };
        if forward_cached_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_v8_forward_cached_symbol_missing={}", dlerror_text()));
        }
        let sim_step_fn = unsafe { dlsym(handle, SIM_STEP_SYMBOL.as_ptr() as *const c_char) };
        if sim_step_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_step_symbol_missing={}", dlerror_text()));
        }
        let sim_create_fn = unsafe { dlsym(handle, SIM_CREATE_SYMBOL.as_ptr() as *const c_char) };
        if sim_create_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_create_symbol_missing={}", dlerror_text()));
        }
        let sim_destroy_fn = unsafe { dlsym(handle, SIM_DESTROY_SYMBOL.as_ptr() as *const c_char) };
        if sim_destroy_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_destroy_symbol_missing={}", dlerror_text()));
        }
        let sim_load_fn = unsafe { dlsym(handle, SIM_LOAD_SYMBOL.as_ptr() as *const c_char) };
        if sim_load_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_load_symbol_missing={}", dlerror_text()));
        }
        let sim_load_with_angular_fn = unsafe { dlsym(handle, SIM_LOAD_WITH_ANGULAR_SYMBOL.as_ptr() as *const c_char) };
        if sim_load_with_angular_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_load_with_angular_symbol_missing={}", dlerror_text()));
        }
        let sim_step_persistent_fn = unsafe { dlsym(handle, SIM_STEP_PERSISTENT_SYMBOL.as_ptr() as *const c_char) };
        if sim_step_persistent_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_step_persistent_symbol_missing={}", dlerror_text()));
        }
        let sim_clear_actions_fn = unsafe { dlsym(handle, SIM_CLEAR_ACTIONS_SYMBOL.as_ptr() as *const c_char) };
        if sim_clear_actions_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_clear_actions_symbol_missing={}", dlerror_text()));
        }
        let sim_step_device_actions_fn = unsafe { dlsym(handle, SIM_STEP_DEVICE_ACTIONS_SYMBOL.as_ptr() as *const c_char) };
        if sim_step_device_actions_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_step_device_actions_symbol_missing={}", dlerror_text()));
        }
        let sim_read_fn = unsafe { dlsym(handle, SIM_READ_SYMBOL.as_ptr() as *const c_char) };
        if sim_read_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_read_symbol_missing={}", dlerror_text()));
        }
        let sim_read_planets_stats_fn = unsafe { dlsym(handle, SIM_READ_PLANETS_STATS_SYMBOL.as_ptr() as *const c_char) };
        if sim_read_planets_stats_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_sim_read_planets_stats_symbol_missing={}", dlerror_text()));
        }
        let resident_model_decode_fn = unsafe { dlsym(handle, RESIDENT_MODEL_DECODE_SYMBOL.as_ptr() as *const c_char) };
        if resident_model_decode_fn.is_null() {
            unsafe { dlclose(handle) };
            return Err(format!("cuda_resident_model_decode_symbol_missing={}", dlerror_text()));
        }
        Ok(Self {
            library_handle: handle,
            status_fn: unsafe { std::mem::transmute::<*mut c_void, V8StatusFn>(status_fn) },
            forward_fn: unsafe { std::mem::transmute::<*mut c_void, V8ForwardFn>(forward_fn) },
            model_create_fn: unsafe { std::mem::transmute::<*mut c_void, V8ModelCreateFn>(model_create_fn) },
            model_destroy_fn: unsafe { std::mem::transmute::<*mut c_void, V8ModelDestroyFn>(model_destroy_fn) },
            forward_cached_fn: unsafe { std::mem::transmute::<*mut c_void, V8ForwardCachedFn>(forward_cached_fn) },
            sim_step_fn: unsafe { std::mem::transmute::<*mut c_void, SimStepFn>(sim_step_fn) },
            sim_create_fn: unsafe { std::mem::transmute::<*mut c_void, SimCreateFn>(sim_create_fn) },
            sim_destroy_fn: unsafe { std::mem::transmute::<*mut c_void, SimDestroyFn>(sim_destroy_fn) },
            sim_load_fn: unsafe { std::mem::transmute::<*mut c_void, SimLoadFn>(sim_load_fn) },
            sim_load_with_angular_fn: unsafe { std::mem::transmute::<*mut c_void, SimLoadWithAngularFn>(sim_load_with_angular_fn) },
            sim_step_persistent_fn: unsafe { std::mem::transmute::<*mut c_void, SimStepPersistentFn>(sim_step_persistent_fn) },
            sim_clear_actions_fn: unsafe { std::mem::transmute::<*mut c_void, SimClearActionsFn>(sim_clear_actions_fn) },
            sim_step_device_actions_fn: unsafe { std::mem::transmute::<*mut c_void, SimStepDeviceActionsFn>(sim_step_device_actions_fn) },
            sim_read_fn: unsafe { std::mem::transmute::<*mut c_void, SimReadFn>(sim_read_fn) },
            sim_read_planets_stats_fn: unsafe { std::mem::transmute::<*mut c_void, SimReadPlanetsStatsFn>(sim_read_planets_stats_fn) },
            resident_model_decode_fn: unsafe { std::mem::transmute::<*mut c_void, ResidentModelDecodeFn>(resident_model_decode_fn) },
        })
    }

    pub fn status(&self) -> Result<(), String> {
        let status = unsafe { (self.status_fn)() };
        require_ok(status, "cuda_v8_status")
    }

    pub fn create_model<'a>(&'a self, model: &V8Model) -> Result<V8CudaModel<'a>, String> {
        let (names, tensors) = ffi_tensors(model)?;
        let mut pointer: *mut c_void = std::ptr::null_mut();
        let status = unsafe { (self.model_create_fn)(tensors.as_ptr(), tensors.len(), &mut pointer) };
        require_ok(status, "cuda_v8_model_create")?;
        if pointer.is_null() {
            return Err("cuda_v8_model_create_returned_null".to_string());
        }
        drop(names);
        Ok(V8CudaModel {
            cuda: self,
            pointer,
        })
    }

    pub fn sim_step(
        &self,
        planets: &mut [OrbitWarsCudaPlanet],
        initial_planets: &[OrbitWarsCudaPlanet],
        fleets: &mut [OrbitWarsCudaFleet],
        next_fleet_ids: &mut [i32],
        actions: &[OrbitWarsCudaAction],
        action_counts: &[i32],
        stats: &mut [OrbitWarsCudaSimStats],
        config: OrbitWarsCudaSimConfig,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.sim_step_fn)(
                planets.as_mut_ptr(),
                initial_planets.as_ptr(),
                fleets.as_mut_ptr(),
                next_fleet_ids.as_mut_ptr(),
                actions.as_ptr(),
                action_counts.as_ptr(),
                stats.as_mut_ptr(),
                config,
            )
        };
        require_ok(status, "cuda_sim_step")
    }

    pub fn create_sim_state(
        &self,
        config: OrbitWarsCudaSimConfig,
    ) -> Result<CudaSimState<'_>, String> {
        let mut pointer: *mut c_void = std::ptr::null_mut();
        let status = unsafe { (self.sim_create_fn)(config, &mut pointer) };
        require_ok(status, "cuda_sim_create")?;
        if pointer.is_null() {
            return Err("cuda_sim_create_returned_null".to_string());
        }
        Ok(CudaSimState {
            cuda: self,
            pointer,
        })
    }

    #[allow(dead_code)]
    pub fn forward_tokens(
        &self,
        model: &V8Model,
        batch: &[V8Tokens],
    ) -> Result<Vec<Vec<ActionSlotOutput>>, String> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let config = model.config();
        let token_count = batch
            .iter()
            .map(|tokens| tokens.tokens.len())
            .max()
            .ok_or_else(|| "empty_cuda_batch".to_string())?;
        let batch_count = batch.len();
        let mut flat_tokens = vec![0.0f32; batch_count * token_count * TOKEN_FEATURES];
        let mut token_type_ids = vec![0i64; batch_count * token_count];
        let mut owner_ids = vec![0i64; batch_count * token_count];
        let mut padding_mask = vec![1u8; batch_count * token_count];
        let mut planet_mask = vec![0u8; batch_count * PLANETS];
        for (batch_index, tokens) in batch.iter().enumerate() {
            for row in 0..tokens.tokens.len() {
                let row_offset = (batch_index * token_count + row) * TOKEN_FEATURES;
                flat_tokens[row_offset..row_offset + TOKEN_FEATURES].copy_from_slice(&tokens.tokens[row]);
                token_type_ids[batch_index * token_count + row] = tokens.token_type_ids[row] as i64;
                owner_ids[batch_index * token_count + row] = tokens.owner_ids[row] as i64;
                padding_mask[batch_index * token_count + row] = tokens.padding_mask[row] as u8;
            }
            for planet in 0..PLANETS {
                planet_mask[batch_index * PLANETS + planet] = tokens.planet_mask[planet] as u8;
            }
        }
        let mut names = Vec::new();
        let mut tensors = Vec::new();
        for (name, data) in model.tensor_items() {
            let name = CString::new(name).map_err(|_| "cuda_tensor_name_contains_nul".to_string())?;
            tensors.push(OrbitWarsV8CudaTensor {
                name: name.as_ptr(),
                data: data.as_ptr(),
                len: data.len(),
            });
            names.push(name);
        }
        let mut fire_logits = vec![0.0f32; batch_count * ACTION_SLOTS];
        let mut source_logits = vec![f32::NEG_INFINITY; batch_count * ACTION_SLOTS * PLANETS];
        let mut target_logits = vec![f32::NEG_INFINITY; batch_count * ACTION_SLOTS * PLANETS];
        let mut amount_logits = vec![f32::NEG_INFINITY; batch_count * ACTION_SLOTS * AMOUNTS];
        let shape = OrbitWarsV8CudaShape {
            batch_count,
            token_count,
            token_features: TOKEN_FEATURES,
            d_model: config.d_model,
            head_count: config.heads,
            encoder_layer_count: config.encoder_layers,
            decoder_layer_count: config.decoder_layers,
            action_slot_count: config.action_slots,
            planet_count: config.max_planets,
            amount_class_count: config.amount_classes,
        };
        let status = unsafe {
            (self.forward_fn)(
                flat_tokens.as_ptr(),
                token_type_ids.as_ptr(),
                owner_ids.as_ptr(),
                padding_mask.as_ptr(),
                planet_mask.as_ptr(),
                tensors.as_ptr(),
                tensors.len(),
                fire_logits.as_mut_ptr(),
                source_logits.as_mut_ptr(),
                target_logits.as_mut_ptr(),
                amount_logits.as_mut_ptr(),
                shape,
            )
        };
        require_ok(status, "cuda_v8_forward")?;
        let mut output = Vec::with_capacity(batch_count);
        for batch_index in 0..batch_count {
            let mut slots = Vec::with_capacity(ACTION_SLOTS);
            for slot in 0..ACTION_SLOTS {
                let mut source = [f32::NEG_INFINITY; PLANETS];
                let mut target = [f32::NEG_INFINITY; PLANETS];
                let mut amount = [f32::NEG_INFINITY; AMOUNTS];
                let planet_offset = (batch_index * ACTION_SLOTS + slot) * PLANETS;
                source.copy_from_slice(&source_logits[planet_offset..planet_offset + PLANETS]);
                target.copy_from_slice(&target_logits[planet_offset..planet_offset + PLANETS]);
                let amount_offset = (batch_index * ACTION_SLOTS + slot) * AMOUNTS;
                amount.copy_from_slice(&amount_logits[amount_offset..amount_offset + AMOUNTS]);
                slots.push(ActionSlotOutput {
                    fire_logit: fire_logits[batch_index * ACTION_SLOTS + slot],
                    source_logits: source,
                    target_logits: target,
                    amount_logits: amount,
                });
            }
            output.push(slots);
        }
        Ok(output)
    }
}

pub struct V8CudaModel<'a> {
    cuda: &'a V8Cuda,
    pointer: *mut c_void,
}

pub struct CudaSimState<'a> {
    cuda: &'a V8Cuda,
    pointer: *mut c_void,
}

impl CudaSimState<'_> {
    pub fn load(
        &self,
        planets: &[OrbitWarsCudaPlanet],
        initial_planets: &[OrbitWarsCudaPlanet],
        fleets: &[OrbitWarsCudaFleet],
        next_fleet_ids: &[i32],
    ) -> Result<(), String> {
        let status = unsafe {
            (self.cuda.sim_load_fn)(
                self.pointer,
                planets.as_ptr(),
                initial_planets.as_ptr(),
                fleets.as_ptr(),
                next_fleet_ids.as_ptr(),
            )
        };
        require_ok(status, "cuda_sim_load")
    }

    pub fn load_with_angular_velocities(
        &self,
        planets: &[OrbitWarsCudaPlanet],
        initial_planets: &[OrbitWarsCudaPlanet],
        fleets: &[OrbitWarsCudaFleet],
        next_fleet_ids: &[i32],
        angular_velocities: &[f32],
    ) -> Result<(), String> {
        let status = unsafe {
            (self.cuda.sim_load_with_angular_fn)(
                self.pointer,
                planets.as_ptr(),
                initial_planets.as_ptr(),
                fleets.as_ptr(),
                next_fleet_ids.as_ptr(),
                angular_velocities.as_ptr(),
            )
        };
        require_ok(status, "cuda_sim_load_with_angular")
    }

    pub fn step(
        &self,
        actions: &[OrbitWarsCudaAction],
        action_counts: &[i32],
        step: usize,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.cuda.sim_step_persistent_fn)(
                self.pointer,
                actions.as_ptr(),
                action_counts.as_ptr(),
                step as i32,
            )
        };
        require_ok(status, "cuda_sim_step_persistent")
    }

    pub fn clear_actions(&self) -> Result<(), String> {
        let status = unsafe { (self.cuda.sim_clear_actions_fn)(self.pointer) };
        require_ok(status, "cuda_sim_clear_actions")
    }

    pub fn step_device_actions(&self, step: usize) -> Result<(), String> {
        let status = unsafe { (self.cuda.sim_step_device_actions_fn)(self.pointer, step as i32) };
        require_ok(status, "cuda_sim_step_device_actions")
    }

    pub fn read(
        &self,
        planets: &mut [OrbitWarsCudaPlanet],
        fleets: &mut [OrbitWarsCudaFleet],
        next_fleet_ids: &mut [i32],
        stats: &mut [OrbitWarsCudaSimStats],
    ) -> Result<(), String> {
        let status = unsafe {
            (self.cuda.sim_read_fn)(
                self.pointer,
                planets.as_mut_ptr(),
                fleets.as_mut_ptr(),
                next_fleet_ids.as_mut_ptr(),
                stats.as_mut_ptr(),
            )
        };
        require_ok(status, "cuda_sim_read")
    }

    pub fn read_planets_stats(
        &self,
        planets: &mut [OrbitWarsCudaPlanet],
        next_fleet_ids: &mut [i32],
        stats: &mut [OrbitWarsCudaSimStats],
    ) -> Result<(), String> {
        let status = unsafe {
            (self.cuda.sim_read_planets_stats_fn)(
                self.pointer,
                planets.as_mut_ptr(),
                next_fleet_ids.as_mut_ptr(),
                stats.as_mut_ptr(),
            )
        };
        require_ok(status, "cuda_sim_read_planets_stats")
    }
}

impl Drop for CudaSimState<'_> {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { (self.cuda.sim_destroy_fn)(self.pointer) };
            self.pointer = std::ptr::null_mut();
        }
    }
}

impl V8CudaModel<'_> {
    pub fn resident_decode(
        &self,
        sim_state: &CudaSimState<'_>,
        request_game_indices: &[i32],
        request_player_ids: &[i32],
        step: usize,
    ) -> Result<(), String> {
        if request_game_indices.len() != request_player_ids.len() {
            return Err(format!(
                "resident_decode_request_len_mismatch={}!={}",
                request_game_indices.len(),
                request_player_ids.len()
            ));
        }
        if request_game_indices.is_empty() {
            return Ok(());
        }
        let status = unsafe {
            (self.cuda.resident_model_decode_fn)(
                self.pointer,
                sim_state.pointer,
                request_game_indices.as_ptr(),
                request_player_ids.as_ptr(),
                request_game_indices.len(),
                step as i32,
            )
        };
        require_ok(status, "cuda_resident_model_decode")
    }

    pub fn forward_tokens(
        &self,
        model: &V8Model,
        batch: &[V8Tokens],
    ) -> Result<Vec<Vec<ActionSlotOutput>>, String> {
        let PreparedBatch {
            flat_tokens,
            token_type_ids,
            owner_ids,
            padding_mask,
            planet_mask,
            shape,
            batch_count,
        } = prepare_batch(model, batch)?;
        let mut fire_logits = vec![0.0f32; batch_count * ACTION_SLOTS];
        let mut source_logits = vec![f32::NEG_INFINITY; batch_count * ACTION_SLOTS * PLANETS];
        let mut target_logits = vec![f32::NEG_INFINITY; batch_count * ACTION_SLOTS * PLANETS];
        let mut amount_logits = vec![f32::NEG_INFINITY; batch_count * ACTION_SLOTS * AMOUNTS];
        let status = unsafe {
            (self.cuda.forward_cached_fn)(
                self.pointer,
                flat_tokens.as_ptr(),
                token_type_ids.as_ptr(),
                owner_ids.as_ptr(),
                padding_mask.as_ptr(),
                planet_mask.as_ptr(),
                fire_logits.as_mut_ptr(),
                source_logits.as_mut_ptr(),
                target_logits.as_mut_ptr(),
                amount_logits.as_mut_ptr(),
                shape,
            )
        };
        require_ok(status, "cuda_v8_forward_cached")?;
        unpack_output(batch_count, &fire_logits, &source_logits, &target_logits, &amount_logits)
    }
}

impl Drop for V8CudaModel<'_> {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { (self.cuda.model_destroy_fn)(self.pointer) };
            self.pointer = std::ptr::null_mut();
        }
    }
}

impl Drop for V8Cuda {
    fn drop(&mut self) {
        if !self.library_handle.is_null() {
            unsafe { dlclose(self.library_handle) };
            self.library_handle = std::ptr::null_mut();
        }
    }
}

struct PreparedBatch {
    flat_tokens: Vec<f32>,
    token_type_ids: Vec<i64>,
    owner_ids: Vec<i64>,
    padding_mask: Vec<u8>,
    planet_mask: Vec<u8>,
    shape: OrbitWarsV8CudaShape,
    batch_count: usize,
}

fn ffi_tensors(model: &V8Model) -> Result<(Vec<CString>, Vec<OrbitWarsV8CudaTensor>), String> {
    let mut names = Vec::new();
    let mut tensors = Vec::new();
    for (name, data) in model.tensor_items() {
        let name = CString::new(name).map_err(|_| "cuda_tensor_name_contains_nul".to_string())?;
        tensors.push(OrbitWarsV8CudaTensor {
            name: name.as_ptr(),
            data: data.as_ptr(),
            len: data.len(),
        });
        names.push(name);
    }
    Ok((names, tensors))
}

fn prepare_batch(model: &V8Model, batch: &[V8Tokens]) -> Result<PreparedBatch, String> {
    if batch.is_empty() {
        return Err("empty_cuda_batch".to_string());
    }
    let config = model.config();
    let token_count = batch
        .iter()
        .map(|tokens| tokens.tokens.len())
        .max()
        .ok_or_else(|| "empty_cuda_batch".to_string())?;
    let batch_count = batch.len();
    let mut flat_tokens = vec![0.0f32; batch_count * token_count * TOKEN_FEATURES];
    let mut token_type_ids = vec![0i64; batch_count * token_count];
    let mut owner_ids = vec![0i64; batch_count * token_count];
    let mut padding_mask = vec![1u8; batch_count * token_count];
    let mut planet_mask = vec![0u8; batch_count * PLANETS];
    for (batch_index, tokens) in batch.iter().enumerate() {
        for row in 0..tokens.tokens.len() {
            let row_offset = (batch_index * token_count + row) * TOKEN_FEATURES;
            flat_tokens[row_offset..row_offset + TOKEN_FEATURES].copy_from_slice(&tokens.tokens[row]);
            token_type_ids[batch_index * token_count + row] = tokens.token_type_ids[row] as i64;
            owner_ids[batch_index * token_count + row] = tokens.owner_ids[row] as i64;
            padding_mask[batch_index * token_count + row] = tokens.padding_mask[row] as u8;
        }
        for planet in 0..PLANETS {
            planet_mask[batch_index * PLANETS + planet] = tokens.planet_mask[planet] as u8;
        }
    }
    Ok(PreparedBatch {
        flat_tokens,
        token_type_ids,
        owner_ids,
        padding_mask,
        planet_mask,
        shape: OrbitWarsV8CudaShape {
            batch_count,
            token_count,
            token_features: TOKEN_FEATURES,
            d_model: config.d_model,
            head_count: config.heads,
            encoder_layer_count: config.encoder_layers,
            decoder_layer_count: config.decoder_layers,
            action_slot_count: config.action_slots,
            planet_count: config.max_planets,
            amount_class_count: config.amount_classes,
        },
        batch_count,
    })
}

fn unpack_output(
    batch_count: usize,
    fire_logits: &[f32],
    source_logits: &[f32],
    target_logits: &[f32],
    amount_logits: &[f32],
) -> Result<Vec<Vec<ActionSlotOutput>>, String> {
    let mut output = Vec::with_capacity(batch_count);
    for batch_index in 0..batch_count {
        let mut slots = Vec::with_capacity(ACTION_SLOTS);
        for slot in 0..ACTION_SLOTS {
            let mut source = [f32::NEG_INFINITY; PLANETS];
            let mut target = [f32::NEG_INFINITY; PLANETS];
            let mut amount = [f32::NEG_INFINITY; AMOUNTS];
            let planet_offset = (batch_index * ACTION_SLOTS + slot) * PLANETS;
            source.copy_from_slice(&source_logits[planet_offset..planet_offset + PLANETS]);
            target.copy_from_slice(&target_logits[planet_offset..planet_offset + PLANETS]);
            let amount_offset = (batch_index * ACTION_SLOTS + slot) * AMOUNTS;
            amount.copy_from_slice(&amount_logits[amount_offset..amount_offset + AMOUNTS]);
            slots.push(ActionSlotOutput {
                fire_logit: fire_logits[batch_index * ACTION_SLOTS + slot],
                source_logits: source,
                target_logits: target,
                amount_logits: amount,
            });
        }
        output.push(slots);
    }
    Ok(output)
}

fn require_ok(status: OrbitWarsV8CudaStatus, label: &str) -> Result<(), String> {
    if status.code == CUDA_STATUS_OK {
        return Ok(());
    }
    let message = if status.message.is_null() {
        "null cuda status message".to_string()
    } else {
        unsafe { CStr::from_ptr(status.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(format!("{label}_failed; code={}; message={message}", status.code))
}

fn dlerror_text() -> String {
    let pointer = unsafe { dlerror() };
    if pointer.is_null() {
        return "unknown dlerror".to_string();
    }
    unsafe { CStr::from_ptr(pointer) }.to_string_lossy().into_owned()
}

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}
