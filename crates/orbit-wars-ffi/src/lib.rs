use std::ptr;
use std::slice;

use orbit_wars_core::{
    decode_model_outputs, encode_state, AgentConfig, ModelError, Planet, TransformerModel,
};

const PLANET_STRIDE: usize = 9;
const ACTION_STRIDE: usize = 3;
const ERROR_NULL_HANDLE: i32 = -1;
const ERROR_NULL_INPUT: i32 = -2;
const ERROR_MODEL: i32 = -3;
const ERROR_ENCODE: i32 = -4;
const ERROR_INFERENCE: i32 = -5;
const ERROR_DECODE: i32 = -6;
const ERROR_OUTPUT_CAPACITY: i32 = -7;

pub struct AgentRuntime {
    config: AgentConfig,
    model: TransformerModel,
}

#[no_mangle]
pub unsafe extern "C" fn agent_create(
    model_bytes: *const u8,
    model_len: usize,
) -> *mut AgentRuntime {
    if model_len == 0 || model_bytes.is_null() {
        return ptr::null_mut();
    }

    let bytes = slice::from_raw_parts(model_bytes, model_len);
    let Ok(model) = TransformerModel::from_model_bytes(bytes) else {
        return ptr::null_mut();
    };

    let config = AgentConfig::default();
    if config.validate().is_err() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(AgentRuntime { config, model }))
}

#[no_mangle]
pub unsafe extern "C" fn agent_destroy(handle: *mut AgentRuntime) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn agent_last_error_text(code: i32) -> *const u8 {
    match code {
        ERROR_NULL_HANDLE => b"null native agent handle\0".as_ptr(),
        ERROR_NULL_INPUT => b"null input pointer\0".as_ptr(),
        ERROR_MODEL => b"model contract error\0".as_ptr(),
        ERROR_ENCODE => b"observation encoding contract error\0".as_ptr(),
        ERROR_INFERENCE => b"native inference error\0".as_ptr(),
        ERROR_DECODE => b"action decoder error\0".as_ptr(),
        ERROR_OUTPUT_CAPACITY => b"output action buffer capacity exceeded\0".as_ptr(),
        _ => b"unknown native agent error\0".as_ptr(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn agent_act(
    handle: *mut AgentRuntime,
    player: i32,
    angular_velocity: f64,
    current_step: usize,
    planet_ptr: *const f64,
    planet_count: usize,
    initial_planet_ptr: *const f64,
    initial_planet_count: usize,
    comet_id_ptr: *const i32,
    comet_id_count: usize,
    output_ptr: *mut f64,
    output_capacity: usize,
) -> i32 {
    if handle.is_null() {
        return ERROR_NULL_HANDLE;
    }
    if (planet_count > 0 && planet_ptr.is_null())
        || (initial_planet_count > 0 && initial_planet_ptr.is_null())
        || (comet_id_count > 0 && comet_id_ptr.is_null())
        || (output_capacity > 0 && output_ptr.is_null())
    {
        return ERROR_NULL_INPUT;
    }

    let runtime = &mut *handle;
    let planets = read_planets(planet_ptr, planet_count);
    let initial_planets = read_planets(initial_planet_ptr, initial_planet_count);
    let comet_ids = if comet_id_count == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(comet_id_ptr, comet_id_count).to_vec()
    };

    let encoded = match encode_state(
        &planets,
        &initial_planets,
        &comet_ids,
        player,
        &runtime.config,
    ) {
        Ok(value) => value,
        Err(_) => return ERROR_ENCODE,
    };

    let outputs = match runtime.model.run(&encoded.rows, &runtime.config) {
        Ok(value) => value,
        Err(ModelError::InvalidParameter) => return ERROR_INFERENCE,
        Err(_) => return ERROR_MODEL,
    };

    let commands = match decode_model_outputs(
        &planets,
        &initial_planets,
        &comet_ids,
        player,
        angular_velocity as f32,
        current_step,
        &outputs,
        &runtime.config,
    ) {
        Ok(value) => value,
        Err(_) => return ERROR_DECODE,
    };

    if commands.len() > output_capacity {
        return ERROR_OUTPUT_CAPACITY;
    }

    let output_slice = slice::from_raw_parts_mut(output_ptr, output_capacity * ACTION_STRIDE);
    for (command_index, command) in commands.iter().enumerate() {
        let offset = command_index * ACTION_STRIDE;
        output_slice[offset] = command.from_planet_id as f64;
        output_slice[offset + 1] = command.direction_angle as f64;
        output_slice[offset + 2] = command.ship_count as f64;
    }

    commands.len() as i32
}

unsafe fn read_planets(ptr: *const f64, count: usize) -> Vec<Planet> {
    if count == 0 {
        return Vec::new();
    }
    let values = slice::from_raw_parts(ptr, count * PLANET_STRIDE);
    values
        .chunks_exact(PLANET_STRIDE)
        .map(|row| Planet {
            id: row[0] as i32,
            owner: row[1] as i32,
            x: row[2] as f32,
            y: row[3] as f32,
            radius: row[4] as f32,
            ships: row[5] as f32,
            production: row[6] as f32,
            velocity_x: row[7] as f32,
            velocity_y: row[8] as f32,
        })
        .collect()
}
