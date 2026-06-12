use std::slice;

use orbit_wars_core::{decode_action_slots_at_step, ActionSlotOutput, AgentConfig, Fleet, Planet, V8Model};

const PLANET_STRIDE: usize = 9;
const FLEET_STRIDE: usize = 7;
const ACTION_STRIDE: usize = 3;
const ERROR_NULL_INPUT: i32 = -2;
const ERROR_OUTPUT_CAPACITY: i32 = -7;
const ERROR_BAD_MODEL: i32 = -9;

pub struct AgentRuntime {
    config: AgentConfig,
    model: Option<V8Model>,
}

#[no_mangle]
pub unsafe extern "C" fn agent_create(model_bytes: *const u8, model_len: usize) -> *mut AgentRuntime {
    if model_len > 0 && model_bytes.is_null() {
        return std::ptr::null_mut();
    }
    let model = if model_len > 0 {
        let bytes = slice::from_raw_parts(model_bytes, model_len);
        match V8Model::from_bytes(bytes) {
            Ok(model) => Some(model),
            Err(_) => return std::ptr::null_mut(),
        }
    } else {
        None
    };
    Box::into_raw(Box::new(AgentRuntime {
        config: AgentConfig::default(),
        model,
    }))
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
        ERROR_NULL_INPUT => b"null input pointer\0".as_ptr(),
        ERROR_OUTPUT_CAPACITY => b"output action buffer capacity exceeded\0".as_ptr(),
        ERROR_BAD_MODEL => b"invalid OWV8 model artifact\0".as_ptr(),
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
    agent_act_v8(
        handle,
        player,
        angular_velocity,
        current_step,
        planet_ptr,
        planet_count,
        initial_planet_ptr,
        initial_planet_count,
        std::ptr::null(),
        0,
        comet_id_ptr,
        comet_id_count,
        output_ptr,
        output_capacity,
    )
}

#[no_mangle]
pub unsafe extern "C" fn agent_act_v8(
    handle: *mut AgentRuntime,
    player: i32,
    angular_velocity: f64,
    _current_step: usize,
    planet_ptr: *const f64,
    planet_count: usize,
    initial_planet_ptr: *const f64,
    initial_planet_count: usize,
    fleet_ptr: *const f64,
    fleet_count: usize,
    comet_id_ptr: *const i32,
    comet_id_count: usize,
    output_ptr: *mut f64,
    output_capacity: usize,
) -> i32 {
    if handle.is_null()
        || (planet_count > 0 && planet_ptr.is_null())
        || (initial_planet_count > 0 && initial_planet_ptr.is_null())
        || (fleet_count > 0 && fleet_ptr.is_null())
        || (comet_id_count > 0 && comet_id_ptr.is_null())
        || (output_capacity > 0 && output_ptr.is_null())
    {
        return ERROR_NULL_INPUT;
    }

    let runtime = &*handle;
    let planets = read_planets(planet_ptr, planet_count);
    let initial_planets = read_planets(initial_planet_ptr, initial_planet_count);
    let fleets = read_fleets(fleet_ptr, fleet_count);
    let comet_ids = if comet_id_count == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(comet_id_ptr, comet_id_count).to_vec()
    };

    let slots = match runtime.model.as_ref() {
        Some(model) => match model.action_slots(
            player,
            _current_step,
            angular_velocity as f32,
            &planets,
            &fleets,
        ) {
            Ok(slots) => slots,
            Err(_) => return ERROR_BAD_MODEL,
        },
        None => heuristic_slots(&planets, player),
    };
    let commands = match decode_action_slots_at_step(
        &planets,
        if initial_planets.is_empty() {
            &planets
        } else {
            &initial_planets
        },
        &comet_ids,
        player,
        angular_velocity as f32,
        _current_step,
        &slots,
        &runtime.config,
    ) {
        Ok(commands) => commands,
        Err(_) => Vec::new(),
    };

    if commands.len() > output_capacity {
        return ERROR_OUTPUT_CAPACITY;
    }
    let output_slice = slice::from_raw_parts_mut(output_ptr, output_capacity * ACTION_STRIDE);
    for (index, command) in commands.iter().enumerate() {
        let offset = index * ACTION_STRIDE;
        output_slice[offset] = f64::from(command.from_planet_id);
        output_slice[offset + 1] = f64::from(command.direction_angle);
        output_slice[offset + 2] = f64::from(command.ship_count);
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

unsafe fn read_fleets(ptr: *const f64, count: usize) -> Vec<Fleet> {
    if count == 0 {
        return Vec::new();
    }
    let values = slice::from_raw_parts(ptr, count * FLEET_STRIDE);
    values
        .chunks_exact(FLEET_STRIDE)
        .map(|row| Fleet {
            id: row[0] as i32,
            owner: row[1] as i32,
            x: row[2] as f32,
            y: row[3] as f32,
            angle: row[4] as f32,
            from_planet_id: row[5] as i32,
            ships: row[6] as f32,
        })
        .collect()
}

fn heuristic_slots(
    planets: &[Planet],
    player: i32,
) -> Vec<ActionSlotOutput> {
    let mut slots = vec![empty_slot(); 8];
    let Some((source_row, source)) = planets
        .iter()
        .enumerate()
        .filter(|(_, planet)| planet.owner == player && planet.ships >= 1.0)
        .max_by(|(_, left), (_, right)| left.ships.total_cmp(&right.ships))
    else {
        return slots;
    };
    let Some((target_row, _target)) = planets
        .iter()
        .enumerate()
        .filter(|(_, planet)| planet.id != source.id && planet.owner != player)
        .min_by(|(_, left), (_, right)| {
            source
                .distance_squared_to(left)
                .total_cmp(&source.distance_squared_to(right))
        })
    else {
        return slots;
    };
    slots[0].fire_logit = 6.0;
    slots[0].source_logits[source_row] = 10.0;
    slots[0].target_logits[target_row] = 10.0;
    slots[0].amount_logits[0] = 10.0;
    slots
}

fn empty_slot() -> ActionSlotOutput {
    ActionSlotOutput {
        fire_logit: -8.0,
        source_logits: [f32::NEG_INFINITY; 64],
        target_logits: [f32::NEG_INFINITY; 64],
        amount_logits: [f32::NEG_INFINITY; 16],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_artifact_rejects_bad_magic() {
        assert!(V8Model::from_bytes(b"bad").is_err());
        let mut bytes = vec![0u8; 16];
        bytes[..4].copy_from_slice(b"NOPE");
        assert!(V8Model::from_bytes(&bytes).is_err());
    }

    #[test]
    fn model_artifact_accepts_crc32_payload() {
        let payload = br#"{"config":{"action_slots":8,"amount_classes":16,"d_model":16,"decoder_layers":1,"encoder_layers":1,"heads":4,"max_planets":64,"token_features":14},"schema":"OWV8","tensors":[{"bytes":4,"name":"x","offset":0,"shape":[1]}]}"#;
        let mut checksum_bytes = Vec::new();
        checksum_bytes.extend_from_slice(payload);
        checksum_bytes.extend_from_slice(&0.0f32.to_le_bytes());
        let checksum = crc32(&checksum_bytes);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"OWV8");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&checksum.to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        assert!(V8Model::from_bytes(&bytes).is_ok());
    }

    #[test]
    fn agent_act_v8_rejects_null_handle() {
        let mut output = [0.0; 3];
        let code = unsafe {
            agent_act_v8(
                std::ptr::null_mut(),
                0,
                0.03,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                output.as_mut_ptr(),
                1,
            )
        };
        assert_eq!(code, ERROR_NULL_INPUT);
    }

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = 0u32.wrapping_sub(crc & 1);
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }
}
