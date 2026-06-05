use crate::types::{ActionSlotOutput, Fleet, Planet};

const MODEL_MAGIC: &[u8] = b"OWV8";
const TOKEN_FEATURES: usize = 14;
const ACTION_SLOTS: usize = 8;
const MAX_PLANETS: usize = 64;
const MAX_FLEETS: usize = 640;
const OWNER_EMBEDDINGS: usize = 8;
const LAYER_NORM_EPS: f32 = 1.0e-5;

#[derive(Clone, Debug)]
pub struct V8Model {
    config: V8ModelConfig,
    tensors: Vec<Tensor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct V8ModelConfig {
    pub token_features: usize,
    pub d_model: usize,
    pub heads: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub action_slots: usize,
    pub amount_classes: usize,
    pub max_planets: usize,
}

#[derive(Clone, Debug)]
struct Tensor {
    name: String,
    data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct V8Tokens {
    pub tokens: Vec<[f32; TOKEN_FEATURES]>,
    pub token_type_ids: Vec<usize>,
    pub owner_ids: Vec<usize>,
    pub padding_mask: Vec<bool>,
    pub planet_mask: [bool; MAX_PLANETS],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V8ModelError {
    TooShort,
    BadMagic,
    Checksum,
    HeaderUtf8,
    HeaderParse,
    TensorMissing,
    TensorShape,
    InvalidConfig,
}

impl V8Model {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, V8ModelError> {
        if bytes.len() < 16 {
            return Err(V8ModelError::TooShort);
        }
        if &bytes[..4] != MODEL_MAGIC {
            return Err(V8ModelError::BadMagic);
        }
        let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let checksum = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let header_start = 16usize;
        let header_end = header_start
            .checked_add(header_len)
            .ok_or(V8ModelError::TooShort)?;
        if bytes.len() < header_end {
            return Err(V8ModelError::TooShort);
        }
        if crc32(&bytes[header_start..]) != checksum {
            return Err(V8ModelError::Checksum);
        }
        let header = std::str::from_utf8(&bytes[header_start..header_end])
            .map_err(|_| V8ModelError::HeaderUtf8)?;
        let payload = &bytes[header_end..];
        let config = parse_config(header)?;
        validate_config(config)?;
        let tensor_specs = parse_tensor_specs(header)?;
        let mut tensors = Vec::with_capacity(tensor_specs.len());
        for spec in tensor_specs {
            let end = spec
                .offset
                .checked_add(spec.bytes)
                .ok_or(V8ModelError::TensorShape)?;
            let raw = payload.get(spec.offset..end).ok_or(V8ModelError::TensorShape)?;
            if raw.len() % 4 != 0 {
                return Err(V8ModelError::TensorShape);
            }
            let mut data = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.chunks_exact(4) {
                let value = f32::from_le_bytes(chunk.try_into().unwrap());
                if !value.is_finite() {
                    return Err(V8ModelError::TensorShape);
                }
                data.push(value);
            }
            tensors.push(Tensor {
                name: spec.name,
                data,
            });
        }
        Ok(Self { config, tensors })
    }

    pub fn action_slots(
        &self,
        player: i32,
        step: usize,
        angular_velocity: f32,
        planets: &[Planet],
        fleets: &[Fleet],
    ) -> Result<Vec<ActionSlotOutput>, V8ModelError> {
        let token_rows = build_tokens(player, step, angular_velocity, planets, fleets);
        self.forward_tokens(&token_rows)
    }

    pub fn tokenize(
        &self,
        player: i32,
        step: usize,
        angular_velocity: f32,
        planets: &[Planet],
        fleets: &[Fleet],
    ) -> V8Tokens {
        build_tokens(player, step, angular_velocity, planets, fleets)
    }

    pub fn config(&self) -> V8ModelConfig {
        self.config
    }

    pub fn tensor_items(&self) -> impl Iterator<Item = (&str, &[f32])> {
        self.tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.data.as_slice()))
    }

    fn forward_tokens(&self, input: &V8Tokens) -> Result<Vec<ActionSlotOutput>, V8ModelError> {
        let d = self.config.d_model;
        let mut hidden = vec![vec![0.0; d]; input.tokens.len()];
        let token_projection_weight = self.tensor("token_projection.weight")?;
        let token_projection_bias = self.tensor("token_projection.bias")?;
        let type_embedding = self.tensor("type_embedding.weight")?;
        let owner_embedding = self.tensor("owner_embedding.weight")?;
        for row in 0..input.tokens.len() {
            for out in 0..d {
                let mut value = token_projection_bias[out];
                for feature in 0..self.config.token_features {
                    value += token_projection_weight[out * self.config.token_features + feature]
                        * input.tokens[row][feature];
                }
                value += type_embedding[input.token_type_ids[row] * d + out];
                value += owner_embedding[input.owner_ids[row] * d + out];
                hidden[row][out] = value;
            }
        }

        for layer in 0..self.config.encoder_layers {
            encoder_layer(self, layer, &mut hidden, &input.padding_mask)?;
        }

        let slot_queries = self.tensor("slot_queries")?;
        let mut slots = vec![vec![0.0; d]; self.config.action_slots];
        for slot in 0..self.config.action_slots {
            slots[slot].copy_from_slice(&slot_queries[slot * d..(slot + 1) * d]);
        }
        for layer in 0..self.config.decoder_layers {
            decoder_layer(self, layer, &mut slots, &hidden, &input.padding_mask)?;
        }

        let fire_weight = self.tensor("fire_head.weight")?;
        let fire_bias = self.tensor("fire_head.bias")?;
        let source_weight = self.tensor("source_head.weight")?;
        let source_bias = self.tensor("source_head.bias")?;
        let target_weight = self.tensor("target_head.weight")?;
        let target_bias = self.tensor("target_head.bias")?;
        let amount_weight = self.tensor("amount_head.weight")?;
        let amount_bias = self.tensor("amount_head.bias")?;
        let mut outputs = Vec::with_capacity(self.config.action_slots);
        for slot in slots.iter() {
            let mut source_logits = [f32::NEG_INFINITY; MAX_PLANETS];
            let mut target_logits = [f32::NEG_INFINITY; MAX_PLANETS];
            let mut amount_logits = [f32::NEG_INFINITY; 16];
            for row in 0..self.config.max_planets {
                if input.planet_mask[row] {
                    source_logits[row] = linear_row(slot, source_weight, source_bias, row, d);
                    target_logits[row] = linear_row(slot, target_weight, target_bias, row, d);
                }
            }
            for row in 0..self.config.amount_classes {
                amount_logits[row] = linear_row(slot, amount_weight, amount_bias, row, d);
            }
            outputs.push(ActionSlotOutput {
                fire_logit: linear_row(slot, fire_weight, fire_bias, 0, d),
                source_logits,
                target_logits,
                amount_logits,
            });
        }
        Ok(outputs)
    }

    fn tensor(&self, name: &str) -> Result<&[f32], V8ModelError> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .map(|tensor| tensor.data.as_slice())
            .ok_or(V8ModelError::TensorMissing)
    }
}

fn encoder_layer(
    model: &V8Model,
    layer: usize,
    hidden: &mut Vec<Vec<f32>>,
    padding_mask: &[bool],
) -> Result<(), V8ModelError> {
    let prefix = format!("encoder.layers.{layer}");
    let normed = layer_norm(
        hidden,
        model.tensor(&format!("{prefix}.norm1.weight"))?,
        model.tensor(&format!("{prefix}.norm1.bias"))?,
    );
    let attn = multi_head_attention(
        model,
        &normed,
        &normed,
        &format!("{prefix}.self_attn"),
        Some(padding_mask),
    )?;
    add_in_place(hidden, &attn);
    let normed = layer_norm(
        hidden,
        model.tensor(&format!("{prefix}.norm2.weight"))?,
        model.tensor(&format!("{prefix}.norm2.bias"))?,
    );
    let ff = feed_forward(model, &prefix, &normed)?;
    add_in_place(hidden, &ff);
    Ok(())
}

fn decoder_layer(
    model: &V8Model,
    layer: usize,
    slots: &mut Vec<Vec<f32>>,
    memory: &[Vec<f32>],
    memory_padding_mask: &[bool],
) -> Result<(), V8ModelError> {
    let prefix = format!("decoder.layers.{layer}");
    let normed = layer_norm(
        slots,
        model.tensor(&format!("{prefix}.norm1.weight"))?,
        model.tensor(&format!("{prefix}.norm1.bias"))?,
    );
    let self_attn = multi_head_attention(model, &normed, &normed, &format!("{prefix}.self_attn"), None)?;
    add_in_place(slots, &self_attn);
    let normed = layer_norm(
        slots,
        model.tensor(&format!("{prefix}.norm2.weight"))?,
        model.tensor(&format!("{prefix}.norm2.bias"))?,
    );
    let cross_attn = multi_head_attention(
        model,
        &normed,
        memory,
        &format!("{prefix}.multihead_attn"),
        Some(memory_padding_mask),
    )?;
    add_in_place(slots, &cross_attn);
    let normed = layer_norm(
        slots,
        model.tensor(&format!("{prefix}.norm3.weight"))?,
        model.tensor(&format!("{prefix}.norm3.bias"))?,
    );
    let ff = feed_forward(model, &prefix, &normed)?;
    add_in_place(slots, &ff);
    Ok(())
}

fn multi_head_attention(
    model: &V8Model,
    query: &[Vec<f32>],
    key_value: &[Vec<f32>],
    prefix: &str,
    key_padding_mask: Option<&[bool]>,
) -> Result<Vec<Vec<f32>>, V8ModelError> {
    let d = model.config.d_model;
    let heads = model.config.heads;
    let head_dim = d / heads;
    let in_proj_weight = model.tensor(&format!("{prefix}.in_proj_weight"))?;
    let in_proj_bias = model.tensor(&format!("{prefix}.in_proj_bias"))?;
    let out_proj_weight = model.tensor(&format!("{prefix}.out_proj.weight"))?;
    let out_proj_bias = model.tensor(&format!("{prefix}.out_proj.bias"))?;
    let q = project_qkv(query, in_proj_weight, in_proj_bias, 0, d);
    let k = project_qkv(key_value, in_proj_weight, in_proj_bias, d, d);
    let v = project_qkv(key_value, in_proj_weight, in_proj_bias, d * 2, d);
    let mut context = vec![vec![0.0; d]; query.len()];
    for target_index in 0..query.len() {
        for head in 0..heads {
            let dim_start = head * head_dim;
            let mut scores = vec![f32::NEG_INFINITY; key_value.len()];
            let mut max_score = f32::NEG_INFINITY;
            for source_index in 0..key_value.len() {
                if key_padding_mask
                    .and_then(|mask| mask.get(source_index))
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }
                let mut score = 0.0;
                for dim in 0..head_dim {
                    score += q[target_index][dim_start + dim] * k[source_index][dim_start + dim];
                }
                score /= (head_dim as f32).sqrt();
                scores[source_index] = score;
                max_score = max_score.max(score);
            }
            if !max_score.is_finite() {
                continue;
            }
            let mut denominator = 0.0;
            for score in scores.iter_mut() {
                if score.is_finite() {
                    *score = (*score - max_score).exp();
                    denominator += *score;
                }
            }
            if denominator <= 0.0 {
                continue;
            }
            for source_index in 0..key_value.len() {
                let attention = scores[source_index] / denominator;
                if !attention.is_finite() {
                    continue;
                }
                for dim in 0..head_dim {
                    context[target_index][dim_start + dim] +=
                        attention * v[source_index][dim_start + dim];
                }
            }
        }
    }
    let mut output = vec![vec![0.0; d]; query.len()];
    for row in 0..query.len() {
        for out in 0..d {
            output[row][out] = linear_row(&context[row], out_proj_weight, out_proj_bias, out, d);
        }
    }
    Ok(output)
}

fn project_qkv(input: &[Vec<f32>], weight: &[f32], bias: &[f32], offset: usize, d: usize) -> Vec<Vec<f32>> {
    let mut output = vec![vec![0.0; d]; input.len()];
    for row in 0..input.len() {
        for out in 0..d {
            let mut value = bias[offset + out];
            let weight_offset = (offset + out) * d;
            for dim in 0..d {
                value += weight[weight_offset + dim] * input[row][dim];
            }
            output[row][out] = value;
        }
    }
    output
}

fn feed_forward(model: &V8Model, prefix: &str, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, V8ModelError> {
    let d = model.config.d_model;
    let hidden_dim = d * 4;
    let linear1_weight = model.tensor(&format!("{prefix}.linear1.weight"))?;
    let linear1_bias = model.tensor(&format!("{prefix}.linear1.bias"))?;
    let linear2_weight = model.tensor(&format!("{prefix}.linear2.weight"))?;
    let linear2_bias = model.tensor(&format!("{prefix}.linear2.bias"))?;
    let mut middle = vec![vec![0.0; hidden_dim]; input.len()];
    for row in 0..input.len() {
        for out in 0..hidden_dim {
            middle[row][out] = gelu(linear_row(&input[row], linear1_weight, linear1_bias, out, d));
        }
    }
    let mut output = vec![vec![0.0; d]; input.len()];
    for row in 0..input.len() {
        for out in 0..d {
            output[row][out] = linear_row(&middle[row], linear2_weight, linear2_bias, out, hidden_dim);
        }
    }
    Ok(output)
}

fn linear_row(input: &[f32], weight: &[f32], bias: &[f32], row: usize, in_features: usize) -> f32 {
    let mut value = bias[row];
    let offset = row * in_features;
    for index in 0..in_features {
        value += weight[offset + index] * input[index];
    }
    value
}

fn layer_norm(input: &[Vec<f32>], weight: &[f32], bias: &[f32]) -> Vec<Vec<f32>> {
    let d = weight.len();
    let mut output = vec![vec![0.0; d]; input.len()];
    for row in 0..input.len() {
        let mean = input[row].iter().sum::<f32>() / d as f32;
        let variance = input[row]
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / d as f32;
        let scale = 1.0 / (variance + LAYER_NORM_EPS).sqrt();
        for dim in 0..d {
            output[row][dim] = (input[row][dim] - mean) * scale * weight[dim] + bias[dim];
        }
    }
    output
}

fn add_in_place(left: &mut [Vec<f32>], right: &[Vec<f32>]) {
    for row in 0..left.len() {
        for dim in 0..left[row].len() {
            left[row][dim] += right[row][dim];
        }
    }
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value * value * value)).tanh())
}

fn build_tokens(
    player: i32,
    step: usize,
    angular_velocity: f32,
    planets: &[Planet],
    fleets: &[Fleet],
) -> V8Tokens {
    let selected_fleets = select_fleets(fleets, planets, player, MAX_FLEETS);
    let mut tokens = Vec::<[f32; TOKEN_FEATURES]>::with_capacity(1 + planets.len() + selected_fleets.len());
    let mut token_type_ids = Vec::with_capacity(tokens.capacity());
    let mut owner_ids = Vec::with_capacity(tokens.capacity());
    let mut planet_mask = [false; MAX_PLANETS];

    tokens.push([
        player as f32 / 3.0,
        step as f32 / 500.0,
        angular_velocity,
        planets.len() as f32 / MAX_PLANETS as f32,
        selected_fleets.len() as f32 / MAX_FLEETS as f32,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    token_type_ids.push(0);
    owner_ids.push(owner_id(player));

    for (row, planet) in planets.iter().take(MAX_PLANETS).enumerate() {
        planet_mask[row] = true;
        tokens.push([
            planet.id as f32 / 128.0,
            planet.owner as f32 / 3.0,
            planet.x / 100.0,
            planet.y / 100.0,
            planet.radius / 10.0,
            (planet.ships.max(0.0)).ln_1p() / 10.0,
            planet.production / 20.0,
            planet.velocity_x / 10.0,
            planet.velocity_y / 10.0,
            0.0,
            0.0,
            -1.0,
            row as f32 / MAX_PLANETS as f32,
            0.0,
        ]);
        token_type_ids.push(1);
        owner_ids.push(owner_id(planet.owner));
    }

    for fleet in selected_fleets {
        let (target_row, progress) = infer_fleet_target_and_progress(fleet, planets);
        tokens.push([
            fleet.id as f32 / 2048.0,
            fleet.owner as f32 / 3.0,
            fleet.x / 100.0,
            fleet.y / 100.0,
            0.0,
            fleet.ships.max(0.0).ln_1p() / 10.0,
            0.0,
            0.0,
            0.0,
            fleet.angle.sin(),
            fleet.angle.cos(),
            fleet.from_planet_id as f32 / 128.0,
            target_row.map(|row| row as f32 / MAX_PLANETS as f32).unwrap_or(-1.0),
            progress,
        ]);
        token_type_ids.push(2);
        owner_ids.push(owner_id(fleet.owner));
    }
    let padding_mask = vec![false; tokens.len()];
    V8Tokens {
        tokens,
        token_type_ids,
        owner_ids,
        padding_mask,
        planet_mask,
    }
}

fn select_fleets<'a>(fleets: &'a [Fleet], planets: &[Planet], player: i32, max_fleets: usize) -> Vec<&'a Fleet> {
    if fleets.len() <= max_fleets {
        return fleets.iter().collect();
    }
    let owned = planets
        .iter()
        .filter(|planet| planet.owner == player)
        .copied()
        .collect::<Vec<_>>();
    let mut ranked = fleets.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        fleet_relevance(right, &owned, player).total_cmp(&fleet_relevance(left, &owned, player))
    });
    ranked.truncate(max_fleets);
    ranked
}

fn fleet_relevance(fleet: &Fleet, owned_planets: &[Planet], player: i32) -> f32 {
    let threat = if fleet.owner != player && !owned_planets.is_empty() {
        let nearest = owned_planets
            .iter()
            .map(|planet| {
                let dx = fleet.x - planet.x;
                let dy = fleet.y - planet.y;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(f32::INFINITY, f32::min);
        1.0 / (1.0 + nearest)
    } else {
        0.0
    };
    threat * 1.0e6 + fleet.ships
}

fn infer_fleet_target_and_progress(fleet: &Fleet, planets: &[Planet]) -> (Option<usize>, f32) {
    let direction_x = fleet.angle.cos();
    let direction_y = fleet.angle.sin();
    let mut best: Option<(f32, usize)> = None;
    for (row, planet) in planets.iter().take(MAX_PLANETS).enumerate() {
        let dx = planet.x - fleet.x;
        let dy = planet.y - fleet.y;
        let projection = dx * direction_x + dy * direction_y;
        if projection <= 0.0 {
            continue;
        }
        let perpendicular = (dx * direction_y - dy * direction_x).abs();
        let score = perpendicular + projection * 0.001;
        if best.map(|(best_score, _)| score < best_score).unwrap_or(true) {
            best = Some((score, row));
        }
    }
    let Some((_, target_row)) = best else {
        return (None, 0.0);
    };
    let Some(source) = planets.iter().find(|planet| planet.id == fleet.from_planet_id) else {
        return (Some(target_row), 0.0);
    };
    let target = planets[target_row];
    let total = distance(source.x, source.y, target.x, target.y);
    let remaining = distance(fleet.x, fleet.y, target.x, target.y);
    let progress = if total <= 0.0 {
        0.0
    } else {
        (1.0 - remaining / total).clamp(0.0, 1.0)
    };
    (Some(target_row), progress)
}

fn owner_id(owner: i32) -> usize {
    (owner + 1).clamp(0, (OWNER_EMBEDDINGS - 1) as i32) as usize
}

fn distance(left_x: f32, left_y: f32, right_x: f32, right_y: f32) -> f32 {
    let dx = right_x - left_x;
    let dy = right_y - left_y;
    (dx * dx + dy * dy).sqrt()
}

#[derive(Clone, Debug)]
struct TensorSpec {
    name: String,
    offset: usize,
    bytes: usize,
}

fn parse_config(header: &str) -> Result<V8ModelConfig, V8ModelError> {
    Ok(V8ModelConfig {
        token_features: parse_usize_field(header, "token_features")?,
        d_model: parse_usize_field(header, "d_model")?,
        heads: parse_usize_field(header, "heads")?,
        encoder_layers: parse_usize_field(header, "encoder_layers")?,
        decoder_layers: parse_usize_field(header, "decoder_layers")?,
        action_slots: parse_usize_field(header, "action_slots")?,
        amount_classes: parse_usize_field(header, "amount_classes")?,
        max_planets: parse_usize_field(header, "max_planets")?,
    })
}

fn validate_config(config: V8ModelConfig) -> Result<(), V8ModelError> {
    if config.token_features != TOKEN_FEATURES
        || config.d_model == 0
        || config.heads == 0
        || config.d_model % config.heads != 0
        || config.action_slots != ACTION_SLOTS
        || config.amount_classes > 16
        || config.max_planets != MAX_PLANETS
    {
        return Err(V8ModelError::InvalidConfig);
    }
    Ok(())
}

fn parse_usize_field(header: &str, key: &str) -> Result<usize, V8ModelError> {
    let needle = format!("\"{key}\":");
    let start = header.find(&needle).ok_or(V8ModelError::HeaderParse)? + needle.len();
    let tail = &header[start..];
    let end = tail
        .find(|ch: char| !ch.is_ascii_digit())
        .ok_or(V8ModelError::HeaderParse)?;
    tail[..end]
        .parse::<usize>()
        .map_err(|_| V8ModelError::HeaderParse)
}

fn parse_tensor_specs(header: &str) -> Result<Vec<TensorSpec>, V8ModelError> {
    let tensor_start = header.find("\"tensors\":[").ok_or(V8ModelError::HeaderParse)?;
    let mut cursor = tensor_start;
    let mut tensors = Vec::new();
    while let Some(name_pos) = header[cursor..].find("\"name\":\"") {
        let name_start = cursor + name_pos + "\"name\":\"".len();
        let object_start = header[..name_start]
            .rfind('{')
            .ok_or(V8ModelError::HeaderParse)?;
        let name_end = header[name_start..]
            .find('"')
            .map(|offset| name_start + offset)
            .ok_or(V8ModelError::HeaderParse)?;
        let name = header[name_start..name_end].to_string();
        let object_end = header[name_end..]
            .find('}')
            .map(|offset| name_end + offset)
            .ok_or(V8ModelError::HeaderParse)?;
        let object = &header[object_start..=object_end];
        let offset = parse_usize_field(object, "offset")?;
        let bytes = parse_usize_field(object, "bytes")?;
        tensors.push(TensorSpec { name, offset, bytes });
        cursor = object_end + 1;
    }
    if tensors.is_empty() {
        return Err(V8ModelError::HeaderParse);
    }
    Ok(tensors)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_config_reads_compact_json_fields() {
        let header = r#"{"config":{"action_slots":8,"amount_classes":16,"d_model":16,"decoder_layers":1,"encoder_layers":1,"heads":4,"max_planets":64,"token_features":14},"tensors":[{"bytes":4,"name":"x","offset":0,"shape":[1]}]}"#;
        let config = parse_config(header).unwrap();
        assert_eq!(config.d_model, 16);
        assert_eq!(config.heads, 4);
        let tensors = parse_tensor_specs(header).unwrap();
        assert_eq!(tensors[0].name, "x");
        assert_eq!(tensors[0].offset, 0);
        assert_eq!(tensors[0].bytes, 4);
    }
}
