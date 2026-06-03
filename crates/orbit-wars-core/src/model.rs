use crate::config::{AgentConfig, MODEL_D_MODEL_MAX, MODEL_D_MODEL_MIN};
use crate::true2d_model::{True2DTransformer, True2DTransformerShape};
use crate::types::{ActionOutput, RowFeature};

const MODEL_MAGIC: &str = "ORBIT_WARS_MODEL_V1";
const TRUE_2D_MODEL_MODE: &str = "true_2d_pair_matrix_transformer";
const TRUE_2D_BINARY_MODEL_MAGIC: &[u8] = b"ORBIT_WARS_TRUE2D_BIN_V1\n";
const DEFAULT_D_MODEL: usize = 32;
const DEFAULT_HEADS: usize = 4;
const DEFAULT_SEED: u64 = 17;
const INPUT_FEATURE_COUNT: usize = 2;
const OUTPUT_FEATURE_COUNT: usize = 2;
const U64_BYTE_COUNT: usize = std::mem::size_of::<u64>();
const F32_BYTE_COUNT: usize = std::mem::size_of::<f32>();
const TRUE_2D_BINARY_HEADER_U64_FIELDS: usize = 3;

#[derive(Clone, Debug)]
pub struct TransformerModel {
    inner: True2DTransformer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelError {
    EmptyModel,
    InvalidUtf8,
    InvalidMagic,
    InvalidParameter,
    InvalidWeightCount,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformerShape {
    pub layers: usize,
    pub d_model: usize,
    pub heads: usize,
}

#[derive(Clone, Debug)]
pub struct TrainableTransformer {
    pub shape: TransformerShape,
    pub weights: Vec<f32>,
}

impl TransformerModel {
    pub fn from_model_bytes(model_bytes: &[u8]) -> Result<Self, ModelError> {
        if model_bytes.is_empty() {
            return Err(ModelError::EmptyModel);
        }
        if model_bytes.starts_with(TRUE_2D_BINARY_MODEL_MAGIC) {
            return Self::from_binary_true2d_model_bytes(model_bytes);
        }
        let model_text = std::str::from_utf8(model_bytes).map_err(|_| ModelError::InvalidUtf8)?;
        let mut lines = model_text.lines();
        if lines.next().map(str::trim) != Some(MODEL_MAGIC) {
            return Err(ModelError::InvalidMagic);
        }

        let config = AgentConfig::default();
        let mut layers = config.default_model_layers;
        let mut d_model = DEFAULT_D_MODEL;
        let mut heads = DEFAULT_HEADS;
        let mut seed = DEFAULT_SEED;
        let mut mode_seen = false;

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Some((key, value)) = trimmed.split_once('=') else {
                return Err(ModelError::InvalidParameter);
            };
            match key.trim() {
                "layers" => layers = parse_usize(value)?,
                "d_model" => d_model = parse_usize(value)?,
                "heads" => heads = parse_usize(value)?,
                "seed" => seed = parse_u64(value)?,
                "mode" => {
                    if value.trim() != TRUE_2D_MODEL_MODE {
                        return Err(ModelError::InvalidParameter);
                    }
                    mode_seen = true;
                }
                _ => return Err(ModelError::InvalidParameter),
            }
        }

        if !mode_seen
            || layers == 0
            || heads == 0
            || d_model < MODEL_D_MODEL_MIN
            || d_model > MODEL_D_MODEL_MAX
            || d_model % heads != 0
        {
            return Err(ModelError::InvalidParameter);
        }

        let shape = True2DTransformerShape {
            layers,
            row_count: config.true2d_internal_rows,
        };
        Ok(Self {
            inner: True2DTransformer::seeded(shape, seed, &config)?,
        })
    }

    fn from_binary_true2d_model_bytes(model_bytes: &[u8]) -> Result<Self, ModelError> {
        let config = AgentConfig::default();
        let mut cursor = TRUE_2D_BINARY_MODEL_MAGIC.len();
        let layers = read_u64_as_usize(model_bytes, &mut cursor)?;
        let row_count = read_u64_as_usize(model_bytes, &mut cursor)?;
        let weight_count = read_u64_as_usize(model_bytes, &mut cursor)?;
        let shape = True2DTransformerShape { layers, row_count };
        let expected_weight_count = shape.parameter_count(&config)?;
        if weight_count != expected_weight_count {
            return Err(ModelError::InvalidWeightCount);
        }
        let expected_byte_count = cursor
            .checked_add(
                weight_count
                    .checked_mul(F32_BYTE_COUNT)
                    .ok_or(ModelError::InvalidWeightCount)?,
            )
            .ok_or(ModelError::InvalidWeightCount)?;
        if model_bytes.len() != expected_byte_count {
            return Err(ModelError::InvalidWeightCount);
        }

        let mut weights = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            weights.push(read_f32(model_bytes, &mut cursor)?);
        }
        Ok(Self {
            inner: True2DTransformer::from_weights(shape, weights, &config)?,
        })
    }

    pub fn run(
        &self,
        rows: &[RowFeature],
        config: &AgentConfig,
    ) -> Result<Vec<ActionOutput>, ModelError> {
        self.inner.forward(rows, config)
    }
}

pub fn true2d_model_binary_bytes(model: &True2DTransformer) -> Result<Vec<u8>, ModelError> {
    let config = AgentConfig::default();
    let expected_weight_count = model.shape.parameter_count(&config)?;
    if model.weights.len() != expected_weight_count
        || model.weights.iter().any(|value| !value.is_finite())
    {
        return Err(ModelError::InvalidWeightCount);
    }
    let mut bytes = Vec::with_capacity(
        TRUE_2D_BINARY_MODEL_MAGIC.len()
            + U64_BYTE_COUNT * TRUE_2D_BINARY_HEADER_U64_FIELDS
            + model.weights.len() * F32_BYTE_COUNT,
    );
    bytes.extend_from_slice(TRUE_2D_BINARY_MODEL_MAGIC);
    push_usize_as_u64(&mut bytes, model.shape.layers)?;
    push_usize_as_u64(&mut bytes, model.shape.row_count)?;
    push_usize_as_u64(&mut bytes, model.weights.len())?;
    for weight in &model.weights {
        bytes.extend_from_slice(&weight.to_le_bytes());
    }
    Ok(bytes)
}

impl TransformerShape {
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.layers == 0
            || self.heads == 0
            || self.d_model < MODEL_D_MODEL_MIN
            || self.d_model > MODEL_D_MODEL_MAX
            || self.d_model % self.heads != 0
        {
            return Err(ModelError::InvalidParameter);
        }
        Ok(())
    }

    pub fn parameter_count(&self) -> Result<usize, ModelError> {
        self.validate()?;
        let input_projection = INPUT_FEATURE_COUNT * self.d_model;
        let position_embedding = self.d_model;
        let layer_weights = self.layers * self.d_model;
        let output_projection = self.d_model * OUTPUT_FEATURE_COUNT;
        Ok(input_projection + position_embedding + layer_weights + output_projection)
    }
}

impl TrainableTransformer {
    pub fn seeded(shape: TransformerShape, seed: u64) -> Result<Self, ModelError> {
        let parameter_count = shape.parameter_count()?;
        let weights = (0..parameter_count)
            .map(|index| deterministic_weight(seed, 0, index) * 0.25)
            .collect();
        Ok(Self { shape, weights })
    }

    pub fn from_weights(shape: TransformerShape, weights: Vec<f32>) -> Result<Self, ModelError> {
        let expected = shape.parameter_count()?;
        if weights.len() != expected || weights.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::InvalidWeightCount);
        }
        Ok(Self { shape, weights })
    }

    pub fn forward(
        &self,
        rows: &[RowFeature],
        config: &AgentConfig,
    ) -> Result<Vec<ActionOutput>, ModelError> {
        if rows.len() != config.max_rows {
            return Err(ModelError::InvalidParameter);
        }
        if self.weights.len() != self.shape.parameter_count()? {
            return Err(ModelError::InvalidWeightCount);
        }

        let mut cursor = 0usize;
        let input_projection =
            &self.weights[cursor..cursor + INPUT_FEATURE_COUNT * self.shape.d_model];
        cursor += INPUT_FEATURE_COUNT * self.shape.d_model;
        let position_embedding = &self.weights[cursor..cursor + self.shape.d_model];
        cursor += self.shape.d_model;
        let layer_scales = &self.weights[cursor..cursor + self.shape.layers * self.shape.d_model];
        cursor += self.shape.layers * self.shape.d_model;
        let output_projection =
            &self.weights[cursor..cursor + self.shape.d_model * OUTPUT_FEATURE_COUNT];

        let mut hidden = vec![vec![0.0f32; self.shape.d_model]; config.max_rows];
        for (row_index, row) in rows.iter().enumerate() {
            let features = [row.owner_class, row.ship_log_percent];
            let position = row_index as f32 / config.max_rows as f32;
            for dim in 0..self.shape.d_model {
                let mut value = position * position_embedding[dim];
                for feature_index in 0..INPUT_FEATURE_COUNT {
                    value += features[feature_index]
                        * input_projection[feature_index * self.shape.d_model + dim];
                }
                hidden[row_index][dim] = value.tanh();
            }
        }

        let head_dim = self.shape.d_model / self.shape.heads;
        for layer_index in 0..self.shape.layers {
            let mut next = vec![vec![0.0f32; self.shape.d_model]; config.max_rows];
            for head_index in 0..self.shape.heads {
                let head_start = head_index * head_dim;
                for source_row in 0..config.max_rows {
                    let mut scores = vec![0.0f32; config.max_rows];
                    let mut max_score = f32::NEG_INFINITY;
                    for target_row in 0..config.max_rows {
                        let mut dot = 0.0f32;
                        for dim_offset in 0..head_dim {
                            let dim = head_start + dim_offset;
                            dot += hidden[source_row][dim] * hidden[target_row][dim];
                        }
                        let score = dot / (head_dim as f32).sqrt();
                        scores[target_row] = score;
                        max_score = max_score.max(score);
                    }
                    let mut denominator = 0.0f32;
                    for score in scores.iter_mut() {
                        *score = (*score - max_score).exp();
                        denominator += *score;
                    }
                    for target_row in 0..config.max_rows {
                        let attention = scores[target_row] / denominator;
                        for dim_offset in 0..head_dim {
                            let dim = head_start + dim_offset;
                            next[source_row][dim] += attention * hidden[target_row][dim];
                        }
                    }
                }
            }

            let scale_offset = layer_index * self.shape.d_model;
            for row_index in 0..config.max_rows {
                for dim in 0..self.shape.d_model {
                    hidden[row_index][dim] = (hidden[row_index][dim]
                        + next[row_index][dim] * layer_scales[scale_offset + dim])
                        .tanh();
                }
            }
        }

        let mut outputs = Vec::with_capacity(config.max_rows);
        for row_hidden in hidden.iter() {
            let target_logit =
                dense_projection(row_hidden, output_projection, 0, self.shape.d_model);
            let send_logit = dense_projection(row_hidden, output_projection, 1, self.shape.d_model);
            outputs.push(ActionOutput::repeated(
                sigmoid(target_logit),
                sigmoid(send_logit),
            ));
        }
        Ok(outputs)
    }

    pub fn mutate(&self, seed: u64, sigma: f32) -> Self {
        let weights = self
            .weights
            .iter()
            .enumerate()
            .map(|(index, weight)| {
                let noise = deterministic_weight(seed, 0, index) * sigma;
                *weight + noise
            })
            .collect();
        Self {
            shape: self.shape,
            weights,
        }
    }

    pub fn crossover(left: &Self, right: &Self, seed: u64) -> Result<Self, ModelError> {
        if left.shape != right.shape || left.weights.len() != right.weights.len() {
            return Err(ModelError::InvalidWeightCount);
        }
        let weights = left
            .weights
            .iter()
            .zip(right.weights.iter())
            .enumerate()
            .map(|(index, (left_weight, right_weight))| {
                if deterministic_weight(seed, 1, index) >= 0.0 {
                    *left_weight
                } else {
                    *right_weight
                }
            })
            .collect();
        Ok(Self {
            shape: left.shape,
            weights,
        })
    }
}

fn read_u64_as_usize(model_bytes: &[u8], cursor: &mut usize) -> Result<usize, ModelError> {
    let next_cursor = cursor
        .checked_add(U64_BYTE_COUNT)
        .ok_or(ModelError::InvalidParameter)?;
    let value_bytes = model_bytes
        .get(*cursor..next_cursor)
        .ok_or(ModelError::InvalidParameter)?;
    let value = u64::from_le_bytes(
        value_bytes
            .try_into()
            .map_err(|_| ModelError::InvalidParameter)?,
    );
    *cursor = next_cursor;
    usize::try_from(value).map_err(|_| ModelError::InvalidParameter)
}

fn read_f32(model_bytes: &[u8], cursor: &mut usize) -> Result<f32, ModelError> {
    let next_cursor = cursor
        .checked_add(F32_BYTE_COUNT)
        .ok_or(ModelError::InvalidWeightCount)?;
    let value_bytes = model_bytes
        .get(*cursor..next_cursor)
        .ok_or(ModelError::InvalidWeightCount)?;
    let value = f32::from_le_bytes(
        value_bytes
            .try_into()
            .map_err(|_| ModelError::InvalidWeightCount)?,
    );
    if !value.is_finite() {
        return Err(ModelError::InvalidWeightCount);
    }
    *cursor = next_cursor;
    Ok(value)
}

fn push_usize_as_u64(bytes: &mut Vec<u8>, value: usize) -> Result<(), ModelError> {
    let value = u64::try_from(value).map_err(|_| ModelError::InvalidParameter)?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn parse_usize(value: &str) -> Result<usize, ModelError> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|_| ModelError::InvalidParameter)
}

fn parse_u64(value: &str) -> Result<u64, ModelError> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|_| ModelError::InvalidParameter)
}

fn dense_projection(values: &[f32], weights: &[f32], output_index: usize, d_model: usize) -> f32 {
    let offset = output_index * d_model;
    values
        .iter()
        .enumerate()
        .map(|(dim, value)| *value * weights[offset + dim])
        .sum::<f32>()
        / (d_model as f32).sqrt()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn deterministic_weight(seed: u64, row: u64, column: usize) -> f32 {
    let mut x = seed
        ^ row.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (column as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let normalized = (x as f64 / u64::MAX as f64) as f32;
    normalized * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_parses_text_model() {
        let config = AgentConfig::default();
        let bytes = b"ORBIT_WARS_MODEL_V1\nmode=true_2d_pair_matrix_transformer\nlayers=61\nd_model=32\nheads=4\nseed=17\n";
        let model = TransformerModel::from_model_bytes(bytes).unwrap();
        assert_eq!(model.inner.shape.layers, config.default_model_layers);
        assert_eq!(model.inner.shape.row_count, config.max_rows);
    }

    #[test]
    fn model_config_parses_binary_true2d_weights() {
        let config = AgentConfig::default();
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.true2d_internal_rows,
        };
        let trained_model = True2DTransformer::seeded(shape, DEFAULT_SEED, &config)
            .unwrap()
            .mutate(19, config.default_mutation_sigma);
        let bytes = true2d_model_binary_bytes(&trained_model).unwrap();

        let parsed_model = TransformerModel::from_model_bytes(&bytes).unwrap();

        assert_eq!(parsed_model.inner.shape, shape);
        assert_eq!(parsed_model.inner.weights, trained_model.weights);
    }

    #[test]
    fn model_outputs_one_action_row_per_input_row() {
        let config = AgentConfig::default();
        let bytes = b"ORBIT_WARS_MODEL_V1\nmode=true_2d_pair_matrix_transformer\nlayers=61\nd_model=32\nheads=4\nseed=17\n";
        let model = TransformerModel::from_model_bytes(bytes).unwrap();
        let rows = vec![RowFeature::empty(&config); config.max_rows];
        let outputs = model.run(&rows, &config).unwrap();
        assert_eq!(outputs.len(), config.max_rows);
        assert!(outputs.iter().all(|output| output
            .targets
            .iter()
            .all(|target| target.target_fraction >= 0.0 && target.target_fraction <= 1.0)));
    }

    #[test]
    fn trainable_transformer_has_stable_parameter_count_and_outputs() {
        let config = AgentConfig::default();
        let shape = TransformerShape {
            layers: 2,
            d_model: 32,
            heads: 4,
        };
        let model = TrainableTransformer::seeded(shape, 19).unwrap();
        assert_eq!(model.weights.len(), shape.parameter_count().unwrap());
        let rows = vec![RowFeature::empty(&config); config.max_rows];
        let outputs = model.forward(&rows, &config).unwrap();
        assert_eq!(outputs.len(), config.max_rows);
    }

    #[test]
    fn trainable_transformer_mutation_and_crossover_preserve_shape() {
        let shape = TransformerShape {
            layers: 2,
            d_model: 32,
            heads: 4,
        };
        let left = TrainableTransformer::seeded(shape, 1).unwrap();
        let right = left.mutate(2, 0.1);
        let child = TrainableTransformer::crossover(&left, &right, 3).unwrap();
        assert_eq!(child.shape, shape);
        assert_eq!(child.weights.len(), left.weights.len());
    }
}
