use crate::config::{AgentConfig, TRUE2D_INPUT_FEATURES, TRUE2D_MATRIX_LAYER_PLANES};
use crate::model::ModelError;
use crate::types::{ActionOutput, ActionTargetOutput, RowFeature};

const INPUT_PAIR_FEATURE_WEIGHTS: usize = TRUE2D_INPUT_FEATURES * 2;
const LEGACY_FOUR_FEATURE_INPUT_PAIR_WEIGHTS: usize = 8;
const LEGACY_FIVE_FEATURE_INPUT_PAIR_WEIGHTS: usize = 10;
const EXPAND_BIAS_WEIGHTS: usize = 1;
const TRUE_2D_ATTENTION_HEAD_WEIGHTS: usize = 7;
const TRUE_2D_ATTENTION_SHARED_WEIGHTS: usize = 6;
const TRUE_2D_MATRIX_LAYER_CELL_WEIGHTS: usize = TRUE2D_MATRIX_LAYER_PLANES;
const OUTPUT_SCALAR_WEIGHTS: usize = 5;
const TARGET_OUTPUT_SCALE_INDEX: usize = 0;
const SEND_KEY_SCALE_INDEX: usize = 1;
const SEND_VALUE_SCALE_INDEX: usize = 2;
const SEND_OUTPUT_SCALE_INDEX: usize = 3;
const SEND_OUTPUT_BIAS_INDEX: usize = 4;
const SOURCE_OWNER_WEIGHT_INDEX: usize = 0;
const SOURCE_SHIP_WEIGHT_INDEX: usize = 1;
const SOURCE_X_WEIGHT_INDEX: usize = 2;
const SOURCE_Y_WEIGHT_INDEX: usize = 3;
const SOURCE_PRODUCTION_WEIGHT_INDEX: usize = 4;
const SOURCE_VELOCITY_X_WEIGHT_INDEX: usize = 5;
const SOURCE_VELOCITY_Y_WEIGHT_INDEX: usize = 6;
const TARGET_OWNER_WEIGHT_INDEX: usize = 7;
const TARGET_SHIP_WEIGHT_INDEX: usize = 8;
const TARGET_X_WEIGHT_INDEX: usize = 9;
const TARGET_Y_WEIGHT_INDEX: usize = 10;
const TARGET_PRODUCTION_WEIGHT_INDEX: usize = 11;
const TARGET_VELOCITY_X_WEIGHT_INDEX: usize = 12;
const TARGET_VELOCITY_Y_WEIGHT_INDEX: usize = 13;
const HEAD_QUERY_SCALE_INDEX: usize = 0;
const HEAD_QUERY_BIAS_INDEX: usize = 1;
const HEAD_KEY_SCALE_INDEX: usize = 2;
const HEAD_KEY_BIAS_INDEX: usize = 3;
const HEAD_VALUE_SCALE_INDEX: usize = 4;
const HEAD_VALUE_BIAS_INDEX: usize = 5;
const HEAD_OUTPUT_SCALE_INDEX: usize = 6;
const ATTENTION_RESIDUAL_SCALE_INDEX: usize = 0;
const ATTENTION_BIAS_INDEX: usize = 1;
const ATTENTION_FFN_INPUT_SCALE_INDEX: usize = 2;
const ATTENTION_FFN_INPUT_BIAS_INDEX: usize = 3;
const ATTENTION_FFN_OUTPUT_SCALE_INDEX: usize = 4;
const ATTENTION_FFN_OUTPUT_BIAS_INDEX: usize = 5;
const MATRIX_RESIDUAL_SCALE_INDEX: usize = 0;
const MATRIX_RESIDUAL_BIAS_INDEX: usize = 1;
const MATRIX_HIDDEN_A_SCALE_INDEX: usize = 2;
const MATRIX_HIDDEN_A_BIAS_INDEX: usize = 3;
const MATRIX_HIDDEN_B_CELL_SCALE_INDEX: usize = 4;
const MATRIX_HIDDEN_B_HIDDEN_SCALE_INDEX: usize = 5;
const MATRIX_HIDDEN_B_BIAS_INDEX: usize = 6;
const MATRIX_GATE_SCALE_INDEX: usize = 7;
const MATRIX_GATE_BIAS_INDEX: usize = 8;
const MATRIX_UPDATE_HIDDEN_B_SCALE_INDEX: usize = 9;
const MATRIX_UPDATE_HIDDEN_A_SCALE_INDEX: usize = 10;
const MATRIX_UPDATE_BIAS_INDEX: usize = 11;
const HASH_ROW_FACTOR: u64 = 0x9E37_79B9_7F4A_7C15;
const HASH_COLUMN_FACTOR: u64 = 0xBF58_476D_1CE4_E5B9;
const HASH_MIX_FACTOR: u64 = 0x94D0_49BB_1331_11EB;
const INITIAL_WEIGHT_SCALE: f32 = 0.02;
const PROBABILITY_MIN: f32 = 0.0;
const PROBABILITY_MAX: f32 = 1.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct True2DTransformerShape {
    pub layers: usize,
    pub row_count: usize,
}

#[derive(Clone, Debug)]
pub struct PairMatrix2D {
    pub row_count: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct True2DForwardTrace {
    pub expanded_shape: (usize, usize),
    pub attention_shapes: Vec<(usize, usize)>,
    pub layer_shapes: Vec<(usize, usize)>,
    pub outputs: Vec<ActionOutput>,
}

#[derive(Clone, Debug)]
pub struct True2DTransformer {
    pub shape: True2DTransformerShape,
    pub weights: Vec<f32>,
}

impl PairMatrix2D {
    pub fn shape(&self) -> (usize, usize) {
        (self.row_count, self.row_count)
    }
}

impl True2DTransformerShape {
    pub fn validate(&self, config: &AgentConfig) -> Result<(), ModelError> {
        if self.layers == 0 || self.row_count != config.true2d_internal_rows || self.row_count == 0
        {
            return Err(ModelError::InvalidParameter);
        }
        Ok(())
    }

    pub fn cell_count(&self) -> Result<usize, ModelError> {
        self.row_count
            .checked_mul(self.row_count)
            .ok_or(ModelError::InvalidParameter)
    }

    pub fn parameter_count(&self, config: &AgentConfig) -> Result<usize, ModelError> {
        self.validate(config)?;
        let cell_count = self.cell_count()?;
        let matrix_layer_weight_count = true_2d_matrix_layer_weight_count(config, self.row_count)?;
        let attention_weight_count = true_2d_attention_weight_count(config)?;
        Ok(INPUT_PAIR_FEATURE_WEIGHTS
            + self.row_count
            + self.row_count
            + cell_count
            + EXPAND_BIAS_WEIGHTS
            + self.layers * matrix_layer_weight_count
            + attention_weight_count
            + self.row_count * config.true2d_action_targets_per_source
            + self.row_count * config.true2d_action_targets_per_source
            + OUTPUT_SCALAR_WEIGHTS * config.true2d_action_targets_per_source)
    }

    pub fn attention_layer_index(&self) -> usize {
        self.layers / 2
    }
}

fn true_2d_matrix_layer_weight_count(
    config: &AgentConfig,
    row_count: usize,
) -> Result<usize, ModelError> {
    if config.true2d_matrix_layer_planes != TRUE_2D_MATRIX_LAYER_CELL_WEIGHTS {
        return Err(ModelError::InvalidParameter);
    }
    let cell_count = row_count
        .checked_mul(row_count)
        .ok_or(ModelError::InvalidParameter)?;
    cell_count
        .checked_mul(config.true2d_matrix_layer_planes)
        .ok_or(ModelError::InvalidParameter)
}

fn true_2d_attention_weight_count(config: &AgentConfig) -> Result<usize, ModelError> {
    if config.true2d_attention_heads == 0 {
        return Err(ModelError::InvalidParameter);
    }
    Ok(
        config.true2d_attention_heads * TRUE_2D_ATTENTION_HEAD_WEIGHTS
            + TRUE_2D_ATTENTION_SHARED_WEIGHTS,
    )
}

impl True2DTransformer {
    pub fn seeded(
        shape: True2DTransformerShape,
        seed: u64,
        config: &AgentConfig,
    ) -> Result<Self, ModelError> {
        let parameter_count = shape.parameter_count(config)?;
        let weights = (0..parameter_count)
            .map(|index| deterministic_weight(seed, 0, index) * INITIAL_WEIGHT_SCALE)
            .collect();
        Ok(Self { shape, weights })
    }

    pub fn from_weights(
        shape: True2DTransformerShape,
        weights: Vec<f32>,
        config: &AgentConfig,
    ) -> Result<Self, ModelError> {
        let expected = shape.parameter_count(config)?;
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::InvalidWeightCount);
        }
        let weights = if weights.len() == expected {
            weights
        } else {
            migrate_legacy_input_pair_weights(shape, weights, expected, config)?
        };
        Ok(Self { shape, weights })
    }

    pub fn forward(
        &self,
        rows: &[RowFeature],
        config: &AgentConfig,
    ) -> Result<Vec<ActionOutput>, ModelError> {
        let trace = self.forward_with_trace(rows, config)?;
        Ok(trace.outputs)
    }

    pub fn forward_with_trace(
        &self,
        rows: &[RowFeature],
        config: &AgentConfig,
    ) -> Result<True2DForwardTrace, ModelError> {
        if rows.len() != self.shape.row_count || rows.len() != config.max_rows {
            return Err(ModelError::InvalidParameter);
        }
        if self.weights.len() != self.shape.parameter_count(config)? {
            return Err(ModelError::InvalidWeightCount);
        }

        let mut cursor = 0usize;
        let input_pair_weights = &self.weights[cursor..cursor + INPUT_PAIR_FEATURE_WEIGHTS];
        cursor += INPUT_PAIR_FEATURE_WEIGHTS;
        let source_row_embeddings = &self.weights[cursor..cursor + self.shape.row_count];
        cursor += self.shape.row_count;
        let target_row_embeddings = &self.weights[cursor..cursor + self.shape.row_count];
        cursor += self.shape.row_count;
        let cell_count = self.shape.cell_count()?;
        let pair_embeddings = &self.weights[cursor..cursor + cell_count];
        cursor += cell_count;
        let expand_bias = self.weights[cursor];
        cursor += EXPAND_BIAS_WEIGHTS;
        let matrix_layer_weight_count =
            true_2d_matrix_layer_weight_count(config, self.shape.row_count)?;
        let matrix_layer_weights =
            &self.weights[cursor..cursor + self.shape.layers * matrix_layer_weight_count];
        cursor += self.shape.layers * matrix_layer_weight_count;
        let attention_weight_count = true_2d_attention_weight_count(config)?;
        let attention_weights = &self.weights[cursor..cursor + attention_weight_count];
        cursor += attention_weight_count;
        let output_bias_count = self.shape.row_count * config.true2d_action_targets_per_source;
        let target_output_bias = &self.weights[cursor..cursor + output_bias_count];
        cursor += output_bias_count;
        let send_output_bias = &self.weights[cursor..cursor + output_bias_count];
        cursor += output_bias_count;
        let output_scalar_weight_count =
            OUTPUT_SCALAR_WEIGHTS * config.true2d_action_targets_per_source;
        let output_scalar_weights = &self.weights[cursor..cursor + output_scalar_weight_count];

        let mut matrix = self.expand_to_pair_matrix_from_parts(
            rows,
            input_pair_weights,
            source_row_embeddings,
            target_row_embeddings,
            pair_embeddings,
            expand_bias,
        )?;
        let expanded_shape = matrix.shape();
        let mut layer_shapes = Vec::with_capacity(self.shape.layers);
        let mut attention_shapes = Vec::with_capacity(1);
        let attention_layer_index = self.shape.attention_layer_index();
        for layer_index in 0..attention_layer_index {
            let offset = layer_index * matrix_layer_weight_count;
            matrix = apply_matrix_layer(
                &matrix,
                &matrix_layer_weights[offset..offset + matrix_layer_weight_count],
                config.true2d_matrix_layer_planes,
            )?;
            layer_shapes.push(matrix.shape());
        }
        matrix = apply_attention_layer(&matrix, attention_weights, config.true2d_attention_heads)?;
        attention_shapes.push((cell_count, cell_count));
        for layer_index in attention_layer_index..self.shape.layers {
            let offset = layer_index * matrix_layer_weight_count;
            matrix = apply_matrix_layer(
                &matrix,
                &matrix_layer_weights[offset..offset + matrix_layer_weight_count],
                config.true2d_matrix_layer_planes,
            )?;
            layer_shapes.push(matrix.shape());
        }
        let outputs = collapse_pair_matrix(
            &matrix,
            target_output_bias,
            send_output_bias,
            output_scalar_weights,
            config.true2d_action_targets_per_source,
        )?;
        Ok(True2DForwardTrace {
            expanded_shape,
            attention_shapes,
            layer_shapes,
            outputs,
        })
    }

    pub fn mutate(&self, seed: u64, sigma: f32) -> Self {
        let weights = self
            .weights
            .iter()
            .enumerate()
            .map(|(index, weight)| *weight + deterministic_weight(seed, 0, index) * sigma)
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

    fn expand_to_pair_matrix_from_parts(
        &self,
        rows: &[RowFeature],
        input_pair_weights: &[f32],
        source_row_embeddings: &[f32],
        target_row_embeddings: &[f32],
        pair_embeddings: &[f32],
        expand_bias: f32,
    ) -> Result<PairMatrix2D, ModelError> {
        let row_count = self.shape.row_count;
        let cell_count = self.shape.cell_count()?;
        let mut values = vec![0.0f32; cell_count];
        for source_row in 0..row_count {
            for target_row in 0..row_count {
                let cell_index = source_row * row_count + target_row;
                let source = rows[source_row];
                let target = rows[target_row];
                let value = source.owner_class * input_pair_weights[SOURCE_OWNER_WEIGHT_INDEX]
                    + source.ship_log_percent * input_pair_weights[SOURCE_SHIP_WEIGHT_INDEX]
                    + source.x_position_normalized * input_pair_weights[SOURCE_X_WEIGHT_INDEX]
                    + source.y_position_normalized * input_pair_weights[SOURCE_Y_WEIGHT_INDEX]
                    + source.production_normalized
                        * input_pair_weights[SOURCE_PRODUCTION_WEIGHT_INDEX]
                    + source.velocity_x_normalized
                        * input_pair_weights[SOURCE_VELOCITY_X_WEIGHT_INDEX]
                    + source.velocity_y_normalized
                        * input_pair_weights[SOURCE_VELOCITY_Y_WEIGHT_INDEX]
                    + target.owner_class * input_pair_weights[TARGET_OWNER_WEIGHT_INDEX]
                    + target.ship_log_percent * input_pair_weights[TARGET_SHIP_WEIGHT_INDEX]
                    + target.x_position_normalized * input_pair_weights[TARGET_X_WEIGHT_INDEX]
                    + target.y_position_normalized * input_pair_weights[TARGET_Y_WEIGHT_INDEX]
                    + target.production_normalized
                        * input_pair_weights[TARGET_PRODUCTION_WEIGHT_INDEX]
                    + target.velocity_x_normalized
                        * input_pair_weights[TARGET_VELOCITY_X_WEIGHT_INDEX]
                    + target.velocity_y_normalized
                        * input_pair_weights[TARGET_VELOCITY_Y_WEIGHT_INDEX]
                    + source_row_embeddings[source_row]
                    + target_row_embeddings[target_row]
                    + pair_embeddings[cell_index]
                    + expand_bias;
                values[cell_index] = value.tanh();
            }
        }
        Ok(PairMatrix2D { row_count, values })
    }
}

fn migrate_legacy_input_pair_weights(
    shape: True2DTransformerShape,
    weights: Vec<f32>,
    expected: usize,
    config: &AgentConfig,
) -> Result<Vec<f32>, ModelError> {
    if TRUE2D_INPUT_FEATURES != 7 || INPUT_PAIR_FEATURE_WEIGHTS != 14 {
        return Err(ModelError::InvalidWeightCount);
    }
    shape.validate(config)?;

    let legacy_five_expected = expected
        .checked_sub(INPUT_PAIR_FEATURE_WEIGHTS - LEGACY_FIVE_FEATURE_INPUT_PAIR_WEIGHTS)
        .ok_or(ModelError::InvalidWeightCount)?;
    if weights.len() == legacy_five_expected {
        let mut migrated = Vec::with_capacity(expected);
        migrated.extend_from_slice(&weights[0..=SOURCE_PRODUCTION_WEIGHT_INDEX]);
        migrated.extend_from_slice(&[0.0, 0.0]);
        migrated.extend_from_slice(
            &weights[SOURCE_PRODUCTION_WEIGHT_INDEX + 1..LEGACY_FIVE_FEATURE_INPUT_PAIR_WEIGHTS],
        );
        migrated.extend_from_slice(&[0.0, 0.0]);
        migrated.extend_from_slice(&weights[LEGACY_FIVE_FEATURE_INPUT_PAIR_WEIGHTS..]);
        if migrated.len() != expected {
            return Err(ModelError::InvalidWeightCount);
        }
        return Ok(migrated);
    }

    let legacy_four_expected = expected
        .checked_sub(INPUT_PAIR_FEATURE_WEIGHTS - LEGACY_FOUR_FEATURE_INPUT_PAIR_WEIGHTS)
        .ok_or(ModelError::InvalidWeightCount)?;
    if weights.len() == legacy_four_expected {
        let mut migrated = Vec::with_capacity(expected);
        migrated.extend_from_slice(&weights[0..SOURCE_PRODUCTION_WEIGHT_INDEX]);
        migrated.extend_from_slice(&[0.0, 0.0, 0.0]);
        migrated.extend_from_slice(
            &weights[SOURCE_PRODUCTION_WEIGHT_INDEX..LEGACY_FOUR_FEATURE_INPUT_PAIR_WEIGHTS],
        );
        migrated.extend_from_slice(&[0.0, 0.0, 0.0]);
        migrated.extend_from_slice(&weights[LEGACY_FOUR_FEATURE_INPUT_PAIR_WEIGHTS..]);
        if migrated.len() != expected {
            return Err(ModelError::InvalidWeightCount);
        }
        return Ok(migrated);
    }

    Err(ModelError::InvalidWeightCount)
}

fn apply_matrix_layer(
    matrix: &PairMatrix2D,
    layer_weights: &[f32],
    cell_weight_count: usize,
) -> Result<PairMatrix2D, ModelError> {
    let row_count = matrix.row_count;
    let cell_count = row_count
        .checked_mul(row_count)
        .ok_or(ModelError::InvalidParameter)?;
    if matrix.values.len() != cell_count
        || cell_weight_count != TRUE_2D_MATRIX_LAYER_CELL_WEIGHTS
        || layer_weights.len() != cell_count * cell_weight_count
    {
        return Err(ModelError::InvalidParameter);
    }

    let mut next_values = vec![0.0f32; cell_count];
    for (cell_index, next_value) in next_values.iter_mut().enumerate() {
        let cell_value = matrix.values[cell_index];
        let weights =
            &layer_weights[cell_index * cell_weight_count..(cell_index + 1) * cell_weight_count];
        let hidden_a = (cell_value * weights[MATRIX_HIDDEN_A_SCALE_INDEX]
            + weights[MATRIX_HIDDEN_A_BIAS_INDEX])
            .tanh();
        let hidden_b = (cell_value * weights[MATRIX_HIDDEN_B_CELL_SCALE_INDEX]
            + hidden_a * weights[MATRIX_HIDDEN_B_HIDDEN_SCALE_INDEX]
            + weights[MATRIX_HIDDEN_B_BIAS_INDEX])
            .tanh();
        let gate = sigmoid(
            cell_value * weights[MATRIX_GATE_SCALE_INDEX] + weights[MATRIX_GATE_BIAS_INDEX],
        );
        let update = hidden_b * weights[MATRIX_UPDATE_HIDDEN_B_SCALE_INDEX]
            + hidden_a * weights[MATRIX_UPDATE_HIDDEN_A_SCALE_INDEX]
            + weights[MATRIX_UPDATE_BIAS_INDEX];
        let residual = cell_value * (1.0 + weights[MATRIX_RESIDUAL_SCALE_INDEX])
            + weights[MATRIX_RESIDUAL_BIAS_INDEX];
        *next_value = (residual + gate * update).tanh();
    }
    Ok(PairMatrix2D {
        row_count,
        values: next_values,
    })
}

fn apply_attention_layer(
    matrix: &PairMatrix2D,
    layer_weights: &[f32],
    attention_head_count: usize,
) -> Result<PairMatrix2D, ModelError> {
    let row_count = matrix.row_count;
    let cell_count = row_count
        .checked_mul(row_count)
        .ok_or(ModelError::InvalidParameter)?;
    let expected_layer_weights =
        attention_head_count * TRUE_2D_ATTENTION_HEAD_WEIGHTS + TRUE_2D_ATTENTION_SHARED_WEIGHTS;
    if matrix.values.len() != cell_count
        || layer_weights.len() != expected_layer_weights
        || attention_head_count == 0
    {
        return Err(ModelError::InvalidParameter);
    }

    let attention_values = full_self_attention_values(matrix, layer_weights, attention_head_count);
    let shared_offset = attention_head_count * TRUE_2D_ATTENTION_HEAD_WEIGHTS;
    let shared_weights =
        &layer_weights[shared_offset..shared_offset + TRUE_2D_ATTENTION_SHARED_WEIGHTS];
    let mut next_values = vec![0.0f32; cell_count];
    for cell_index in 0..cell_count {
        let cell_value = matrix.values[cell_index];
        let residual_attention = cell_value * shared_weights[ATTENTION_RESIDUAL_SCALE_INDEX]
            + attention_values[cell_index]
            + shared_weights[ATTENTION_BIAS_INDEX];
        let ffn_hidden = (residual_attention * shared_weights[ATTENTION_FFN_INPUT_SCALE_INDEX]
            + shared_weights[ATTENTION_FFN_INPUT_BIAS_INDEX])
            .tanh();
        let value = residual_attention
            + ffn_hidden * shared_weights[ATTENTION_FFN_OUTPUT_SCALE_INDEX]
            + shared_weights[ATTENTION_FFN_OUTPUT_BIAS_INDEX];
        next_values[cell_index] = value.tanh();
    }
    Ok(PairMatrix2D {
        row_count,
        values: next_values,
    })
}

fn full_self_attention_values(
    matrix: &PairMatrix2D,
    layer_weights: &[f32],
    attention_head_count: usize,
) -> Vec<f32> {
    let cell_count = matrix.values.len();
    let mut attention_values = vec![0.0f32; cell_count];
    for query_index in 0..cell_count {
        let query_cell = matrix.values[query_index];
        let mut combined_context = 0.0f32;
        for head_index in 0..attention_head_count {
            let head_offset = head_index * TRUE_2D_ATTENTION_HEAD_WEIGHTS;
            let head_weights =
                &layer_weights[head_offset..head_offset + TRUE_2D_ATTENTION_HEAD_WEIGHTS];
            let query = query_cell * head_weights[HEAD_QUERY_SCALE_INDEX]
                + head_weights[HEAD_QUERY_BIAS_INDEX];
            let mut max_score = f32::NEG_INFINITY;
            for key_cell in &matrix.values {
                let key = *key_cell * head_weights[HEAD_KEY_SCALE_INDEX]
                    + head_weights[HEAD_KEY_BIAS_INDEX];
                max_score = max_score.max(query * key);
            }
            let mut denominator = 0.0f32;
            let mut weighted_value = 0.0f32;
            for key_cell in &matrix.values {
                let key = *key_cell * head_weights[HEAD_KEY_SCALE_INDEX]
                    + head_weights[HEAD_KEY_BIAS_INDEX];
                let value = *key_cell * head_weights[HEAD_VALUE_SCALE_INDEX]
                    + head_weights[HEAD_VALUE_BIAS_INDEX];
                let attention = (query * key - max_score).exp();
                denominator += attention;
                weighted_value += attention * value;
            }
            combined_context +=
                weighted_value / denominator * head_weights[HEAD_OUTPUT_SCALE_INDEX];
        }
        attention_values[query_index] = combined_context;
    }
    attention_values
}

fn collapse_pair_matrix(
    matrix: &PairMatrix2D,
    target_output_bias: &[f32],
    send_output_bias: &[f32],
    output_scalar_weights: &[f32],
    action_targets_per_source: usize,
) -> Result<Vec<ActionOutput>, ModelError> {
    let row_count = matrix.row_count;
    if action_targets_per_source == 0
        || target_output_bias.len() != row_count * action_targets_per_source
        || send_output_bias.len() != row_count * action_targets_per_source
        || output_scalar_weights.len() != OUTPUT_SCALAR_WEIGHTS * action_targets_per_source
    {
        return Err(ModelError::InvalidParameter);
    }
    let mut outputs = Vec::with_capacity(row_count);
    for source_row in 0..row_count {
        let mut targets = [ActionTargetOutput {
            target_fraction: 0.0,
            send_fraction: 0.0,
        }; crate::config::TRUE2D_ACTION_TARGETS_PER_SOURCE];
        let mut send_fraction_sum = 0.0f32;
        for action_index in 0..action_targets_per_source {
            let bias_start = action_index * row_count;
            let bias_end = bias_start + row_count;
            let scalar_start = action_index * OUTPUT_SCALAR_WEIGHTS;
            let scalar_weights =
                &output_scalar_weights[scalar_start..scalar_start + OUTPUT_SCALAR_WEIGHTS];
            let target_weights = row_softmax(
                matrix,
                source_row,
                &target_output_bias[bias_start..bias_end],
                scalar_weights[TARGET_OUTPUT_SCALE_INDEX],
            );
            let target_fraction = weighted_target_fraction(&target_weights);
            let send_weights = row_softmax(
                matrix,
                source_row,
                &send_output_bias[bias_start..bias_end],
                scalar_weights[SEND_KEY_SCALE_INDEX],
            );
            let mut send_logit = scalar_weights[SEND_OUTPUT_BIAS_INDEX];
            for target_row in 0..row_count {
                let cell_index = source_row * row_count + target_row;
                send_logit += send_weights[target_row]
                    * matrix.values[cell_index]
                    * scalar_weights[SEND_VALUE_SCALE_INDEX];
            }
            targets[action_index] = ActionTargetOutput {
                target_fraction: target_fraction.clamp(PROBABILITY_MIN, PROBABILITY_MAX),
                send_fraction: sigmoid(send_logit * scalar_weights[SEND_OUTPUT_SCALE_INDEX]),
            };
            send_fraction_sum += targets[action_index].send_fraction;
        }
        if send_fraction_sum > PROBABILITY_MAX {
            for target in targets.iter_mut().take(action_targets_per_source) {
                target.send_fraction /= send_fraction_sum;
            }
        }
        outputs.push(ActionOutput { targets });
    }
    Ok(outputs)
}

fn row_softmax(
    matrix: &PairMatrix2D,
    source_row: usize,
    output_bias: &[f32],
    scale: f32,
) -> Vec<f32> {
    let row_count = matrix.row_count;
    let row_offset = source_row * row_count;
    let mut max_score = f32::NEG_INFINITY;
    for target_row in 0..row_count {
        let score = matrix.values[row_offset + target_row] * scale + output_bias[target_row];
        max_score = max_score.max(score);
    }
    let mut weights = vec![0.0f32; row_count];
    let mut denominator = 0.0f32;
    for target_row in 0..row_count {
        let score = matrix.values[row_offset + target_row] * scale + output_bias[target_row];
        let attention = (score - max_score).exp();
        weights[target_row] = attention;
        denominator += attention;
    }
    for weight in weights.iter_mut() {
        *weight /= denominator;
    }
    weights
}

fn weighted_target_fraction(target_weights: &[f32]) -> f32 {
    if target_weights.len() <= 1 {
        return PROBABILITY_MIN;
    }
    let denominator = (target_weights.len() - 1) as f32;
    target_weights
        .iter()
        .enumerate()
        .map(|(target_row, weight)| *weight * (target_row as f32 / denominator))
        .sum()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn deterministic_weight(seed: u64, row: u64, column: usize) -> f32 {
    let mut x =
        seed ^ row.wrapping_mul(HASH_ROW_FACTOR) ^ (column as u64).wrapping_mul(HASH_COLUMN_FACTOR);
    x ^= x >> 30;
    x = x.wrapping_mul(HASH_COLUMN_FACTOR);
    x ^= x >> 27;
    x = x.wrapping_mul(HASH_MIX_FACTOR);
    x ^= x >> 31;
    let normalized = (x as f64 / u64::MAX as f64) as f32;
    normalized * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const MULTI_TARGET_HEAD_PARAMETER_COUNT: usize = 3_004_120;
    const SEND_SUM_TOLERANCE: f32 = 0.000_001;

    #[test]
    fn true_2d_transformer_preserves_pair_matrix_shape_with_mid_attention() {
        let config = AgentConfig::default();
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.max_rows,
        };
        let model = True2DTransformer::seeded(shape, 19, &config).unwrap();
        let rows = vec![RowFeature::empty(&config); config.max_rows];
        let trace = model.forward_with_trace(&rows, &config).unwrap();
        let parameter_count = shape.parameter_count(&config).unwrap();

        assert_eq!(trace.expanded_shape, (config.max_rows, config.max_rows));
        assert_eq!(parameter_count, MULTI_TARGET_HEAD_PARAMETER_COUNT);
        assert_eq!(trace.attention_shapes.len(), 1);
        assert!(trace.attention_shapes.iter().all(|shape| {
            *shape
                == (
                    config.max_rows * config.max_rows,
                    config.max_rows * config.max_rows,
                )
        }));
        assert_eq!(trace.layer_shapes.len(), config.default_model_layers);
        assert!(trace
            .layer_shapes
            .iter()
            .all(|shape| *shape == (config.max_rows, config.max_rows)));
        assert_eq!(trace.outputs.len(), config.max_rows);
        assert!(trace.outputs.iter().all(|output| {
            output
                .targets
                .iter()
                .map(|target| target.send_fraction)
                .sum::<f32>()
                <= 1.0 + SEND_SUM_TOLERANCE
        }));
    }

    #[test]
    fn true_2d_mutation_and_crossover_preserve_shape() {
        let config = AgentConfig::default();
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.max_rows,
        };
        let left = True2DTransformer::seeded(shape, 1, &config).unwrap();
        let right = left.mutate(2, config.default_mutation_sigma);
        let child = True2DTransformer::crossover(&left, &right, 3).unwrap();

        assert_eq!(child.shape, shape);
        assert_eq!(child.weights.len(), left.weights.len());
    }

    #[test]
    fn true_2d_legacy_five_feature_weights_migrate_to_velocity_input() {
        let config = AgentConfig::default();
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.max_rows,
        };
        let expected = shape.parameter_count(&config).unwrap();
        let legacy_expected = expected - 4;
        let legacy_weights = (0..legacy_expected)
            .map(|index| index as f32 + 0.25)
            .collect::<Vec<_>>();

        let model =
            True2DTransformer::from_weights(shape, legacy_weights.clone(), &config).unwrap();

        assert_eq!(model.weights.len(), expected);
        assert_eq!(model.weights[0], legacy_weights[0]);
        assert_eq!(model.weights[SOURCE_Y_WEIGHT_INDEX], legacy_weights[3]);
        assert_eq!(
            model.weights[SOURCE_PRODUCTION_WEIGHT_INDEX],
            legacy_weights[4]
        );
        assert_eq!(model.weights[SOURCE_VELOCITY_X_WEIGHT_INDEX], 0.0);
        assert_eq!(model.weights[SOURCE_VELOCITY_Y_WEIGHT_INDEX], 0.0);
        assert_eq!(model.weights[TARGET_OWNER_WEIGHT_INDEX], legacy_weights[5]);
        assert_eq!(model.weights[TARGET_Y_WEIGHT_INDEX], legacy_weights[8]);
        assert_eq!(
            model.weights[TARGET_PRODUCTION_WEIGHT_INDEX],
            legacy_weights[9]
        );
        assert_eq!(model.weights[TARGET_VELOCITY_X_WEIGHT_INDEX], 0.0);
        assert_eq!(model.weights[TARGET_VELOCITY_Y_WEIGHT_INDEX], 0.0);
        assert_eq!(
            model.weights[INPUT_PAIR_FEATURE_WEIGHTS],
            legacy_weights[LEGACY_FIVE_FEATURE_INPUT_PAIR_WEIGHTS]
        );
    }

    #[test]
    fn true_2d_legacy_four_feature_weights_migrate_to_production_and_velocity_input() {
        let config = AgentConfig::default();
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.max_rows,
        };
        let expected = shape.parameter_count(&config).unwrap();
        let legacy_expected = expected - 6;
        let legacy_weights = (0..legacy_expected)
            .map(|index| index as f32 + 0.25)
            .collect::<Vec<_>>();

        let model =
            True2DTransformer::from_weights(shape, legacy_weights.clone(), &config).unwrap();

        assert_eq!(model.weights.len(), expected);
        assert_eq!(model.weights[0], legacy_weights[0]);
        assert_eq!(model.weights[SOURCE_Y_WEIGHT_INDEX], legacy_weights[3]);
        assert_eq!(model.weights[SOURCE_PRODUCTION_WEIGHT_INDEX], 0.0);
        assert_eq!(model.weights[SOURCE_VELOCITY_X_WEIGHT_INDEX], 0.0);
        assert_eq!(model.weights[SOURCE_VELOCITY_Y_WEIGHT_INDEX], 0.0);
        assert_eq!(model.weights[TARGET_OWNER_WEIGHT_INDEX], legacy_weights[4]);
        assert_eq!(model.weights[TARGET_Y_WEIGHT_INDEX], legacy_weights[7]);
        assert_eq!(model.weights[TARGET_PRODUCTION_WEIGHT_INDEX], 0.0);
        assert_eq!(model.weights[TARGET_VELOCITY_X_WEIGHT_INDEX], 0.0);
        assert_eq!(model.weights[TARGET_VELOCITY_Y_WEIGHT_INDEX], 0.0);
        assert_eq!(
            model.weights[INPUT_PAIR_FEATURE_WEIGHTS],
            legacy_weights[LEGACY_FOUR_FEATURE_INPUT_PAIR_WEIGHTS]
        );
    }
}
