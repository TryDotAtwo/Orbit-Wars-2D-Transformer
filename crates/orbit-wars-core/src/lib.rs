pub mod config;
pub mod decoder;
pub mod encoder;
pub mod geometry;
pub mod model;
pub mod scaler;
pub mod simulator;
pub mod true2d_model;
pub mod types;

pub use config::AgentConfig;
pub use decoder::{
    decode_model_outputs, decode_model_outputs_with_trace, DecodeError, DecodedMoveCommand,
};
pub use encoder::{encode_state, EncodeError, EncodedState};
pub use model::{
    true2d_model_binary_bytes, ModelError, TrainableTransformer, TransformerModel, TransformerShape,
};
pub use simulator::{
    is_terminal, CometGroup, LaunchedFleetEvent, PlayerStepEvents, SimulationState,
    SimulationStepEvents, SunDestroyedFleetEvent,
};
pub use true2d_model::{
    PairMatrix2D, True2DForwardTrace, True2DTransformer, True2DTransformerShape,
};
pub use types::{ActionOutput, ActionTargetOutput, Fleet, MoveCommand, Planet, RowFeature};
