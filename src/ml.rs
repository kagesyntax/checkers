use crate::engine::{square_coords, BitBoard, GameState, Side};

pub const INPUT_PLANES: usize = 4;
pub const BOARD_ROWS: usize = 8;
pub const DARK_COLS: usize = 4;
pub const INPUT_SIZE: usize = INPUT_PLANES * BOARD_ROWS * DARK_COLS;
pub const POLICY_SIZE: usize = 32 * 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualModelConfig {
    pub residual_blocks: usize,
    pub channels: usize,
    pub policy_outputs: usize,
}

impl Default for ResidualModelConfig {
    fn default() -> Self {
        Self {
            residual_blocks: 10,
            channels: 64,
            policy_outputs: POLICY_SIZE,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetworkOutput {
    pub policy: Vec<f32>,
    pub value: f32,
}

pub fn encode_state(state: GameState) -> [f32; INPUT_SIZE] {
    let mut features = [0.0; INPUT_SIZE];
    encode_piece_plane(&mut features, state.board, Side::Black, false, 0);
    encode_piece_plane(&mut features, state.board, Side::Black, true, 1);
    encode_piece_plane(&mut features, state.board, Side::Red, false, 2);

    let turn_value = if state.turn == Side::Black { 1.0 } else { -1.0 };
    for row in 0..BOARD_ROWS {
        for dark_col in 0..DARK_COLS {
            features[feature_index(3, row, dark_col)] = turn_value;
        }
    }

    features
}

pub fn move_policy_index(from: u8, to: u8) -> usize {
    from as usize * 32 + to as usize
}

pub fn legal_policy_mask(state: GameState) -> Vec<f32> {
    let mut mask = vec![0.0; POLICY_SIZE];
    for mv in state.legal_moves() {
        mask[move_policy_index(mv.from, mv.to)] = 1.0;
    }
    mask
}

pub fn initial_policy_value(state: GameState) -> NetworkOutput {
    let legal = state.legal_moves();
    let mut policy = vec![0.0; POLICY_SIZE];
    let probability = if legal.is_empty() {
        0.0
    } else {
        1.0 / legal.len() as f32
    };

    for mv in legal {
        policy[move_policy_index(mv.from, mv.to)] = probability;
    }

    NetworkOutput {
        policy,
        value: crate::search::evaluate(state, state.turn),
    }
}

fn encode_piece_plane(
    features: &mut [f32; INPUT_SIZE],
    board: BitBoard,
    side: Side,
    kings_only: bool,
    plane: usize,
) {
    let pieces = board.side_bits(side);
    for square in 0..32_u8 {
        let mask = 1u32 << square;
        if pieces & mask == 0 {
            continue;
        }

        let is_king = board.kings & mask != 0;
        if kings_only != is_king {
            continue;
        }

        let (row, col) = square_coords(square);
        features[feature_index(plane, row, col / 2)] = 1.0;
    }
}

fn feature_index(plane: usize, row: usize, dark_col: usize) -> usize {
    plane * BOARD_ROWS * DARK_COLS + row * DARK_COLS + dark_col
}

// ═══════════════════════════════════════════════════════════════════
// Burn backend helpers (smoke tests, label)
// ═══════════════════════════════════════════════════════════════════

#[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
pub mod burn_backend {
    use super::{encode_state, BOARD_ROWS, DARK_COLS, INPUT_PLANES};
    use crate::engine::GameState;
    use burn::tensor::{backend::Backend, Tensor};

    pub fn board_tensor<B: Backend>(state: GameState, device: &B::Device) -> Tensor<B, 4> {
        Tensor::<B, 1>::from_floats(encode_state(state), device).reshape([
            1,
            INPUT_PLANES,
            BOARD_ROWS,
            DARK_COLS,
        ])
    }

    pub fn tensor_shape_smoke<B: Backend>(device: &B::Device) -> [usize; 4] {
        board_tensor::<B>(GameState::new(), device).shape().dims()
    }

    pub fn backend_label() -> &'static str {
        #[cfg(feature = "ml-gpu")]
        {
            "Wgpu (GPU)"
        }
        #[cfg(not(feature = "ml-gpu"))]
        {
            "NdArray (CPU)"
        }
    }

    pub async fn run_backend_smoke() -> String {
        #[cfg(feature = "ml-gpu")]
        {
            type B = burn::backend::Wgpu;
            match std::panic::catch_unwind(|| {
                let device = Default::default();
                let shape = tensor_shape_smoke::<B>(&device);
                format!("Wgpu (GPU) OK — tensor shape {:?}", shape)
            }) {
                Ok(msg) => msg,
                Err(_) => "Wgpu initialization failed".to_string(),
            }
        }
        #[cfg(not(feature = "ml-gpu"))]
        {
            type B = burn::backend::NdArray;
            let device = Default::default();
            let shape = tensor_shape_smoke::<B>(&device);
            format!("NdArray (CPU) OK — tensor shape {:?}", shape)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Neural network: model, inference, training
// ═══════════════════════════════════════════════════════════════════

#[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
pub mod network {
    use burn::module::Module;
    use burn::nn::conv::{Conv2d, Conv2dConfig};
    use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
    use burn::tensor::activation::{log_softmax, tanh};
    use burn::tensor::{backend::Backend, Tensor};

    use super::{BOARD_ROWS, DARK_COLS, INPUT_PLANES, POLICY_SIZE};

    // ── Model output ───────────────────────────────────────────────

    #[derive(Debug, Clone)]
    pub struct ModelOutput<B: Backend> {
        pub policy_logits: Tensor<B, 2>,
        pub value: Tensor<B, 1>,
    }

    // ── Residual block ─────────────────────────────────────────────

    #[derive(Module, Debug)]
    pub struct ResidualBlock<B: Backend> {
        conv1: Conv2d<B>,
        bn1: BatchNorm<B>,
        conv2: Conv2d<B>,
        bn2: BatchNorm<B>,
        relu: Relu,
    }

    impl<B: Backend> ResidualBlock<B> {
        pub fn new(channels: usize, device: &B::Device) -> Self {
            Self {
                conv1: Conv2dConfig::new([channels, channels], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                bn1: BatchNormConfig::new(channels).init(device),
                conv2: Conv2dConfig::new([channels, channels], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                bn2: BatchNormConfig::new(channels).init(device),
                relu: Relu::new(),
            }
        }

        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            let residual = x.clone();
            let x = self.relu.forward(self.bn1.forward(self.conv1.forward(x)));
            let x = self.bn2.forward(self.conv2.forward(x));
            self.relu.forward(x + residual)
        }
    }

    // ── Full network ───────────────────────────────────────────────

    #[derive(Module, Debug)]
    pub struct CheckersNet<B: Backend> {
        conv_entry: Conv2d<B>,
        bn_entry: BatchNorm<B>,
        blocks: Vec<ResidualBlock<B>>,
        relu: Relu,
        policy_head: Linear<B>,
        value_fc1: Linear<B>,
        value_fc2: Linear<B>,
    }

    impl<B: Backend> CheckersNet<B> {
        pub fn new(num_blocks: usize, channels: usize, device: &B::Device) -> Self {
            let blocks = (0..num_blocks)
                .map(|_| ResidualBlock::new(channels, device))
                .collect();
            let shared_flat = channels * BOARD_ROWS * DARK_COLS;
            Self {
                conv_entry: Conv2dConfig::new([INPUT_PLANES, channels], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                bn_entry: BatchNormConfig::new(channels).init(device),
                blocks,
                relu: Relu::new(),
                policy_head: LinearConfig::new(shared_flat, POLICY_SIZE).init(device),
                value_fc1: LinearConfig::new(shared_flat, 256).init(device),
                value_fc2: LinearConfig::new(256, 1).init(device),
            }
        }

        pub fn forward(&self, x: Tensor<B, 4>) -> ModelOutput<B> {
            let x = self
                .relu
                .forward(self.bn_entry.forward(self.conv_entry.forward(x)));
            let x = self.blocks.iter().fold(x, |acc, block| block.forward(acc));
            let [batch, channels, _h, _w] = x.dims();
            let flat = x.reshape([batch, channels * BOARD_ROWS * DARK_COLS]);
            let policy_logits = self.policy_head.forward(flat.clone());
            let value_hidden = self.relu.forward(self.value_fc1.forward(flat));
            let value = tanh(self.value_fc2.forward(value_hidden)).squeeze_dims::<1>(&[1]);
            ModelOutput {
                policy_logits,
                value,
            }
        }
    }

    // ── NeuralModel wrapper ────────────────────────────────────────
    //
    // Uses Autodiff<NdArray> for both inference and training so the
    // model can be trained in-place without copying weights between
    // backends.  The gpu_enabled flag selects Wgpu for inference
    // when available — but because Burn's backends are separate Rust
    // types, runtime switching requires re-creating the model.  For
    // Uses Autodiff<NdArray> for both inference and training so the
    // model can be trained in-place without copying weights between
    // backends. NdArray is used for WASM compatibility (no blocking futures).

    use burn::backend::{Autodiff, NdArray};
    use burn::nn::loss::{MseLoss, Reduction};
    use burn::optim::LearningRate;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

    use super::{encode_state, NetworkOutput, INPUT_SIZE};
    use crate::engine::GameState;

    type TB = Autodiff<NdArray>;

    use std::sync::{Arc, Mutex};

    /// Owns the neural network, optimizer, and handles both
    /// inference (predict) and training (train_batch).
    #[derive(Clone)]
    pub struct NeuralModel {
        inner: Arc<Mutex<NeuralModelInner>>,
        device: <NdArray as Backend>::Device,
    }

    struct NeuralModelInner {
        model: CheckersNet<TB>,
        optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, CheckersNet<TB>, TB>,
    }

    impl NeuralModel {
        pub fn new(num_blocks: usize, channels: usize) -> Self {
            let device = Default::default();
            let model = CheckersNet::new(num_blocks, channels, &device);
            let optimizer = AdamWConfig::new()
                .with_weight_decay(1e-4)
                .with_beta_1(0.9)
                .with_beta_2(0.999)
                .init();
            Self {
                inner: Arc::new(Mutex::new(NeuralModelInner { model, optimizer })),
                device,
            }
        }

        /// Run a forward pass and return (policy, value) for a single position.
        pub async fn predict(&self, state: GameState) -> NetworkOutput {
            let features = encode_state(state);
            let input = Tensor::<TB, 1>::from_floats(features, &self.device).reshape([
                1,
                INPUT_PLANES,
                BOARD_ROWS,
                DARK_COLS,
            ]);
            let output = self.inner.lock().unwrap().model.forward(input);

            // Extract policy logits → softmax probabilities
            // Use into_data().await to avoid blocking on WASM.
            let policy_data = output.policy_logits.into_data();
            let policy = softmax(policy_data.to_vec::<f32>().unwrap_or_default());

            // Extract scalar value
            let value_data = output.value.into_data();
            let value = value_data
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);

            NetworkOutput { policy, value }
        }

        /// Train on a mini-batch. Returns the combined loss value.
        pub async fn train_batch(
            &mut self,
            states: Vec<[f32; INPUT_SIZE]>,
            target_policies: Vec<Vec<f32>>,
            target_values: Vec<f32>,
            learning_rate: f64,
        ) -> f32 {
            let batch = states.len();

            // Build input tensor [batch, 4, 8, 4]
            let flat: Vec<f32> = states.iter().flat_map(|s| s.iter().copied()).collect();
            let input = Tensor::<TB, 1>::from_floats(flat.as_slice(), &self.device).reshape([
                batch,
                INPUT_PLANES,
                BOARD_ROWS,
                DARK_COLS,
            ]);

            // Build target policy tensor [batch, POLICY_SIZE]
            let pflat: Vec<f32> = target_policies
                .iter()
                .flat_map(|p| p.iter().copied())
                .collect();
            let target_policy = Tensor::<TB, 1>::from_floats(pflat.as_slice(), &self.device)
                .reshape([batch, POLICY_SIZE]);

            // Build target value tensor [batch]
            let target_value = Tensor::<TB, 1>::from_floats(target_values.as_slice(), &self.device);

            // Forward
            let output = self.inner.lock().unwrap().model.forward(input);

            // Policy loss: cross-entropy with soft targets
            let log_policy = log_softmax(output.policy_logits, 1);
            let policy_loss = log_policy.mul(target_policy).neg().sum_dim(1).mean();

            // Value loss: MSE
            let mse = MseLoss::new();
            let value_loss = mse.forward(
                output.value.unsqueeze::<2>(),
                target_value.unsqueeze::<2>(),
                Reduction::Mean,
            );

            // Combined
            let loss = policy_loss + value_loss;

            // Backward + optimizer step
            let grads = loss.backward();
            let mut inner = self.inner.lock().unwrap();
            let grads = GradientsParams::from_grads(grads, &inner.model);
            let lr: LearningRate = learning_rate;
            let model = inner.model.clone();
            let updated = inner.optimizer.step(lr, model, grads);
            inner.model = updated;

            loss.clone()
                .inner()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0)
        }
    }

    fn softmax(mut logits: Vec<f32>) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for v in &mut logits {
            *v = (*v - max).exp();
        }
        let sum: f32 = logits.iter().sum();
        if sum > 0.0 {
            for v in &mut logits {
                *v /= sum;
            }
        }
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_has_expected_size_and_piece_counts() {
        let encoded = encode_state(GameState::new());
        assert_eq!(encoded.len(), INPUT_SIZE);
        assert_eq!(encoded.iter().filter(|value| **value == 1.0).count(), 56);
    }

    #[test]
    fn initial_policy_assigns_probability_to_legal_moves() {
        let state = GameState::new();
        let output = initial_policy_value(state);
        assert_eq!(output.policy.len(), POLICY_SIZE);
        assert_eq!(
            output.policy.iter().filter(|value| **value > 0.0).count(),
            7
        );
    }

    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    #[test]
    fn burn_ndarray_tensor_shape_matches_model_input() {
        type B = burn::backend::NdArray;
        let device = Default::default();
        assert_eq!(
            burn_backend::tensor_shape_smoke::<B>(&device),
            [1, INPUT_PLANES, BOARD_ROWS, DARK_COLS]
        );
    }

    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    #[tokio::test]
    async fn backend_label_returns_correct_strings() {
        // Now returns platform-specific label
        let label = burn_backend::backend_label();
        assert!(label.contains("NdArray") || label.contains("Wgpu"));
    }

    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    #[tokio::test]
    async fn run_backend_smoke_ndarray_succeeds() {
        let result = burn_backend::run_backend_smoke().await;
        assert!(
            result.contains("OK"),
            "Expected success message, got: {result}"
        );
    }

    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    #[tokio::test]
    async fn neural_model_predict_returns_valid_output() {
        let model = network::NeuralModel::new(2, 32);
        let output = model.predict(GameState::new()).await;
        assert_eq!(output.policy.len(), POLICY_SIZE);
        assert!(output.value >= -1.0 && output.value <= 1.0);
        let policy_sum: f32 = output.policy.iter().sum();
        assert!(
            (policy_sum - 1.0).abs() < 0.01,
            "policy should sum to ~1.0, got {policy_sum}"
        );
    }

    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    #[tokio::test]
    async fn neural_model_train_batch_produces_finite_loss() {
        let mut model = network::NeuralModel::new(2, 32);
        let state = encode_state(GameState::new());
        let target_policy = vec![0.0_f32; POLICY_SIZE];
        let target_value = 0.5_f32;

        let loss = model
            .train_batch(
                vec![state; 4],
                vec![target_policy; 4],
                vec![target_value; 4],
                0.001,
            )
            .await;
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
    }
}
