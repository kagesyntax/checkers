# AlphaCheckers

AlphaCheckers is a zero-knowledge self-play checkers engine and training dashboard built from the ground up in Rust. It combines a high-performance bitboard-based rules engine with a Deep Residual Neural Network and Monte Carlo Tree Search (MCTS).

## 🚀 Features

- **Bitboard Engine:** High-performance move generation using `u32` bitmasks, supporting full checkers rules including forced captures and multi-jump sequences.
- **Neural Network:** A Residual Network (ResNet) built with the **Burn** framework, featuring dual heads for policy (move probabilities) and value (win probability) prediction.
- **PUCT Search:** Monte Carlo Tree Search guided by neural network priors, allowing the engine to explore promising lines of play efficiently.
- **Self-Play Training:** Integrated training loop that generates experience from self-play games and optimizes the model in real-time.
- **Dioxus UI:** A modern, interactive dashboard for monitoring training progress, visualizing search heatmaps, and playing against the AI.
- **Cross-Platform:** Supports Web (WASM), Desktop, and Server targets with CPU (NdArray) and GPU (Wgpu) acceleration.

## 🏗️ Architecture

The project is structured into several modular components:

- **`src/engine.rs`**: The "physics" of the game. Handles board representation, legal move generation, and game state transitions using bitboards.
- **`src/ml.rs`**: Neural network definitions and backend integration. Implements the ResNet architecture and provides wrappers for inference and training using the Burn framework.
- **`src/search.rs`**: Implements MCTS and PUCT search logic. This is where the engine "thinks" by simulating future moves guided by the neural network.
- **`src/training.rs`**: Orchestrates the self-play loop, manages the experience replay buffer, and handles model optimization.
- **`src/main.rs`**: The Dioxus application entry point, defining the UI components, routing, and state management.

## 🛠️ Getting Started

### Prerequisites

- **Rust:** You'll need the latest stable version of Rust installed.
- **Dioxus CLI:** Install it via `cargo install dioxus-cli`.

### Running the App

To start the development server for the web platform:

```bash
dx serve --platform web
```

For the desktop platform (with GPU acceleration if enabled):

```bash
dx serve --platform desktop
```

### Feature Flags

- `ml-cpu`: Enables neural network support using the NdArray CPU backend.
- `ml-gpu`: Enables neural network support using the Wgpu GPU backend.
- `web`: Optimized for WASM/Browser environments.
- `desktop`: Optimized for native desktop environments.

## 📊 Training Dashboard

The training dashboard provides real-time insights into the engine's learning process:
- **Optimization Trace:** Live chart showing the reduction in policy and value loss over time.
- **Mental Heatmap:** Visualizes the MCTS visit density on the board, showing which squares the AI is focusing on.
- **Metrics:** Track training steps, position evaluations, and tournament wins/losses between different versions of the engine.

## 🧪 Testing

The project includes a comprehensive suite of unit tests for the engine, ML components, and search logic.

```bash
cargo test --all-features
```

## 📜 License

This project is licensed under the MIT License.
