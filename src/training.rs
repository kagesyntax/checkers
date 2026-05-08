use crate::engine::{GameResult, GameState, Move, Side};
use crate::ml::{encode_state, move_policy_index, INPUT_SIZE, POLICY_SIZE};
use crate::search;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperienceSample {
    pub state: Vec<f32>,
    pub policy: Vec<f32>,
    pub player: SideLabel,
    pub outcome: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SideLabel {
    Black,
    Red,
}

impl From<Side> for SideLabel {
    fn from(side: Side) -> Self {
        match side {
            Side::Black => Self::Black,
            Side::Red => Self::Red,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TrainingMetrics {
    pub self_play_steps: u64,
    pub replay_positions: usize,
    pub completed_games: u64,
    pub black_wins: u64,
    pub red_wins: u64,
    pub draws: u64,
    pub train_steps: u64,
    pub last_loss: f32,
    pub cached_positions: usize,
}

/// A cached move from prior search — lets the AI respond instantly
/// for positions it has already evaluated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedMove {
    pub best_move: Move,
    pub value: f32,
    pub visits: u32,
}

/// Hash a board position for the move cache.
fn board_hash(state: GameState) -> u64 {
    let side = match state.turn {
        Side::Black => 0x9e37_79b9_7f4a_7c15,
        Side::Red => 0xbf58_476d_1ce4_e5b9,
    };
    ((state.board.black as u64) << 32)
        ^ (state.board.red as u64)
        ^ ((state.board.kings as u64) << 1)
        ^ side
        ^ (state.halfmoves as u64).wrapping_mul(0x1000_0000_1ce4_e5b9)
}

pub struct SelfPlayTrainer {
    pub game: GameState,
    pub replay: Vec<ExperienceSample>,
    pending: Vec<PendingSample>,
    metrics: TrainingMetrics,
    /// Pre-computed moves from prior searches. When the AI encounters a
    /// position it's already evaluated (from self-play), it can respond
    /// instantly without re-searching.
    move_cache: HashMap<u64, CachedMove>,
    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    model: crate::ml::network::NeuralModel,
}

impl std::fmt::Debug for SelfPlayTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelfPlayTrainer")
            .field("game", &self.game)
            .field("replay", &self.replay.len())
            .field("pending", &self.pending.len())
            .field("metrics", &self.metrics)
            .field("move_cache", &self.move_cache.len())
            .finish()
    }
}

impl Clone for SelfPlayTrainer {
    fn clone(&self) -> Self {
        Self {
            game: self.game,
            replay: self.replay.clone(),
            pending: self.pending.clone(),
            metrics: self.metrics,
            move_cache: self.move_cache.clone(),
            #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
            model: self.model.clone(),
        }
    }
}

impl Default for SelfPlayTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfPlayTrainer {
    pub fn new() -> Self {
        let mut trainer = Self {
            game: GameState::new(),
            replay: Vec::new(),
            pending: Vec::new(),
            metrics: TrainingMetrics::default(),
            move_cache: HashMap::new(),
            #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
            model: crate::ml::network::NeuralModel::new(2, 32),
        };

        // Try to load replay buffer from localStorage for persistence
        trainer.load_replay_from_storage();
        trainer
    }

    pub fn metrics(&self) -> TrainingMetrics {
        TrainingMetrics {
            replay_positions: self.replay.len(),
            cached_positions: self.move_cache.len(),
            ..self.metrics
        }
    }

    /// Run one self-play step: pick a move, record experience, and
    /// train the neural network if we have enough data.
    pub async fn step(&mut self, simulations: u128) -> Option<Move> {
        if self.game.result().is_some() {
            self.reset_game();
        }

        let player = self.game.turn;
        let hash = board_hash(self.game);

        // Check the move cache first — if we've already searched this
        // position during self-play, use the cached result instantly.
        let report = if let Some(cached) = self.move_cache.get(&hash) {
            search::SearchReport {
                best_move: Some(cached.best_move),
                value: cached.value,
                simulations: 0, // cache hit, no new simulations
                visits: vec![(cached.best_move, cached.visits)],
            }
        } else {
            // Use the neural network to guide search when available
            #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
            {
                search::choose_move_with_model(self.game, simulations, &self.model).await
            }
            #[cfg(not(any(feature = "ml-cpu", feature = "ml-gpu")))]
            {
                search::choose_move(self.game, simulations)
            }
        };

        let mv = report.best_move?;
        let policy = visit_policy(&report.visits);

        // Cache this result for future instant lookups
        self.move_cache.insert(
            hash,
            CachedMove {
                best_move: mv,
                value: report.value,
                visits: report.visits.first().map(|(_, v)| *v).unwrap_or(1),
            },
        );

        self.pending.push(PendingSample {
            state: encode_state(self.game).to_vec(),
            policy,
            player,
        });

        self.game = self.game.apply(mv);
        self.metrics.self_play_steps += 1;

        if let Some(result) = self.game.result() {
            self.finish_game(result);
            self.game = GameState::new();
        }

        // Train the neural network on replay data after each step
        #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
        self.train_if_ready().await;

        // Pre-compute AI responses for the next few possible positions
        // so the AI can respond instantly in human games
        for next_mv in self.game.legal_moves().iter().take(3) {
            let next_state = self.game.apply(*next_mv);
            self.precompute_for(next_state, 64).await;
        }

        // Periodically save replay to localStorage
        if self.metrics.self_play_steps.is_multiple_of(50) {
            self.save_replay_to_storage();
        }

        Some(mv)
    }

    /// Look up a pre-computed move for the given position.
    /// Returns `None` if the position isn't in the cache.
    pub fn lookup_cached_move(&self, state: GameState) -> Option<&CachedMove> {
        let hash = board_hash(state);
        self.move_cache.get(&hash)
    }

    /// Pre-compute moves for all legal positions from the current game
    /// state. This is called in the background so the AI already knows
    /// its response before the human even moves.
    pub async fn precompute_for(&mut self, state: GameState, simulations: u128) {
        let hash = board_hash(state);
        if self.move_cache.contains_key(&hash) {
            return; // already computed
        }

        #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
        let report = search::choose_move_with_model(state, simulations, &self.model).await;
        #[cfg(not(any(feature = "ml-cpu", feature = "ml-gpu")))]
        let report = search::choose_move(state, simulations);

        if let Some(mv) = report.best_move {
            self.move_cache.insert(
                hash,
                CachedMove {
                    best_move: mv,
                    value: report.value,
                    visits: report.visits.first().map(|(_, v)| *v).unwrap_or(1),
                },
            );
        }
    }

    pub fn reset_game(&mut self) {
        self.game = GameState::new();
        self.pending.clear();
    }

    fn finish_game(&mut self, result: GameResult) {
        self.metrics.completed_games += 1;
        match result {
            GameResult::Winner(Side::Black) => self.metrics.black_wins += 1,
            GameResult::Winner(Side::Red) => self.metrics.red_wins += 1,
            GameResult::Draw => self.metrics.draws += 1,
        }

        for sample in self.pending.drain(..) {
            self.replay.push(ExperienceSample {
                state: sample.state,
                policy: sample.policy,
                player: sample.player.into(),
                outcome: outcome_for(sample.player, result),
            });
        }
    }

    /// Train the neural network on a mini-batch sampled from the
    /// replay buffer.  Runs when we have at least 16 positions.
    /// Uses a fixed learning rate of 0.0001 for deep, stable learning.
    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    async fn train_if_ready(&mut self) {
        let batch_size = 16;
        if self.replay.len() < batch_size {
            return;
        }

        // Sample the most recent `batch_size` positions
        let start = self.replay.len().saturating_sub(batch_size);
        let batch = &self.replay[start..];

        let states: Vec<[f32; INPUT_SIZE]> = batch
            .iter()
            .map(|s| {
                let mut arr = [0.0_f32; INPUT_SIZE];
                arr.copy_from_slice(&s.state[..INPUT_SIZE]);
                arr
            })
            .collect();

        let policies: Vec<Vec<f32>> = batch.iter().map(|s| s.policy.clone()).collect();

        let values: Vec<f32> = batch.iter().map(|s| s.outcome).collect();

        // Fixed learning rate of 0.0001 — as small as practical for
        // deep, stable learning that converges to strong play.
        let loss = self.model.train_batch(states, policies, values, 0.0001).await;
        self.metrics.train_steps += 1;
        self.metrics.last_loss = loss;
    }

    /// Get the neural network's evaluation of the current game position.
    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    pub async fn network_value(&self) -> f32 {
        self.model.predict(self.game).await.value
    }

    // ── Persistence ──────────────────────────────────────────────

    /// Save the replay buffer to localStorage so the AI remembers
    /// its past experience across page reloads.
    fn save_replay_to_storage(&self) {
        #[cfg(feature = "web")]
        {
            if let Ok(json) = serde_json::to_string(&self.replay) {
                if let Some(window) = web_sys::window() {
                    if let Ok(Some(storage)) = window.local_storage() {
                        let _ = storage.set_item("checkers_replay", &json);
                    }
                }
            }
        }
    }

    /// Load the replay buffer from localStorage so the AI can continue
    /// learning from its past experience.
    fn load_replay_from_storage(&mut self) {
        #[cfg(all(feature = "web", target_arch = "wasm32"))]
        {
            if let Some(window) = web_sys::window() {
                if let Ok(Some(storage)) = window.local_storage() {
                    if let Ok(Some(json)) = storage.get_item("checkers_replay") {
                        if let Ok(replay) = serde_json::from_str::<Vec<ExperienceSample>>(&json) {
                            self.replay = replay;
                            self.metrics.replay_positions = self.replay.len();
                        }
                    }
                }
            }
        }
    }
}

fn visit_policy(visits: &[(Move, u32)]) -> Vec<f32> {
    let mut policy = vec![0.0; POLICY_SIZE];
    let total: u32 = visits.iter().map(|(_, visits)| *visits).sum();
    if total == 0 {
        return policy;
    }

    for (mv, visits) in visits {
        policy[move_policy_index(mv.from, mv.to)] = *visits as f32 / total as f32;
    }

    policy
}

fn outcome_for(player: Side, result: GameResult) -> f32 {
    match result {
        GameResult::Winner(winner) if winner == player => 1.0,
        GameResult::Winner(_) => -1.0,
        GameResult::Draw => 0.0,
    }
}

#[derive(Debug, Clone)]
struct PendingSample {
    state: Vec<f32>,
    policy: Vec<f32>,
    player: Side,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn trainer_records_pending_sample_after_step() {
        let mut trainer = SelfPlayTrainer::new();
        let mv = trainer.step(16).await;
        assert!(mv.is_some());
        assert_eq!(trainer.metrics().self_play_steps, 1);
        assert_eq!(trainer.pending.len(), 1);
        assert_eq!(trainer.pending[0].state.len(), crate::ml::INPUT_SIZE);
    }

    #[test]
    fn visit_policy_normalizes_counts() {
        let policy = visit_policy(&[(Move::quiet(0, 4), 3), (Move::quiet(1, 5), 1)]);
        assert_eq!(policy[move_policy_index(0, 4)], 0.75);
        assert_eq!(policy[move_policy_index(1, 5)], 0.25);
    }

    #[tokio::test]
    async fn move_cache_stores_and_retrieves_moves() {
        let mut trainer = SelfPlayTrainer::new();
        let state = GameState::new();
        trainer.precompute_for(state, 24).await;
        let cached = trainer.lookup_cached_move(state);
        assert!(cached.is_some());
    }

    #[test]
    fn board_hash_differs_for_different_positions() {
        let s1 = GameState::new();
        let s2 = s1.apply(s1.legal_moves()[0]);
        assert_ne!(board_hash(s1), board_hash(s2));
    }
}
