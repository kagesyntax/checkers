use crate::engine::{GameResult, GameState, Move, Piece, Side};

#[derive(Debug, Clone, PartialEq)]
pub struct SearchReport {
    pub best_move: Option<Move>,
    pub value: f32,
    pub simulations: u128,
    pub visits: Vec<(Move, u32)>,
}

/// Pure MCTS search with random rollouts.
/// `simulations` controls how many playouts run — use `u128::MAX` for
/// effectively unlimited search depth.
pub fn choose_move(state: GameState, simulations: u128) -> SearchReport {
    let legal_moves = state.legal_moves();
    if legal_moves.is_empty() {
        return SearchReport {
            best_move: None,
            value: terminal_value(state, state.turn).unwrap_or(0.0),
            simulations: 0,
            visits: Vec::new(),
        };
    }

    let simulations = simulations.max(legal_moves.len() as u128);
    let root_side = state.turn;
    let mut stats = vec![MoveStats::default(); legal_moves.len()];
    let mut rng = SmallRng::new(position_seed(state));

    for i in 0..simulations {
        let index = if (i as usize) < legal_moves.len() {
            i as usize
        } else {
            select_uct(&stats, i)
        };
        let child = state.apply(legal_moves[index]);
        let value = rollout(child, root_side, &mut rng, 72);
        stats[index].visits += 1;
        stats[index].value_sum += value;
    }

    let (best_index, best_stats) = stats
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a_score = a.value_sum / a.visits.max(1) as f32;
            let b_score = b.value_sum / b.visits.max(1) as f32;
            a_score.total_cmp(&b_score)
        })
        .expect("legal moves are non-empty");

    SearchReport {
        best_move: Some(legal_moves[best_index]),
        value: best_stats.value_sum / best_stats.visits.max(1) as f32,
        simulations,
        visits: legal_moves
            .into_iter()
            .zip(stats)
            .map(|(mv, stats)| (mv, stats.visits))
            .collect(),
    }
}

/// Neural-network-guided MCTS: uses the model's policy as a prior for
/// move selection, then runs fast heuristic rollouts.  The neural
/// network is called **once** (for the root prior) so the search
/// stays responsive on CPU.
#[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
pub async fn choose_move_with_model(
    state: GameState,
    simulations: u128,
    model: &crate::ml::network::NeuralModel,
) -> SearchReport {
    let legal_moves = state.legal_moves();
    if legal_moves.is_empty() {
        return SearchReport {
            best_move: None,
            value: terminal_value(state, state.turn).unwrap_or(0.0),
            simulations: 0,
            visits: Vec::new(),
        };
    }

    let simulations = simulations.max(legal_moves.len() as u128);
    let root_side = state.turn;

    // Single neural-network call: get the prior policy for the root.
    // The prior guides MCTS toward promising moves (AlphaZero-style).
    let network_output = model.predict(state).await;
    let root_value = network_output.value;
    let prior: Vec<f32> = legal_moves
        .iter()
        .map(|mv| {
            let idx = crate::ml::move_policy_index(mv.from, mv.to);
            network_output.policy.get(idx).copied().unwrap_or(0.0)
        })
        .collect();

    let mut stats = vec![MoveStats::default(); legal_moves.len()];

    // Seed the network's preferred move with one visit from the root
    // value so the prior has immediate influence on search direction.
    let best_prior_idx = prior
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    stats[best_prior_idx].visits += 1;
    stats[best_prior_idx].value_sum += root_value;

    let mut rng = SmallRng::new(position_seed(state));

    for i in 0..simulations {
        let index = if (i as usize) < legal_moves.len() {
            i as usize
        } else {
            select_uct_with_prior(&stats, i, &prior)
        };
        let child = state.apply(legal_moves[index]);
        // Fast heuristic rollout — no neural network inside the loop.
        let value = rollout(child, root_side, &mut rng, 72);
        stats[index].visits += 1;
        stats[index].value_sum += value;
    }

    let (best_index, best_stats) = stats
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a_score = a.value_sum / a.visits.max(1) as f32;
            let b_score = b.value_sum / b.visits.max(1) as f32;
            a_score.total_cmp(&b_score)
        })
        .expect("legal moves are non-empty");

    SearchReport {
        best_move: Some(legal_moves[best_index]),
        value: best_stats.value_sum / best_stats.visits.max(1) as f32,
        simulations,
        visits: legal_moves
            .into_iter()
            .zip(stats)
            .map(|(mv, stats)| (mv, stats.visits))
            .collect(),
    }
}

pub fn evaluate(state: GameState, side: Side) -> f32 {
    if let Some(value) = terminal_value(state, side) {
        return value;
    }

    let black = material(state, Side::Black);
    let red = material(state, Side::Red);
    let mobility = state.board.legal_moves(side).len() as f32
        - state.board.legal_moves(side.opponent()).len() as f32;
    let score = match side {
        Side::Black => black - red,
        Side::Red => red - black,
    } + mobility * 0.08;

    (score / 18.0).clamp(-1.0, 1.0)
}

fn rollout(mut state: GameState, root_side: Side, rng: &mut SmallRng, depth: u8) -> f32 {
    for _ in 0..depth {
        if let Some(value) = terminal_value(state, root_side) {
            return value;
        }

        let legal_moves = state.legal_moves();
        if legal_moves.is_empty() {
            return terminal_value(state, root_side).unwrap_or(0.0);
        }

        let index = rng.next_index(legal_moves.len());
        state = state.apply(legal_moves[index]);
    }

    evaluate(state, root_side)
}

fn select_uct(stats: &[MoveStats], parent_visits: u128) -> usize {
    let exploration = 1.414_f32;
    stats
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a_score = uct_score(**a, parent_visits, exploration);
            let b_score = uct_score(**b, parent_visits, exploration);
            a_score.total_cmp(&b_score)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// UCT selection enhanced with a neural-network prior policy.
#[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
fn select_uct_with_prior(stats: &[MoveStats], parent_visits: u128, prior: &[f32]) -> usize {
    let exploration = 1.414_f32;
    let prior_weight = 0.5_f32;
    stats
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| {
            let a_score =
                uct_score_with_prior(**a, parent_visits, exploration, prior[*i], prior_weight);
            let b_score =
                uct_score_with_prior(**b, parent_visits, exploration, prior[*j], prior_weight);
            a_score.total_cmp(&b_score)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn uct_score(stats: MoveStats, parent_visits: u128, exploration: f32) -> f32 {
    if stats.visits == 0 {
        return f32::INFINITY;
    }

    let average = stats.value_sum / stats.visits as f32;
    let bonus = ((parent_visits.max(1) as f32).ln() / stats.visits as f32).sqrt();
    average + exploration * bonus
}

#[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
fn uct_score_with_prior(
    stats: MoveStats,
    parent_visits: u128,
    exploration: f32,
    prior_prob: f32,
    prior_weight: f32,
) -> f32 {
    if stats.visits == 0 {
        return f32::INFINITY;
    }

    let average = stats.value_sum / stats.visits as f32;
    let exploration_bonus = exploration
        * (prior_prob + prior_weight)
        * ((parent_visits.max(1) as f32).ln() / (1 + stats.visits) as f32).sqrt();
    average + exploration_bonus
}

fn terminal_value(state: GameState, side: Side) -> Option<f32> {
    match state.result()? {
        GameResult::Winner(winner) if winner == side => Some(1.0),
        GameResult::Winner(_) => Some(-1.0),
        GameResult::Draw => Some(0.0),
    }
}

fn material(state: GameState, side: Side) -> f32 {
    let bits = state.board.side_bits(side);
    let mut score = 0.0;

    for square in 0..32 {
        let mask = 1u32 << square;
        if bits & mask == 0 {
            continue;
        }

        let is_king = state.board.kings & mask != 0;
        let piece = match (side, is_king) {
            (Side::Black, false) => Piece::BlackMan,
            (Side::Black, true) => Piece::BlackKing,
            (Side::Red, false) => Piece::RedMan,
            (Side::Red, true) => Piece::RedKing,
        };
        score += match piece {
            Piece::BlackMan | Piece::RedMan => 1.0,
            Piece::BlackKing | Piece::RedKing => 1.8,
        };
    }

    score
}

fn position_seed(state: GameState) -> u64 {
    let side = match state.turn {
        Side::Black => 0x9e37_79b9_7f4a_7c15,
        Side::Red => 0xbf58_476d_1ce4_e5b9,
    };

    ((state.board.black as u64) << 32)
        ^ state.board.red as u64
        ^ ((state.board.kings as u64) << 1)
        ^ side
}

#[derive(Debug, Clone, Copy, Default)]
struct MoveStats {
    visits: u32,
    value_sum: f32,
}

#[derive(Debug, Clone, Copy)]
struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_index(&mut self, len: usize) -> usize {
        (self.next_u32() as usize) % len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_returns_legal_move_from_starting_position() {
        let state = GameState::new();
        let report = choose_move(state, 24);
        assert!(state.legal_moves().contains(&report.best_move.unwrap()));
        assert_eq!(report.simulations, 24);
    }
}
