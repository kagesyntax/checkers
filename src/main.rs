mod engine;
mod ml;
mod search;
mod training;

use dioxus::prelude::*;
use engine::{
    notation, piece_side, square_index, BitBoard, GameResult, GameState, Move, Piece, Side,
};

#[derive(Debug, Clone, Routable, PartialEq)]
#[rustfmt::skip]
enum Route {
    #[layout(Shell)]
        #[route("/")]
        Home {},
        #[route("/train")]
        Train {},
        #[route("/play")]
        Play {},
}

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let trainer = use_signal(training::SelfPlayTrainer::new);
    use_context_provider(|| trainer);
    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Stylesheet { href: MAIN_CSS }
        document::Stylesheet { href: TAILWIND_CSS }
        Router::<Route> {}
    }
}

#[component]
fn Shell() -> Element {
    rsx! {
        div { class: "app-shell",
            nav { class: "top-nav",
                Link { to: Route::Home {}, class: "brand", "AlphaCheckers" }
                div { class: "nav-links",
                    Link { to: Route::Train {}, "Training" }
                    Link { to: Route::Play {}, "Arena" }
                }
            }
            main { class: "page-frame",
                Outlet::<Route> {}
            }
        }
    }
}

#[component]
fn Home() -> Element {
    rsx! {
        section { class: "hero-panel",
            div { class: "eyebrow", "Zero-knowledge self-play engine" }
            h1 { "GPU-accelerated checkers intelligence, built from bitboards up." }
            p {
                "AlphaCheckers is structured around a fast rules layer, neural policy/value evaluation, \
                PUCT-guided MCTS, and a live Dioxus control surface for training and play."
            }
            div { class: "hero-actions",
                Link { to: Route::Train {}, class: "primary-action", "Open Training Dashboard" }
                Link { to: Route::Play {}, class: "secondary-action", "Play in Arena" }
            }
        }

        section { class: "architecture-grid",
            ArchitectureCard {
                label: "Phase 1",
                title: "Bitboard Physics",
                body: "Three u32 masks represent black pieces, red pieces, and kings. Legal move generation enforces forced captures and exposes perft for correctness testing."
            }
            ArchitectureCard {
                label: "Phase 2",
                title: "Burn ResNet",
                body: "The model seam is designed for a 4 x 8 x 4 board tensor with dual policy/value heads on burn-wgpu."
            }
            ArchitectureCard {
                label: "Phase 3",
                title: "PUCT Search",
                body: "MCTS will consume network priors, expand promising lines, and publish visit-count policy targets back into training."
            }
        }
    }
}

#[component]
fn ArchitectureCard(label: String, title: String, body: String) -> Element {
    rsx! {
        article { class: "architecture-card",
            span { "{label}" }
            h2 { "{title}" }
            p { "{body}" }
        }
    }
}

#[component]
fn Train() -> Element {
    let mut trainer = use_context::<Signal<training::SelfPlayTrainer>>();
    let mut training_last = use_signal(|| "Ready for self-play".to_string());
    let mut auto_run = use_signal(|| false);

    // Continuous self-play: manage the task lifecycle using a RefCell to avoid
    // infinite loops in use_effect (since task_handle doesn't need to trigger re-renders).
    let task_handle = use_hook(|| std::cell::RefCell::new(None::<dioxus::core::Task>));
    use_effect(move || {
        let running = auto_run();
        // Cancel existing task if any
        if let Some(old) = task_handle.borrow_mut().take() {
            old.cancel();
        }
        if running {
            let mut t = trainer;
            let mut tl = training_last;
            let handle = dioxus::prelude::spawn(async move {
                loop {
                    #[cfg(feature = "web")]
                    gloo_timers::future::TimeoutFuture::new(2_000).await;
                    #[cfg(not(feature = "web"))]
                    tokio::time::sleep(std::time::Duration::from_millis(2_000)).await;

                    // Re-check auto_run signal to see if we should continue
                    if !auto_run() {
                        break;
                    }
                    let mut trainer_val = t.read().clone();
                    if let Some(mv) = trainer_val.step(128).await {
                        let metrics = trainer_val.metrics();
                        tl.set(format!(
                            "{} | replay: {} | games: {} | train: {} | loss: {:.4}",
                            mv.notation(),
                            metrics.replay_positions,
                            metrics.completed_games,
                            metrics.train_steps,
                            metrics.last_loss
                        ));
                    }
                    t.set(trainer_val);
                }
            });
            *task_handle.borrow_mut() = Some(handle);
        }
    });

    let trainer_snapshot = trainer();
    let metrics = trainer_snapshot.metrics();

    let training_eval_resource = use_resource(move || async move {
        #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
        {
            let trainer_clone = trainer.read().clone();
            let val = trainer_clone.network_value().await;
            ((val + 1.0) * 50.0) as i32
        }
        #[cfg(not(any(feature = "ml-cpu", feature = "ml-gpu")))]
        {
            ((search::evaluate(trainer.read().game, Side::Black) + 1.0) * 50.0) as i32
        }
    });
    let training_eval = training_eval_resource.read().unwrap_or(50);

    let _network_output = ml::initial_policy_value(trainer_snapshot.game);
    let model_config = ml::ResidualModelConfig::default();
    let legal_mask = ml::legal_policy_mask(trainer_snapshot.game);
    let legal_policy_targets = legal_mask.iter().filter(|value| **value > 0.0).count();

    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    let backend_resource = use_resource(move || async move {
        (
            ml::burn_backend::backend_label(),
            ml::burn_backend::run_backend_smoke().await,
        )
    });
    #[cfg(any(feature = "ml-cpu", feature = "ml-gpu"))]
    let (backend_label, backend_smoke) = backend_resource
        .read()
        .clone()
        .unwrap_or(("Loading backend...", String::new()));

    #[cfg(not(any(feature = "ml-cpu", feature = "ml-gpu")))]
    let (backend_label, backend_smoke): (&str, String) = ("ml feature not enabled", String::new());

    rsx! {
        div { class: "section-heading",
            span { "Training loop" }
            h1 { "Self-play dashboard" }
            p { "Live UI shell for the DeepSeek-style cycle: self-play, optimize, evaluate, promote." }
        }

        section { class: "dashboard-grid",
            MetricCard { label: "Training Steps", value: "{metrics.self_play_steps}", trend: "self-play" }
            MetricCard { label: "Position Value", value: "{training_eval}%", trend: "NN black POV" }
            MetricCard { label: "Replay Positions", value: "{metrics.replay_positions}", trend: "serde-ready" }
            MetricCard { label: "NN Train Steps", value: "{metrics.train_steps}", trend: "backprop" }
            MetricCard { label: "Loss", value: "{metrics.last_loss:.4}", trend: "policy+value" }
            MetricCard { label: "Cached Positions", value: "{metrics.cached_positions}", trend: "instant lookup" }
        }

        section { class: "training-layout",
            article { class: "panel chart-panel",
                div { class: "panel-title",
                    h2 { "Optimization trace" }
                    span { "policy + value loss" }
                }
                svg { class: "line-chart", view_box: "0 0 640 240", role: "img",
                    polyline {
                        points: "0,190 80,168 160,176 240,130 320,118 400,96 480,82 560,66 640,44",
                        fill: "none",
                        stroke: "#f6c453",
                        stroke_width: "5",
                    }
                    polyline {
                        points: "0,210 80,184 160,160 240,150 320,128 400,122 480,92 560,88 640,70",
                        fill: "none",
                        stroke: "#5dd6c8",
                        stroke_width: "5",
                    }
                }
            }

            article { class: "panel heatmap-panel",
                div { class: "panel-title",
                    h2 { "Mental heatmap" }
                    span { "latest MCTS visit density" }
                }
                div { class: "heatmap-grid",
                    for i in 0..32 {
                        div { class: heat_class(i), "{i + 1}" }
                    }
                }
            }

            article { class: "panel control-panel",
                div { class: "panel-title",
                    h2 { "Controls" }
                    span { "training runtime" }
                }
                label { class: "toggle-row",
                    span { "GPU Backend: {backend_label} (Standard)" }
                }
                label { class: "toggle-row",
                    span { "Learning rate: 0.0001 (fixed, for deep learning)" }
                }
                div { class: "button-row",
                    button {
                        onclick: move |_| {
                            auto_run.set(!auto_run());
                        },
                        if auto_run() { "Stop Auto-Play" } else { "Start Auto-Play" }
                    }
                    button {
                        onclick: move |_| {
                            spawn(async move {
                                let mut trainer_val = trainer.read().clone();
                                if let Some(mv) = trainer_val.step(128).await {
                                    let metrics = trainer_val.metrics();
                                    training_last.set(format!(
                                        "{} | replay: {} | games: {} | train: {} | loss: {:.4}",
                                        mv.notation(),
                                        metrics.replay_positions,
                                        metrics.completed_games,
                                        metrics.train_steps,
                                        metrics.last_loss
                                    ));
                                }
                                trainer.set(trainer_val);
                            });
                        },
                        "Step Self-Play"
                    }
                    button { "Export Weights" }
                    button {
                        onclick: move |_| {
                            auto_run.set(false);
                            trainer.with_mut(|trainer| {
                                trainer.reset_game();
                                training_last.set(format!(
                                    "Self-play board reset. Replay still holds {} positions.",
                                    trainer.metrics().replay_positions
                                ));
                            });
                        },
                        "Reset Game"
                    }
                }
                p { class: "runtime-note", "{training_last}" }
                p { class: "runtime-note",
                    "Burn model seam: {model_config.residual_blocks} residual blocks, {model_config.channels} channels, {model_config.policy_outputs}-slot policy head, {legal_policy_targets} legal targets in this position."
                }
                p { class: "runtime-note",
                    "Active backend: {backend_smoke}"
                }
                p { class: "runtime-note",
                    "Tournament counters: black {metrics.black_wins}, red {metrics.red_wins}, draws {metrics.draws}."
                }
            }
        }
    }
}

#[component]
fn MetricCard(label: String, value: String, trend: String) -> Element {
    rsx! {
        article { class: "metric-card",
            span { "{label}" }
            strong { "{value}" }
            small { "{trend}" }
        }
    }
}

#[component]
fn Play() -> Element {
    let mut game = use_signal(GameState::new);
    let mut selected = use_signal(|| None::<u8>);
    let mut history = use_signal(Vec::<String>::new);
    let mut status = use_signal(|| "Black to move. Select a piece.".to_string());
    let mut trainer = use_context::<Signal<training::SelfPlayTrainer>>();
    let board = game().board;
    let legal_moves = game().legal_moves();
    let eval = ((search::evaluate(game(), Side::Black) + 1.0) * 50.0).clamp(0.0, 100.0) as i32;
    let perft_nodes = game().board.perft(game().turn, 1);

    rsx! {
        div { class: "section-heading",
            span { "Arena" }
            h1 { "Human vs AlphaCheckers" }
            p { "Click a black piece, then click a highlighted destination. Red replies with the current CPU search scaffold." }
        }

        section { class: "arena-layout",
            div { class: "board-shell",
                div { class: "checkers-board",
                    for row in 0..8 {
                        for col in 0..8 {
                            ArenaSquare {
                                board,
                                row,
                                col,
                                selected: selected(),
                                legal_moves: legal_moves.clone(),
                                onclick: move |_| {
                                    handle_square_click(row, col, &mut game, &mut selected, &mut history, &mut status, &mut trainer);
                                },
                            }
                        }
                    }
                }
            }

            aside { class: "arena-side panel",
                div { class: "panel-title",
                    h2 { "Search settings" }
                    span { "800 simulations per move | perft(1): {perft_nodes} nodes" }
                }
                div { class: "eval-shell",
                    div { class: "eval-fill", style: "height: {eval}%;" }
                }
                p { class: "runtime-note", "{status}" }
                h3 { "Legal moves" }
                ul { class: "move-list",
                    for mv in legal_moves {
                        li { "{mv.notation()}" }
                    }
                }
                h3 { "Move history" }
                ul { class: "move-list history-list",
                    for item in history() {
                        li { "{item}" }
                    }
                }
                div { class: "button-row",
                    button {
                        disabled: game().result().is_none(),
                        onclick: move |_| {
                            game.set(GameState::new());
                            selected.set(None);
                            history.write().clear();
                            status.set("Black to move. Select a piece.".to_string());
                        },
                        "New Game"
                    }
                }
            }
        }
    }
}

#[component]
fn ArenaSquare(
    board: BitBoard,
    row: usize,
    col: usize,
    selected: Option<u8>,
    legal_moves: Vec<Move>,
    onclick: EventHandler<MouseEvent>,
) -> Element {
    let dark = (row + col) % 2 == 1;
    let square = square_index(row, col);
    let selected_class = if square == selected {
        " selected-square"
    } else {
        ""
    };
    let target_class = if let Some(square) = square {
        if selected
            .map(|from| {
                legal_moves
                    .iter()
                    .any(|mv| mv.from == from && mv.to == square)
            })
            .unwrap_or(false)
        {
            " target-square"
        } else {
            ""
        }
    } else {
        ""
    };
    let base = if dark {
        "square dark-square"
    } else {
        "square light-square"
    };
    let class = format!("{base}{selected_class}{target_class}");

    rsx! {
        div { class, onclick,
            if let Some(piece) = board.piece_at(row, col) {
                div { class: piece_class(piece),
                    if matches!(piece, Piece::BlackKing | Piece::RedKing) {
                        span { "K" }
                    }
                }
            }
        }
    }
}

fn handle_square_click(
    row: usize,
    col: usize,
    game: &mut Signal<GameState>,
    selected: &mut Signal<Option<u8>>,
    history: &mut Signal<Vec<String>>,
    status: &mut Signal<String>,
    trainer: &mut Signal<training::SelfPlayTrainer>,
) {
    let Some(square) = square_index(row, col) else {
        return;
    };

    let current = game();
    if current.turn != Side::Black {
        return;
    }

    if current.result().is_some() {
        status.set("Game over. Start a new game to continue.".to_string());
        return;
    }

    let board = current.board;
    let legal_moves = current.legal_moves();

    if let Some(from) = selected() {
        if let Some(mv) = legal_moves
            .iter()
            .copied()
            .find(|candidate| candidate.from == from && candidate.to == square)
        {
            let after_human = current.apply(mv);
            history.write().push(format!("Black {}", mv.notation()));
            selected.set(None);

            if let Some(result) = after_human.result() {
                game.set(after_human);
                status.set(result_message(result));
                return;
            }

            // Move the AI logic into a spawned task to keep the UI responsive
            // and handle potential async ML calls in the future.
            let mut game = *game;
            let mut history = *history;
            let mut status = *status;
            let trainer = *trainer;

            spawn(async move {
                status.set("Red is thinking...".to_string());
                let (ai_move, ai_value) =
                    if let Some(cached) = trainer.read().lookup_cached_move(after_human) {
                        (cached.best_move, cached.value)
                    } else {
                        let simulations: u128 = 2000;
                        let report = search::choose_move(after_human, simulations);
                        match report.best_move {
                            Some(mv) => (mv, report.value),
                            None => {
                                game.set(after_human);
                                status.set("Red has no legal move. Black wins.".to_string());
                                return;
                            }
                        }
                    };
                let after_ai = after_human.apply(ai_move);
                history
                    .write()
                    .push(format!("Red {} ({:.2})", ai_move.notation(), ai_value));
                game.set(after_ai);
                status.set(
                    after_ai
                        .result()
                        .map(result_message)
                        .unwrap_or_else(|| "Black to move. Select a piece.".to_string()),
                );
            });
            return;
        }
    }

    if let Some(piece) = board.piece_at(row, col) {
        if piece_side(piece) == Side::Black && legal_moves.iter().any(|mv| mv.from == square) {
            selected.set(Some(square));
            status.set(format!("Selected {}.", notation(square)));
            return;
        }
    }

    selected.set(None);
    status.set("Select a black piece with a legal move.".to_string());
}

fn result_message(result: GameResult) -> String {
    match result {
        GameResult::Winner(side) => format!("Game over. {} wins.", side.label()),
        GameResult::Draw => "Game over. Draw by move limit.".to_string(),
    }
}

fn piece_class(piece: Piece) -> &'static str {
    match piece {
        Piece::BlackMan => "piece black-piece",
        Piece::BlackKing => "piece black-piece king-piece",
        Piece::RedMan => "piece red-piece",
        Piece::RedKing => "piece red-piece king-piece",
    }
}

fn heat_class(index: i32) -> &'static str {
    match index % 6 {
        0 => "heat-cell heat-strong",
        1 | 4 => "heat-cell heat-mid",
        2 => "heat-cell heat-low",
        _ => "heat-cell",
    }
}
