#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Black,
    Red,
}

impl Side {
    pub fn opponent(self) -> Self {
        match self {
            Self::Black => Self::Red,
            Self::Red => Self::Black,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Black => "Black",
            Self::Red => "Red",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    BlackMan,
    BlackKing,
    RedMan,
    RedKing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitBoard {
    pub black: u32,
    pub red: u32,
    pub kings: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Winner(Side),
    Draw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameState {
    pub board: BitBoard,
    pub turn: Side,
    pub halfmoves: u16,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub const DRAW_HALFMOVES: u16 = 160;

    pub const fn new() -> Self {
        Self {
            board: BitBoard::starting_position(),
            turn: Side::Black,
            halfmoves: 0,
        }
    }

    pub fn legal_moves(self) -> Vec<Move> {
        self.board.legal_moves(self.turn)
    }

    pub fn apply(self, mv: Move) -> Self {
        Self {
            board: self.board.apply(self.turn, mv),
            turn: self.turn.opponent(),
            halfmoves: self.halfmoves + 1,
        }
    }

    pub fn result(self) -> Option<GameResult> {
        if self.halfmoves >= Self::DRAW_HALFMOVES {
            return Some(GameResult::Draw);
        }

        if self.board.black == 0 {
            return Some(GameResult::Winner(Side::Red));
        }

        if self.board.red == 0 {
            return Some(GameResult::Winner(Side::Black));
        }

        if self.legal_moves().is_empty() {
            return Some(GameResult::Winner(self.turn.opponent()));
        }

        None
    }
}

impl Default for BitBoard {
    fn default() -> Self {
        Self::starting_position()
    }
}

impl BitBoard {
    pub const BLACK_START: u32 = 0xfff0_0000;
    pub const RED_START: u32 = 0x0000_0fff;

    #[cfg(test)]
    pub const fn empty() -> Self {
        Self {
            black: 0,
            red: 0,
            kings: 0,
        }
    }

    pub const fn starting_position() -> Self {
        Self {
            black: Self::BLACK_START,
            red: Self::RED_START,
            kings: 0,
        }
    }

    pub fn occupied(self) -> u32 {
        self.black | self.red
    }

    pub fn piece_at(self, row: usize, col: usize) -> Option<Piece> {
        let square = square_index(row, col)?;
        let mask = 1u32 << square;
        let king = self.kings & mask != 0;

        if self.black & mask != 0 {
            Some(if king {
                Piece::BlackKing
            } else {
                Piece::BlackMan
            })
        } else if self.red & mask != 0 {
            Some(if king { Piece::RedKing } else { Piece::RedMan })
        } else {
            None
        }
    }

    pub fn side_bits(self, side: Side) -> u32 {
        match side {
            Side::Black => self.black,
            Side::Red => self.red,
        }
    }

    pub fn legal_moves(self, side: Side) -> Vec<Move> {
        let captures = self.capture_moves(side);
        if captures.is_empty() {
            self.quiet_moves(side)
        } else {
            captures
        }
    }

    pub fn apply(self, side: Side, mv: Move) -> Self {
        let from_mask = 1u32 << mv.from;
        let to_mask = 1u32 << mv.to;
        let moved_king = self.kings & from_mask != 0;

        let mut next = self;
        match side {
            Side::Black => next.black = (next.black & !from_mask) | to_mask,
            Side::Red => next.red = (next.red & !from_mask) | to_mask,
        }
        next.kings &= !from_mask;

        for captured in mv.captures.iter().flatten() {
            let mask = 1u32 << captured;
            next.black &= !mask;
            next.red &= !mask;
            next.kings &= !mask;
        }

        if moved_king || promotes(side, mv.to) {
            next.kings |= to_mask;
        }

        next
    }

    pub fn perft(self, side: Side, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = self.legal_moves(side);
        if moves.is_empty() {
            return 0;
        }

        moves
            .into_iter()
            .map(|mv| self.apply(side, mv).perft(side.opponent(), depth - 1))
            .sum()
    }

    fn quiet_moves(self, side: Side) -> Vec<Move> {
        let mut moves = Vec::with_capacity(16);
        let own = self.side_bits(side);
        let occupied = self.occupied();

        for from in bit_indices(own) {
            let king = self.kings & (1u32 << from) != 0;
            for &(dr, dc) in move_dirs(side, king) {
                if let Some(to) = offset_square(from, dr, dc) {
                    let to_mask = 1u32 << to;
                    if occupied & to_mask == 0 {
                        moves.push(Move::quiet(from, to));
                    }
                }
            }
        }

        moves
    }

    fn capture_moves(self, side: Side) -> Vec<Move> {
        let mut moves = Vec::with_capacity(8);
        for from in bit_indices(self.side_bits(side)) {
            let mut path = Move::quiet(from, from);
            self.collect_captures(side, from, from, &mut path, &mut moves);
        }
        moves
    }

    fn collect_captures(
        self,
        side: Side,
        origin: u8,
        from: u8,
        path: &mut Move,
        moves: &mut Vec<Move>,
    ) {
        let king = self.kings & (1u32 << from) != 0;
        let enemy = self.side_bits(side.opponent());
        let mut extended = false;

        for &(dr, dc) in move_dirs(side, king) {
            let Some(mid) = offset_square(from, dr, dc) else {
                continue;
            };
            let Some(to) = offset_square(from, dr * 2, dc * 2) else {
                continue;
            };

            let mid_mask = 1u32 << mid;
            let to_mask = 1u32 << to;
            if enemy & mid_mask == 0 || self.occupied() & to_mask != 0 {
                continue;
            }

            let next = self.apply(
                side,
                Move {
                    from,
                    to,
                    captures: [Some(mid), None, None, None, None, None, None],
                    capture_len: 1,
                },
            );

            path.to = to;
            path.captures[path.capture_len as usize] = Some(mid);
            path.capture_len += 1;
            next.collect_captures(side, origin, to, path, moves);
            path.capture_len -= 1;
            path.captures[path.capture_len as usize] = None;
            path.to = from;
            extended = true;
        }

        if !extended && path.capture_len > 0 {
            path.from = origin;
            moves.push(*path);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub captures: [Option<u8>; 7],
    pub capture_len: u8,
}

impl Move {
    pub const fn quiet(from: u8, to: u8) -> Self {
        Self {
            from,
            to,
            captures: [None; 7],
            capture_len: 0,
        }
    }

    pub fn is_capture(self) -> bool {
        self.capture_len > 0
    }

    pub fn notation(self) -> String {
        let separator = if self.is_capture() { "x" } else { "-" };
        format!("{}{}{}", notation(self.from), separator, notation(self.to))
    }
}

pub fn square_index(row: usize, col: usize) -> Option<u8> {
    if row >= 8 || col >= 8 || (row + col).is_multiple_of(2) {
        return None;
    }
    Some((row * 4 + col / 2) as u8)
}

pub fn square_coords(square: u8) -> (usize, usize) {
    let row = square as usize / 4;
    let dark = square as usize % 4;
    let col = if row.is_multiple_of(2) {
        dark * 2 + 1
    } else {
        dark * 2
    };
    (row, col)
}

pub fn notation(square: u8) -> String {
    let (row, col) = square_coords(square);
    let file = (b'a' + col as u8) as char;
    let rank = 8 - row;
    format!("{file}{rank}")
}

fn bit_indices(mut bits: u32) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if bits == 0 {
            return None;
        }
        let index = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        Some(index)
    })
}

fn move_dirs(side: Side, king: bool) -> &'static [(i8, i8)] {
    match (side, king) {
        (Side::Black, false) => &[(-1, -1), (-1, 1)],
        (Side::Red, false) => &[(1, -1), (1, 1)],
        (_, true) => &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
    }
}

fn offset_square(square: u8, dr: i8, dc: i8) -> Option<u8> {
    let (row, col) = square_coords(square);
    let row = row as i8 + dr;
    let col = col as i8 + dc;
    if row < 0 || col < 0 {
        return None;
    }
    square_index(row as usize, col as usize)
}

fn promotes(side: Side, square: u8) -> bool {
    let (row, _) = square_coords(square);
    matches!((side, row), (Side::Black, 0) | (Side::Red, 7))
}

pub fn piece_side(piece: Piece) -> Side {
    match piece {
        Piece::BlackMan | Piece::BlackKing => Side::Black,
        Piece::RedMan | Piece::RedKing => Side::Red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_has_expected_piece_counts() {
        let board = BitBoard::starting_position();
        assert_eq!(board.black.count_ones(), 12);
        assert_eq!(board.red.count_ones(), 12);
        assert_eq!(board.kings.count_ones(), 0);
    }

    #[test]
    fn starting_position_has_seven_black_moves() {
        let board = BitBoard::starting_position();
        let moves = board.legal_moves(Side::Black);
        assert_eq!(moves.len(), 7);
        assert!(moves.iter().all(|mv| !mv.is_capture()));
    }

    #[test]
    fn forced_capture_hides_quiet_moves() {
        let mut board = BitBoard::empty();
        board.black = 1u32 << square_index(5, 0).unwrap();
        board.red = 1u32 << square_index(4, 1).unwrap();

        let moves = board.legal_moves(Side::Black);
        assert_eq!(moves.len(), 1);
        assert!(moves[0].is_capture());
        assert_eq!(moves[0].to, square_index(3, 2).unwrap());
    }

    #[test]
    fn perft_depth_one_matches_starting_moves() {
        assert_eq!(BitBoard::starting_position().perft(Side::Black, 1), 7);
    }

    #[test]
    fn game_result_detects_missing_side() {
        let mut game = GameState::new();
        game.board.red = 0;
        assert_eq!(game.result(), Some(GameResult::Winner(Side::Black)));
    }
}
