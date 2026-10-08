"""Random legal Othello games in the format OthelloGPT was trained on.

The original repo loads `data/board_seqs_int_train.pth` and `data/board_seqs_int_valid.pth`,
which are gitignored and not available any more. OthelloGPT's "synthetic" training data are
games where every move is drawn uniformly from the legal moves, so we generate fresh games
of that kind. We keep only games that last the full 60 moves, like the original dataset.

Squares are numbered 0..63 row-major, where row 0 is "A" and column 0 is "0" (so D3 = 27).
The model's tokens ("int" format) are 1..60: the 60 squares without the four centre squares,
in increasing order. Token 0 is unused.
"""

import numpy as np

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
GAME_LENGTH = 60
CENTRE_SQUARES = [27, 28, 35, 36]
TOKEN_SQUARES = [square for square in range(NUM_SQUARES) if square not in CENTRE_SQUARES]
SQUARE_TO_TOKEN = {square: token + 1 for token, square in enumerate(TOKEN_SQUARES)}

BLACK = 1
WHITE = -1
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def initial_board() -> np.ndarray:
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[3, 3] = board[4, 4] = WHITE
    board[3, 4] = board[4, 3] = BLACK
    return board


def tiles_flipped_by(board: np.ndarray, row: int, col: int, player: int) -> list[tuple[int, int]]:
    if board[row, col] != 0:
        return []
    flipped = []
    for row_delta, col_delta in DIRECTIONS:
        line = []
        r, c = row + row_delta, col + col_delta
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == -player:
            line.append((r, c))
            r, c = r + row_delta, c + col_delta
        if line and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
            flipped += line
    return flipped


def legal_moves(board: np.ndarray, player: int) -> list[int]:
    return [
        square
        for square in range(NUM_SQUARES)
        if tiles_flipped_by(board, square // BOARD_SIZE, square % BOARD_SIZE, player)
    ]


def random_game(rng: np.random.Generator) -> list[int] | None:
    """Plays one game with uniformly random legal moves. Returns None if it ends before 60 moves."""
    board = initial_board()
    player = BLACK
    moves = []
    while len(moves) < GAME_LENGTH:
        options = legal_moves(board, player)
        if not options:
            player = -player
            options = legal_moves(board, player)
            if not options:
                return None
        square = options[rng.integers(len(options))]
        row, col = square // BOARD_SIZE, square % BOARD_SIZE
        for r, c in tiles_flipped_by(board, row, col, player):
            board[r, c] = player
        board[row, col] = player
        moves.append(square)
        player = -player
    return moves


def generate_games(num_games: int, seed: int) -> np.ndarray:
    """Returns an array of shape [num_games, 60] with squares 0..63."""
    rng = np.random.default_rng(seed)
    games = []
    while len(games) < num_games:
        game = random_game(rng)
        if game is not None:
            games.append(game)
    return np.array(games, dtype=np.int64)


def squares_to_tokens(games: np.ndarray) -> np.ndarray:
    lookup = np.zeros(NUM_SQUARES, dtype=np.int64)
    for square, token in SQUARE_TO_TOKEN.items():
        lookup[square] = token
    return lookup[games]


def game_labels(moves: list[int]) -> dict[str, np.ndarray]:
    """Ground-truth labels after each move, with the same conventions as Jim's probes.

    - `board`: 1 for the colour of the player who just moved ("yours" in Jim's probes),
      -1 for the other colour ("mine"), 0 for empty. Like Jim's probe training, the player who
      just moved is taken from the move's parity, which is wrong after a pass. Passes are rare.
    - `flipped`: 1 where this move flipped a tile.
    - `placed`: 1 on the square where this move put its tile.
    """
    board = initial_board()
    boards, flipped, placed = [], [], []
    for move_index, square in enumerate(moves):
        row, col = square // BOARD_SIZE, square % BOARD_SIZE
        player = BLACK if move_index % 2 == 0 else WHITE
        actual_player = player if tiles_flipped_by(board, row, col, player) else -player
        flipped_tiles = tiles_flipped_by(board, row, col, actual_player)
        for r, c in flipped_tiles:
            board[r, c] = actual_player
        board[row, col] = actual_player
        relative_board = board * player
        flipped_board = np.zeros_like(board)
        for r, c in flipped_tiles:
            flipped_board[r, c] = 1
        placed_board = np.zeros_like(board)
        placed_board[row, col] = 1
        boards.append(relative_board.copy())
        flipped.append(flipped_board)
        placed.append(placed_board)
    return {"board": np.stack(boards), "flipped": np.stack(flipped), "placed": np.stack(placed)}

