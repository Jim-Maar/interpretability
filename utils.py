import transformer_lens.utils
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from plotly_utils import imshow
import einops
import sys
from pathlib import Path
import numpy as np

import torch as t


# os.chdir(section_dir)
section_dir = Path.cwd()
assert section_dir.name == "interpretability"

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

sys.path.append(str(OTHELLO_MECHINT_ROOT))
from mech_interp_othello_utils import (
    OthelloBoardState,
    to_int,
    to_string
)

# Load Model
def load_model(device):
    cfg = HookedTransformerConfig(
        n_layers = 8,
        d_model = 512,
        d_head = 64,
        n_heads = 8,
        d_mlp = 2048,
        d_vocab = 61,
        n_ctx = 59,
        act_fn="gelu",
        normalization_type="LNPre",
        device=device,
    )
    model = HookedTransformer(cfg)

    sd = transformer_lens.utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
    # champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
    model.load_state_dict(sd)
    return model

alpha = "ABCDEFGH"


def seq_to_state_stack(str_moves):
    """
    Takes a sequence of moves and returns a stack of states for each move with dimensions (num_moves, rows, cols)
    -1 white, 0 blank, 1 black
    """
    board = OthelloBoardState()
    states = []
    for move in str_moves:
        board.umpire(move)
        states.append(np.copy(board.state))
    states = np.stack(states, axis=0)
    return states


# Ploting Functions

def plot_square_as_board(state, diverging_scale=True, **kwargs):
    """Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0"""
    kwargs = {
        "y": [i for i in alpha],
        "x": [str(i) for i in range(8)],
        "color_continuous_scale": "RdBu" if diverging_scale else "Blues",
        "color_continuous_midpoint": 0. if diverging_scale else None,
        "aspect": "equal",
        **kwargs
    }
    imshow(state, **kwargs)

def plot_probe_outputs(focus_cache, linear_probe, layer, game_index, move, **kwargs):
    residual_stream = focus_cache["resid_post", layer][game_index, move]
    # print("residual_stream", residual_stream.shape)
    # probe_out = einops.einsum(residual_stream, linear_probe, "d_model, d_model row col options -> row col options")
    probe_out = einops.einsum(residual_stream, linear_probe, "d_model, modes d_model row col options -> modes row col options")[0]
    '''if move % 2 == 0:
        probe_out = probe_out[0]
    else:
        probe_out = probe_out[1]'''
    probabilities = probe_out.softmax(dim=-1)
    plot_square_as_board(probabilities, facet_col=2, facet_labels=["P(EMPTY)", "P(YOURS)", "P(MINE)"], **kwargs)

def plot_game(games_str, game_index=0, end_move=16):
    '''
    This shows the game the 0'th move is the first move the display shows the board after the move was made
    '''
    focus_states = seq_to_state_stack(games_str[game_index])
    imshow(
        focus_states[:end_move],
        facet_col=0,
        facet_col_wrap=8,
        facet_labels=[f"Move {i}" for i in range(0, end_move)],
        title="First 16 moves of first game",
        color_continuous_scale="Greys",
        y = [i for i in alpha],
    )

def square_to_tuple(square, is_int=False):
    if is_int:
        square = to_string(square)
    row = square // 8
    col = square % 8
    return (row, col)

def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"

def get_focus_games(model = None, device = "cpu"):
    # Load board data as ints (i.e. 0 to 60)
    board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
    # Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
    board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)

    assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
    assert board_seqs_int.max() == 60

    num_games, length_of_game = board_seqs_int.shape
    start = 30000
    num_games = 50
    focus_games_int = board_seqs_int[start : start + num_games]
    focus_games_string = board_seqs_string[start: start + num_games]

    if model is not None:
        focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))
        return focus_games_int, focus_games_string, focus_logits, focus_cache
    return focus_games_int, focus_games_string

def square_tuple_from_square(square : str):
    return (alpha.index(square[0]), int(square[1])) 