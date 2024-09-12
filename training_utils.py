import os, sys
chapter = "chapter1_transformer_interp"
repo = "ARENA_3.0"
chapter_dir = r"./" if chapter in os.listdir() else os.getcwd().split(chapter)[0]
sys.path.append(chapter_dir + f"{chapter}/exercises")

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
import itertools
import random
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from typing import List, Union, Optional, Tuple, Callable, Dict
import typeguard
from functools import partial
import copy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import utils

from utils import seq_to_state_stack

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

model = utils.load_model(device)

# os.chdir(section_dir)
section_dir = Path.cwd()
assert section_dir.name == "interpretability"

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

sys.path.append(str(OTHELLO_MECHINT_ROOT))

from mech_interp_othello_utils import (
    OthelloBoardState,
    to_string,
    to_int,
)

# Load board data as ints (i.e. 0 to 60)
board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)

assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
assert board_seqs_int.max() == 60

rows = 8
cols = 8
options = 3

BLANK = 0
FILLED_UNAFFECTED = 1
FLIPPED = 2
PLAYED_BLACK = 3
PLAYED_WHITE = 4
UNAFFECTED = 5
ACCESIBLE = 1
LEGAL = 2

'''EMPTY = 0
YOURS = 2
MINE = 1

FLIPPED_TOP = 0
FLIPPED_TOP_RIGHT = 1
FLIPPED_RIGHT = 2
FLIPPED_BOTTOM_RIGHT = 3
FLIPPED_BOTTOM = 4
FLIPPED_BOTTOM_LEFT = 5
FLIPPED_LEFT = 6
FLIPPED_TOP_LEFT = 7'''

from utils import *

from typing import List, Tuple

def get_state_stack_empty_yours_mine(games_str : Int[Tensor, "num_games len_of_game"]):
    num_games, len_of_game = games_str.shape
    game_states = []
    for game_idx in range(num_games):
        game_state = seq_to_state_stack(games_str[game_idx])
        game_states.append(game_state)
    game_states = t.Tensor(np.stack(game_states)) # 0 is blank, -1 is white, 1 is black
    game_states = game_states.to(t.long)
    assert game_states.shape == (num_games, len_of_game, rows, cols)
    game_states[:, 1::2] *= -1 # 0 blank, 1 mine, -1 theirs
    game_states[game_states == -1] = MINE # 2
    game_states[game_states == 1] = YOURS # 1
    return game_states # 0 empty, 1 yours, 2 mine

def get_state_stack_one_hot_empty_yours_mine(games_str : Int[Tensor, "num_games len_of_game"]):
    game_states = get_state_stack_empty_yours_mine(games_str)
    game_states_one_hot = t.nn.functional.one_hot(game_states, num_classes=3).to(device)
    return game_states_one_hot

def state_stack_to_one_hot(state_stack):
    '''
    Creates a tensor of shape (games, moves, rows=8, cols=8, options=3), where the [g, m, r, c, :]-th entry
    is a one-hot encoded vector for the state of game g at move m, at row r and column c. In other words, this
    vector equals (1, 0, 0) when the state is empty, (0, 1, 0) when the state is "their", and (0, 0, 1) when the
    state is "my".
    '''
    one_hot = t.zeros(
        state_stack.shape[0], # num games
        state_stack.shape[1], # num moves
        rows,
        cols,
        3, # the options: empty, white, or black
        device=state_stack.device,
        dtype=t.int,
    )
    one_hot[..., 0] = state_stack == 0
    one_hot[..., 1] = state_stack == -1
    one_hot[..., 2] = state_stack == 1

    return one_hot


def seq_to_state_stack_flipped(str_moves):
    """
    Collects states, where each cell mean the following:
    0: blank
    1: unaffected
    2: flipped
    3: played_black
    4: played_white
    """
    if isinstance(str_moves, t.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move_idx, move in enumerate(str_moves):
        player = "black" if move_idx % 2 == 0 else "white"
        try:
            flipped = board.umpire_return_flipped(move)
        except RuntimeError:
            breakpoint()

        _state = np.copy(board.state)
        _state[:, :] = _state[:, :] != 0

        # Update Flipped cells
        for cell in flipped:
            _state[cell[0], cell[1]] = FLIPPED

        # Update played cell
        row, col = move // 8, move % 8
        if player == "black":
            _state[row, col] = PLAYED_BLACK
        else:
            _state[row, col] = PLAYED_WHITE

        states.append(_state)
    states = np.stack(states, axis=0)
    return states


def seq_to_state_stack_legal(str_moves):
    """
    0: blank
    1: unaffected
    2: accesible
    3: legal
    """
    if isinstance(str_moves, t.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move_idx, move in enumerate(str_moves):
        player = "black" if move_idx % 2 == 0 else "white"
        try:
            board.umpire(move)
        except RuntimeError:
            breakpoint()
    _state = np.copy(board.state)
    print(_state)
        
"""board_seqs_string = t.load(
    os.path.join(
        section_dir,
        "data/board_seqs_string_train.pth",
    )
)
games_str = board_seqs_string[:10]
assert games_str.shape == (10, 60)

state_stack_legal = seq_to_state_stack_legal(games_str)"""


'''state_one_hot = get_state_stack_one_hot_first_tile_places_black_white(games_str)
assert state_one_hot.shape == (10, 60, 8, 8, 3)
# print(state_one_hot[0, 15])

state_one_hot = get_state_stack_one_hot_first_tile_places_mine_theirs(games_str)
assert state_one_hot.shape == (10, 60, 8, 8, 3)
# print(state_one_hot[0, 15])

from utils import plot_game
plot_game(games_str, 0)'''


def seq_to_state_stack_placed_and_flipped(str_moves):
    """
    Collects states, where each cell mean the following:
    0: blank
    1: unaffected
    2: flipped
    3: played_black
    4: played_white
    """
    if isinstance(str_moves, t.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move_idx, move in enumerate(str_moves):
        try:
            flipped = board.umpire_return_flipped(move)
        except RuntimeError:
            breakpoint()

        _state = np.copy(board.state)
        _state[:, :] = _state[:, :] != 0

        # Update Flipped cells
        for cell in flipped:
            _state[cell[0], cell[1]] = FLIPPED

        # Update played cell
        row, col = move // 8, move % 8
        
        _state_one_hot = np.zeros((8, 8, 8))

        if _state[max(row-1, 0), col] == FLIPPED:
            _state_one_hot[row, col, FLIPPED_TOP] = 1
        if _state[max(row-1, 0), min(col+1, 7)] == FLIPPED:
            _state_one_hot[row, col, FLIPPED_TOP_RIGHT] = 1
        if _state[row, min(col+1, 7)] == FLIPPED:
            _state_one_hot[row, col, FLIPPED_RIGHT] = 1
        if _state[min(row+1, 7), min(col+1, 7)] == FLIPPED:
            _state_one_hot[row, col, FLIPPED_BOTTOM_RIGHT] = 1
        if _state[min(row+1, 7), col] == FLIPPED:
            _state_one_hot[row, col, FLIPPED_BOTTOM] = 1
        if _state[min(row+1, 7), max(col-1, 0)] == FLIPPED:
            _state_one_hot[row, col, FLIPPED_BOTTOM_LEFT] = 1
        if _state[row, max(col-1, 0)] == FLIPPED:
            _state_one_hot[row, col, FLIPPED_LEFT] = 1
        if _state[max(row-1, 0), max(col-1, 0)] == FLIPPED:
            _state_one_hot[row, col, FLIPPED_TOP_LEFT] = 1

        states.append(_state_one_hot)
    states = np.stack(states, axis=0)
    return states

def get_diagonal(y, x, up : bool):
    if up:
        return [(y - i, x + i) for i in range(max(-x, y - 7), min(y, 7 - x) + 1)]
    else:
        return [(y + i, x + i) for i in range(max(-x, -y), min(7 - y, 7 - x) + 1)]

def seq_to_state_stack_placed_and_flipped_stripe(str_moves):
    """
    Collects states, where each cell mean the following:
    0: blank
    1: unaffected
    2: flipped
    3: played_black
    4: played_white
    """
    if isinstance(str_moves, t.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move_idx, move in enumerate(str_moves):
        try:
            flipped = board.umpire_return_flipped(move)
        except RuntimeError:
            breakpoint()

        _state = np.copy(board.state)
        _state[:, :] = _state[:, :] != 0

        # Update Flipped cells
        for cell in flipped:
            _state[cell[0], cell[1]] = FLIPPED

        # Update played cell
        row, col = move // 8, move % 8

        _state_one_hot = np.zeros((8, 8, 8))
        diagonal_up = get_diagonal(row, col, up=True)
        diagonal_down = get_diagonal(row, col, up=False)

        if _state[max(row-1, 0), col] == FLIPPED:
            _state_one_hot[:, col, FLIPPED_TOP] = 1
        if _state[max(row-1, 0), min(col+1, 7)] == FLIPPED:
            for y, x in diagonal_up:
                _state_one_hot[y, x, FLIPPED_TOP_RIGHT] = 1
        if _state[row, min(col+1, 7)] == FLIPPED:
            _state_one_hot[row, :, FLIPPED_RIGHT] = 1
        if _state[min(row+1, 7), min(col+1, 7)] == FLIPPED:
            for y, x in diagonal_down:
                _state_one_hot[y, x, FLIPPED_BOTTOM_RIGHT] = 1
        if _state[min(row+1, 7), col] == FLIPPED:
            _state_one_hot[:, col, FLIPPED_BOTTOM] = 1
        if _state[min(row+1, 7), max(col-1, 0)] == FLIPPED:
            for y, x in diagonal_up:
                _state_one_hot[y, x, FLIPPED_BOTTOM_LEFT] = 1
            _state_one_hot[row, col, FLIPPED_BOTTOM_LEFT] = 1
        if _state[row, max(col-1, 0)] == FLIPPED:
            _state_one_hot[row, :, FLIPPED_LEFT] = 1
        if _state[max(row-1, 0), max(col-1, 0)] == FLIPPED:
            for y, x in diagonal_up:
                _state_one_hot[y, x, FLIPPED_TOP_LEFT] = 1

        states.append(_state_one_hot)
    states = np.stack(states, axis=0)
    return states

'''
if _state[max(row-1, 0), col] == FLIPPED:
    _state[max(row-1, 0), col] = FLIPPED_BOTTOM
if _state[max(row-1, 0), min(col+1, 7)] == FLIPPED:
    _state[max(row-1, 0), min(col+1, 7)] = FLIPPED_BOTTOM_LEFT
if _state[row, min(col+1, 7)] == FLIPPED:
    _state[row, min(col+1, 7)] = FLIPPED_LEFT
if _state[min(row+1, 7), min(col+1, 7)] == FLIPPED:
    _state[min(row+1, 7), min(col+1, 7)] = FLIPPED_TOP_LEFT
if _state[min(row+1, 7), col] == FLIPPED:
    _state[min(row+1, 7), col] = FLIPPED_TOP
if _state[min(row+1, 7), max(col-1, 0)] == FLIPPED:
    _state[min(row+1, 7), max(col-1, 0)] = FLIPPED_TOP_RIGHT
if _state[row, max(col-1, 0)] == FLIPPED:
    _state[row, max(col-1, 0)] = FLIPPED_RIGHT
if _state[max(row-1, 0), max(col-1, 0)] == FLIPPED:
    _state[max(row-1, 0), max(col-1, 0)] = FLIPPED_BOTTOM_RIGHT
'''

def build_state_stack_flipped(board_seqs_string):
    """
    Construct stack of board-states.
    This function will also filter out corrputed game-sequences.
    """
    state_stack = []
    for idx, seq in enumerate(board_seqs_string):
        _stack = seq_to_state_stack_flipped(seq)
        state_stack.append(_stack)
    return t.tensor(np.stack(state_stack))


def state_stack_to_one_hot_flipped(state_stack):
    one_hot = t.zeros(
        state_stack.shape[0],
        state_stack.shape[1],
        8,  # rows
        8,  # cols
        2,  # options
        device=state_stack.device,
        dtype=t.int,
    )  # [batch_size, 59, 8, 8, 4]

    # Flipped
    one_hot[..., 0] = state_stack == FLIPPED
    one_hot[..., 1] = 1 - one_hot[..., 0]
    return one_hot

def state_stack_to_one_hot_placed(state_stack):
    one_hot = t.zeros(
        state_stack.shape[0],
        state_stack.shape[1],
        8,  # rows
        8,  # cols
        2,  # options
        device=state_stack.device,
        dtype=t.int,
    )  # [batch_size, 59, 8, 8, 4]

    # Flipped
    one_hot[..., 0] = (state_stack == PLAYED_BLACK) | (state_stack == PLAYED_WHITE)
    one_hot[..., 1] = 1 - one_hot[..., 0]
    return one_hot

'''def state_stack_to_one_hot_placed_and_flipped(state_stack):
    one_hot = t.zeros(
        state_stack.shape[0],
        state_stack.shape[1],
        8,  # rows
        8,  # cols
        9,  # options
        device=state_stack.device,
        dtype=t.int,
    )  # [batch_size, 59, 8, 8, 4]

    # Flipped
    one_hot[..., 0] = state_stack < 3
    one_hot[..., 1] = state_stack == FLIPPED_TOP
    one_hot[..., 2] = state_stack == FLIPPED_TOP_RIGHT
    one_hot[..., 3] = state_stack == FLIPPED_RIGHT
    one_hot[..., 4] = state_stack == FLIPPED_BOTTOM_RIGHT
    one_hot[..., 5] = state_stack == FLIPPED_BOTTOM
    one_hot[..., 6] = state_stack == FLIPPED_BOTTOM_LEFT
    one_hot[..., 7] = state_stack == FLIPPED_LEFT
    one_hot[..., 8] = state_stack == FLIPPED_TOP_LEFT
    return one_hot'''

def get_state_stack_one_hot_general(get_state_stack_specific, state_stack_to_one_hot_specific):
    def get_state_stack_one_hot(games_str : Int[Tensor, "num_games len_of_game"]):
        state_stacks = []
        for batch in range(games_str.shape[0]):
            state_stack = get_state_stack_specific(games_str[batch])
            state_stacks.append(state_stack)
        state_stack = t.Tensor(np.stack(state_stacks))
        state_stack_one_hot = state_stack_to_one_hot_specific(
            state_stack
        ).to(device)
        return state_stack_one_hot
    return get_state_stack_one_hot

def get_state_stack_one_hot_placed_and_flipped(games_str : Int[Tensor, "num_games len_of_game"]):
    state_stacks_one_hot = []
    for batch in range(games_str.shape[0]):
        state_stack = seq_to_state_stack_placed_and_flipped(games_str[batch])
        state_stacks_one_hot.append(state_stack)
    state_stacks_one_hot = t.Tensor(np.stack(state_stacks_one_hot)).to(device)
    return state_stacks_one_hot

def get_state_stack_one_hot_placed_and_flipped_stripe(games_str : Int[Tensor, "num_games len_of_game"]):
    state_stacks_one_hot = []
    for batch in range(games_str.shape[0]):
        state_stack = seq_to_state_stack_placed_and_flipped_stripe(games_str[batch])
        state_stacks_one_hot.append(state_stack)
    state_stacks_one_hot = t.Tensor(np.stack(state_stacks_one_hot)).to(device)
    return state_stacks_one_hot

# Create all the get_state_stack_one_hot_... functions below using teh get_state_stack_one_hot_general function
# get_state_stack_one_hot = get_state_stack_one_hot_general(get_state_stack_empty_yours_mine, state_stack_to_one_hot)
get_state_stack_one_hot_flipped = get_state_stack_one_hot_general(build_state_stack_flipped, state_stack_to_one_hot_flipped)
get_state_stack_one_hot_placed = get_state_stack_one_hot_general(seq_to_state_stack_flipped, state_stack_to_one_hot_placed)
# get_state_stack_one_hot_placed_and_flipped = get_state_stack_one_hot_general(seq_to_state_stack_placed_and_flipped, state_stack_to_one_hot_placed_and_flipped)
# get_state_stack_one_hot_num_flipped = get_state_stack_one_hot_general(get_state_stack_num_flipped, state_stack_to_one_hot_flipped)

'''def get_state_stack_one_hot_flipped(games_str : Int[Tensor, "num_games len_of_game"]):
    """
    Returns a tensor of shape (games, moves, rows=8, cols=8, flipped=2) where the last dimension is a one-hot encoding
    of whether the tile was flipped or not.
    """
    state_stack = build_state_stack_flipped(games_str)
    state_stack_one_hot = state_stack_to_one_hot_flipped(
        state_stack
    ).to(device)
    return state_stack_one_hot'''

def get_state_stack_num_flipped(games_str : Int[Tensor, "num_games len_of_game"]):
    state_stack = build_state_stack_flipped(games_str)
    # turn 2 (Flipped) into 1, everything else to 0
    # Not the same game that gets plotted
    state_stack = state_stack == 2
    state_stack_num_flipped = []
    for idx in range(state_stack.shape[1]):
        _state = einops.reduce(state_stack[:, :idx+1, :, :], 'num_games moves rows cols -> num_games rows cols', 'sum')
        state_stack_num_flipped.append(_state)
    state_stack_num_flipped = t.stack(state_stack_num_flipped, dim=1)
    # if only_middle:
    #     state_stack_num_flipped = state_stack_num_flipped[:, args.pos_start:args.pos_end, :, :]
    return state_stack_num_flipped


# This function takes game indeces as input and returns a tensor of shape (games, moves, rows=8, cols=8, flipped=18)
def get_state_stack_one_hot_num_flipped(games_str : Int[Tensor, "num_games len_of_game"]):
    """
    Returns a tensor of shape (games, moves, rows=8, cols=8, flipped=2) where the last dimension is a one-hot encoding
    of how often the tile was flipped
    """
    # (num_games, moves, rows, cols) numbers from 0 to 18
    state_stack_num_flipped = get_state_stack_num_flipped(games_str)
    state_stack_one_hot_num_flipped = t.nn.functional.one_hot(
        state_stack_num_flipped, num_classes=18
    ).to(device)
    return state_stack_one_hot_num_flipped


def get_state_stack_one_hot_even_odd_flipped(games_str : Int[Tensor, "num_games len_of_game"]):
    """
    Returns a tensor of shape (games, moves, rows=8, cols=8, flipped=2) where the last dimension is a one-hot encoding
    of how often the tile was flipped
    """
    # (num_games, moves, rows, cols) numbers from 0 to 18
    state_stack_even_odd_flipped = get_state_stack_num_flipped(games_str)
    state_stack_even_odd_flipped = state_stack_even_odd_flipped % 2
    state_stack_one_hot_even_odd_flipped = t.nn.functional.one_hot(
        state_stack_even_odd_flipped, num_classes=2
    ).to(device)
    return state_stack_one_hot_even_odd_flipped


def get_state_stack_first_tile_places_black_white(games_str : Int[Tensor, "num_games len_of_game"]):
    """
    Returns a tensor of shape (games, moves, rows=8, cols=8, flipped=2) where the last dimension is a one-hot encoding
    of what the first tile placed was
    """
    # (num_games, moves, rows, cols) numbers from -1 to 1
    state_stack = build_state_stack_flipped(games_str).to(dtype=t.long)
    state_stack_first_tile_places_black_white = []
    _state = t.zeros(state_stack.shape[0], state_stack.shape[2], state_stack.shape[3], device=state_stack.device, dtype=t.long)
    _state[:, 3, 3], _state[:, 4, 4], _state[:, 3, 4], _state[:, 4, 3] = 2, 2, 1, 1

    for idx in range(0, state_stack.shape[1]):
        new_state = state_stack[:, idx, :, :]
        # TODO: This doesen't work plus figure out what device everything should be one plus debug the function not the complete script ...
        new_state[new_state == 1], new_state[new_state == 2], new_state[new_state == 5] = 0, 0, 0
        # black
        new_state[new_state == 3] = 1
        # white
        new_state[new_state == 4] = 2
        _state = _state + new_state
        state_stack_first_tile_places_black_white.append(_state)
    state_stack_first_tile_places_black_white = t.stack(state_stack_first_tile_places_black_white, dim=1)
    return state_stack_first_tile_places_black_white.to(dtype=t.long)

def get_state_stack_one_hot_first_tile_places_black_white(games_str : Int[Tensor, "num_games len_of_game"]):
    state_stack_first_tile_places_black_white = get_state_stack_first_tile_places_black_white(games_str)
    # print(state_stack_first_tile_places_black_white[0, 14])
    state_stack_one_hot_first_tile_places_black_white = t.nn.functional.one_hot(
        state_stack_first_tile_places_black_white, num_classes=3
    ).to(device)
    return state_stack_one_hot_first_tile_places_black_white


def get_state_stack_first_tile_places_mine_theirs(games_str : Int[Tensor, "num_games len_of_game"]):
    state_stack_first_tile_places_mine_theirs = get_state_stack_first_tile_places_black_white(games_str)
    state_stack_first_tile_places_mine_theirs[:, 1::2, :, :] *= -1
    state_stack_first_tile_places_mine_theirs[state_stack_first_tile_places_mine_theirs == 2], state_stack_first_tile_places_mine_theirs[state_stack_first_tile_places_mine_theirs == 1] = 1, 2
    state_stack_first_tile_places_mine_theirs[state_stack_first_tile_places_mine_theirs == -2], state_stack_first_tile_places_mine_theirs[state_stack_first_tile_places_mine_theirs == -1] = 2, 1
    return state_stack_first_tile_places_mine_theirs

def get_state_stack_one_hot_first_tile_places_mine_theirs(games_str : Int[Tensor, "num_games len_of_game"]):
    state_stack_first_tile_places_mine_theirs = get_state_stack_first_tile_places_mine_theirs(games_str)
    # print(state_stack_first_tile_places_mine_theirs[0, 14])
    state_stack_one_hot_first_tile_places_mine_theirs = t.nn.functional.one_hot(
        state_stack_first_tile_places_mine_theirs, num_classes=3
    ).to(device)
    return state_stack_one_hot_first_tile_places_mine_theirs

'''board_seqs_string = t.load(
    os.path.join(
        section_dir,
        "data/board_seqs_string_train.pth",
    )
)
games_str = board_seqs_string[:10]
assert games_str.shape == (10, 60)

state_one_hot = get_state_stack_one_hot_first_tile_places_black_white(games_str)
assert state_one_hot.shape == (10, 60, 8, 8, 3)
# print(state_one_hot[0, 15])

state_one_hot = get_state_stack_one_hot_first_tile_places_mine_theirs(games_str)
assert state_one_hot.shape == (10, 60, 8, 8, 3)
# print(state_one_hot[0, 15])

from utils import plot_game
plot_game(games_str, 0)'''



WHITE = -1
BLACK = 1
BLANK = 0
ACCESIBLE = 2
LEGAL = 3

def is_legal_or_accesible(_state, row, col, player):
    is_accesible = False
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if r < 0 or r >= 8 or c < 0 or c >= 8:
                continue
            if _state[r, c] == -player or _state[r, c] == player:
                is_accesible = True
            if _state[r, c] == player:
                while True:
                    r += dr
                    c += dc
                    if r < 0 or r >= 8 or c < 0 or c >= 8:
                        break
                    if _state[r, c] == BLANK:
                        break
                    if _state[r, c] == -player:
                        return LEGAL
    if is_accesible:
        return ACCESIBLE
    return BLANK
                    

def seq_to_state_stack_legal(str_moves):
    """
    0: blank
    1: unaffected
    2: accesible
    3: legal
    """
    if isinstance(str_moves, t.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move_idx, move in enumerate(str_moves):
        # The Player who just played
        player = BLACK if move_idx % 2 == 0 else WHITE
        try:
            board.umpire(move)
        except RuntimeError:
            breakpoint()
        _state = np.copy(board.state)
        # Do Accessible
        for row in range(8):
            for col in range(8):
                if _state[row, col] != BLANK:
                    continue
                _state[row, col] = is_legal_or_accesible(_state, row, col, player)
        # _state = np.abs(_state)
        states.append(_state)
    states = np.stack(states, axis=0)
    return t.tensor(states)

def build_state_stack_legal(board_seqs_string):
    """
    Construct stack of board-states.
    This function will also filter out corrputed game-sequences.
    """
    state_stack = []
    for idx, seq in enumerate(board_seqs_string):
        _stack = seq_to_state_stack_legal(seq)
        state_stack.append(_stack)
    return t.tensor(np.stack(state_stack))

def state_stack_to_one_hot_accesible(state_stack):
    one_hot = t.zeros(
        state_stack.shape[0],
        state_stack.shape[1],
        8,  # rows
        8,  # cols
        2,  # options
        device=state_stack.device,
        dtype=t.int,
    )  # [batch_size, 59, 8, 8, 4]

    # Accesible
    one_hot[..., 0] = state_stack >= ACCESIBLE
    one_hot[..., 1] = 1 - one_hot[..., 0]
    return one_hot

def state_stack_to_one_hot_legal(state_stack):
    one_hot = t.zeros(
        state_stack.shape[0],
        state_stack.shape[1],
        8,  # rows
        8,  # cols
        2,  # options
        device=state_stack.device,
        dtype=t.int,
    )  # [batch_size, 59, 8, 8, 4]

    # Accesible
    one_hot[..., 0] = state_stack == LEGAL
    one_hot[..., 1] = 1 - one_hot[..., 0]
    return one_hot


get_state_stack_one_hot_accesible = get_state_stack_one_hot_general(seq_to_state_stack_legal, state_stack_to_one_hot_accesible)
get_state_stack_one_hot_legal = get_state_stack_one_hot_general(seq_to_state_stack_legal, state_stack_to_one_hot_legal)