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

device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

model = utils.load_model(device)

# os.chdir(section_dir)
section_dir = Path.cwd()
assert section_dir.name == "interpretability"

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

sys.path.append(str(OTHELLO_MECHINT_ROOT))

from mech_interp_othello_utils import (
    OthelloBoardState
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
PLAYED = 3
UNAFFECTED = 4


def seq_to_state_stack(str_moves):
    """
    Takes a sequence of moves and returns a stack of states for each move with dimensions (num_moves, rows, cols)
    """
    board = OthelloBoardState()
    states = []
    for move in str_moves:
        board.umpire(move)
        states.append(np.copy(board.state))
    states = np.stack(states, axis=0)
    return states

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
    3: played
    """
    if isinstance(str_moves, t.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move in str_moves:
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
        _state[row, col] = PLAYED

        states.append(_state)
    states = np.stack(states, axis=0)
    return states


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


# This function takes game indeces as input and returns a tensor of shape (games, moves, rows=8, cols=8, flipped=2)
def get_state_stack_one_hot_flipped(games_str : Int[Tensor, "num_games len_of_game"], args):
    """
    Returns a tensor of shape (games, moves, rows=8, cols=8, flipped=2) where the last dimension is a one-hot encoding
    of whether the tile was flipped or not.
    """
    # state_stack.shape == (num_games, moves, rows, cols)
    # state_stack = t.stack([t.tensor(seq_to_state_stack(game_str.tolist())) for game_str in games_str])
    # state_stack = state_stack[:, args.pos_start: args.pos_end, :, :]
    #state_stack_one_hot = state_stack_to_one_hot(state_stack).to(device)
    state_stack = build_state_stack_flipped(games_str)
    state_stack = state_stack[:, args.pos_start:args.pos_end, :, :]
    state_stack_one_hot = state_stack_to_one_hot_flipped(
        state_stack
    ).cuda()
    return state_stack_one_hot


def get_state_stack_flipped(games_str : Int[Tensor, "num_games len_of_game"], args):
    state_stack = build_state_stack_flipped(games_str)
    # turn 2 (Flipped) into 1, everything else to 0
    # Not the same game that gets plotted
    state_stack = state_stack == 2
    state_stack_num_flipped = []
    for idx in range(state_stack.shape[1]):
        _state = einops.reduce(state_stack[:, :idx+1, :, :], 'num_games moves rows cols -> num_games rows cols', 'sum')
        state_stack_num_flipped.append(_state)
    state_stack_num_flipped = t.stack(state_stack_num_flipped, dim=1)
    state_stack_num_flipped = state_stack_num_flipped[:, args.pos_start:args.pos_end, :, :]
    return state_stack_num_flipped


# This function takes game indeces as input and returns a tensor of shape (games, moves, rows=8, cols=8, flipped=18)
def get_state_stack_one_hot_num_flipped(games_str : Int[Tensor, "num_games len_of_game"], args):
    """
    Returns a tensor of shape (games, moves, rows=8, cols=8, flipped=2) where the last dimension is a one-hot encoding
    of how often the tile was flipped
    """
    # (num_games, moves, rows, cols) numbers from 0 to 18
    state_stack_num_flipped = get_state_stack_flipped(games_str, args)
    state_stack_one_hot_num_flipped = t.nn.functional.one_hot(
        state_stack_num_flipped, num_classes=18
    ).cuda()
    return state_stack_one_hot_num_flipped


def get_state_stack_one_hot_even_odd_flipped(games_str : Int[Tensor, "num_games len_of_game"], args):
    """
    Returns a tensor of shape (games, moves, rows=8, cols=8, flipped=2) where the last dimension is a one-hot encoding
    of how often the tile was flipped
    """
    # (num_games, moves, rows, cols) numbers from 0 to 18
    state_stack_even_odd_flipped = get_state_stack_flipped(games_str, args)
    state_stack_even_odd_flipped = state_stack_even_odd_flipped % 2
    state_stack_one_hot_even_odd_flipped = t.nn.functional.one_hot(
        state_stack_even_odd_flipped, num_classes=2
    ).cuda()
    return state_stack_one_hot_even_odd_flipped