#!/usr/bin/env python
# coding: utf-8

# ## Setup

# ### Setup 1

# In[1]:


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
# from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import f1_score as multiclass_f1_score
import copy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm.notebook import tqdm
from dataclasses import dataclass
from rich import print as rprint
import pandas as pd

# Make sure exercises are in the path
# exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
# section_dir = exercises_dir / "part6_othellogpt"
# section_dir = "interpretability"
# if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from neel_plotly import scatter, line
# import part6_othellogpt.tests as tests

t.manual_seed(42)

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# ### Setup 2

# In[2]:


MAIN = __name__ == "__main__"

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

sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
# champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
model.load_state_dict(sd)

# An example input
sample_input = t.tensor([[
    20, 19, 18, 10,  2,  1, 27,  3, 41, 42, 34, 12,  4, 40, 11, 29, 43, 13, 48, 56,
    33, 39, 22, 44, 24,  5, 46,  6, 32, 36, 51, 58, 52, 60, 21, 53, 26, 31, 37,  9,
    25, 38, 23, 50, 45, 17, 47, 28, 35, 30, 54, 16, 59, 49, 57, 14, 15, 55, 7
]]).to(device)

# The argmax of the output (ie the most likely next move from each position)
sample_output = t.tensor([[
    21, 41, 40, 34, 40, 41,  3, 11, 21, 43, 40, 21, 28, 50, 33, 50, 33,  5, 33,  5,
    52, 46, 14, 46, 14, 47, 38, 57, 36, 50, 38, 15, 28, 26, 28, 59, 50, 28, 14, 28,
    28, 28, 28, 45, 28, 35, 15, 14, 30, 59, 49, 59, 15, 15, 14, 15,  8,  7,  8
]]).to(device)

assert (model(sample_input).argmax(dim=-1) == sample_output.to(device)).all()

# os.chdir(section_dir)
section_dir = Path.cwd()
sys.path.append(str(section_dir))
print(section_dir.name)

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

# if not OTHELLO_ROOT.exists():
#     !git clone https://github.com/likenneth/othello_world

sys.path.append(str(OTHELLO_MECHINT_ROOT))

from mech_interp_othello_utils import (
    plot_board,
    plot_single_board,
    plot_board_log_probs,
    to_string,
    to_int,
    int_to_label,
    string_to_label,
    OthelloBoardState
)

# Load board data as ints (i.e. 0 to 60)
board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)

assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
assert board_seqs_int.max() == 60

num_games, length_of_game = board_seqs_int.shape

# Define possible indices (excluding the four center squares)
stoi_indices = [i for i in range(64) if i not in [27, 28, 35, 36]]

# Define our rows, and the function that converts an index into a (row, column) label, e.g. `E2`
alpha = "ABCDEFGH"

def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"

# Get our list of board labels
board_labels = list(map(to_board_label, stoi_indices))
full_board_labels = list(map(to_board_label, range(64)))

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

start = 30000
num_games = 50
focus_games_int = board_seqs_int[start : start + num_games]
focus_games_string = board_seqs_string[start: start + num_games]

focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))
focus_logits.shape

def one_hot(list_of_ints, num_classes=64):
    out = t.zeros((num_classes,), dtype=t.float32)
    out[list_of_ints] = 1.
    return out

focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
focus_valid_moves = t.zeros((num_games, 60, 64), dtype=t.float32)

for i in (range(num_games)):
    board = OthelloBoardState()
    for j in range(60):
        board.umpire(focus_games_string[i, j].item())
        focus_states[i, j] = board.state
        focus_valid_moves[i, j] = one_hot(board.get_valid_moves())

print("focus states:", focus_states.shape)
print("focus_valid_moves", tuple(focus_valid_moves.shape))

# full_linear_probe = t.load(OTHELLO_MECHINT_ROOT / "main_linear_probe.pth", map_location=device)

linear_probe2 = t.load("probes/linear/resid_6_linear.pth")

rows = 8
cols = 8
options = 3
assert linear_probe2.shape == (1, cfg.d_model, rows, cols, options)

black_to_play_index = 0
white_to_play_index = 1
blank_index = 0
their_index = 1
my_index = 2

# Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")

'''LAYER = 6
game_index = 0
move = 29'''

BLANK1 = 0
BLACK = 1
WHITE = -1

# MINE = 0
# YOURS = 1
# BLANK2 = 2

EMPTY = 0
YOURS = 1
MINE = 2


# ## Code

# In[3]:


from utils import plot_game
from training_utils import get_state_stack_num_flipped
from utils import plot_probe_outputs


# ### Plots

# In[4]:


'''game_index = 0
move = 4
end_move = 16
LAYER = 5
square = "D5"
square_tuple = (3, 5)
tile_state_clean = 1'''

plot_game(focus_games_string, game_index=0, end_move = 16)
'''# plot_single_board(focus_games_string[game_index, :move+1], title="Original Game (black plays E0)")
# plot_single_board(focus_games_string[game_index, :move].tolist()+[to_string(to_int("C4"))], title="Corrupted Game (blank plays C0)")
focus_states_num_flipped = get_state_stack_num_flipped(focus_games_string)
imshow(
        focus_states_num_flipped[game_index, :end_move],
        facet_col=0,
        facet_col_wrap=8,
        facet_labels=[f"Move {i}" for i in range(0, end_move)],
        title="First 16 moves of first game",
        color_continuous_scale="Greys",
        y = [i for i in alpha],
    )
flipped_list = list(focus_states_num_flipped[game_index, :end_move, 3, 5])
first_flip = True if flipped_list[0] == 1 else False
flipped_list = [first_flip] + [flipped_list[i-1] < flipped_list[i] for i in range(1, end_move)]
print(len(flipped_list))
flipped_list = [i for i in range(0, end_move) if flipped_list[i]]
print(flipped_list)'''
# plot_single_board(int_to_label(moves_int))
# plot_probe_outputs(focus_cache, full_linear_probe, 5, game_index, 4)


# In[5]:


print(focus_games_string[0])


# ### Rest

# In[6]:


LAYER = 4

@dataclass
class Arguments:
    clean_input: Tensor = None
    corrupted_input: Tensor = None
    square: str = None
    corrupted_move: int = None
    start_move: int = None
    end_move: int = None
    move: int = None
    tile_state_clean: int = None
    tile_state_corrupt: int = None
    include_resid: bool = True
    include_heads: bool = True
    denoising: bool = True


# In[7]:


def square_tuple_from_square(square : str):
    return (alpha.index(square[0]), int(square[1]))
    # assert type(square) == int
    # square_str = to_string(square)
    # return (square_str // 8, square_str % 8)


# In[8]:


def cache_to_logit(cache: ActivationCache, args: Arguments) -> Float[Tensor, "1"]:
    square_tuple = square_tuple_from_square(args.square)
    resid = cache["resid_post", LAYER][0]
    logits= einops.einsum(resid, linear_probe2, 'pos d_model, modes d_model rows cols options -> modes pos rows cols options')[0]
    '''logit_diffs = logits.log_softmax(dim=-1)
    logit_diff = logit_diffs[move, square_tuple[0], square_tuple[1], tile_state_clean]
    return logit_diff'''
    logit_correct_clean = logits[args.move, square_tuple[0], square_tuple[1], args.tile_state_clean]
    logit_correct_corrupt = logits[args.move, square_tuple[0], square_tuple[1], args.tile_state_corrupt]
    logit_diff = logit_correct_clean - logit_correct_corrupt
    return logit_diff
    


# In[9]:


def patching_metric(patched_cache: ActivationCache, corrupted_logit_diff: float, clean_logit_diff: float, args: Arguments) -> Float[Tensor, "1"]:
    '''
    Function of patched logits, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.

    Should be linear function of the logits for the d5 token at the final move.
    '''
    # SOLUTION
    patched_logit_diff = cache_to_logit(patched_cache, args)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


# In[10]:


def patch_final_move_output(
    activation: Float[Tensor, "batch (head_ind) seq d_model"],
    hook: HookPoint,
    clean_cache: ActivationCache,
    head: Optional[int] = None
) -> Float[Tensor, "batch (head_ind) seq d_model"]:
    '''
    Hook function which patches activations at the final sequence position.

    Note, we only need to patch in the final sequence position, because the
    prior moves in the clean and corrupted input are identical (and this is
    an autoregressive model).
    '''
    # SOLUTION
    # print(activation.shape)
    if not head is None:
        activation[0, -1, head, :] = clean_cache[hook.name][0, -1, head, :]
    else:
        activation[0, -1, :] = clean_cache[hook.name][0, -1, :]
    return activation


def get_act_patch_resid_pre(
    model: HookedTransformer,
    corrupted_input: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[[ActivationCache], Float[Tensor, ""]],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
    args: Arguments
) -> Float[Tensor, "2 n_layers"]:
    '''
    Returns an array of results, corresponding to the results of patching at
    each (attn_out, mlp_out) for all layers in the model.
    '''
    # SOLUTION
    if args.include_resid:
        PATCH_LAYERS = 4
    else:
        PATCH_LAYERS = 2

    if args.include_heads:
        PATCH_HEADS = 8
    else:
        PATCH_HEADS = 0

    activations = ["attn_out", "mlp_out"]
    if args.include_resid:
        activations += ["resid_pre", "resid_post"]
    
    results = t.zeros(PATCH_LAYERS+PATCH_HEADS, model.cfg.n_layers, device=device, dtype=t.float32)

    for layer in tqdm(range(model.cfg.n_layers)):
        for i, activation in enumerate(activations):
            hook_fn = partial(patch_final_move_output, clean_cache=clean_cache)
            model.reset_hooks()
            cache = model.add_caching_hooks()
            _ = model.run_with_hooks(
                corrupted_input,
                fwd_hooks = [(utils.get_act_name(activation, layer), hook_fn)],
            )
            cache = ActivationCache(cache, model)
            results[i, layer] = patching_metric(cache, corrupted_logit_diff, clean_logit_diff, args)
    
        for head in range(PATCH_HEADS):
            hook_fn = partial(patch_final_move_output, clean_cache=clean_cache, head=head)
            model.reset_hooks()
            cache = model.add_caching_hooks()
            _ = model.run_with_hooks(
                corrupted_input,
                fwd_hooks = [(utils.get_act_name("z", layer), hook_fn)],
            )
            cache = ActivationCache(cache, model)
            results[PATCH_LAYERS+head, layer] = patching_metric(cache, corrupted_logit_diff, clean_logit_diff, args)

    return results


# In[11]:


from utils import seq_to_state_stack


# In[12]:


# Function takes in a tile e.g. "D3" and
# returns a list "blank" / "mine" / "their" with length 60
# For every move it sais whether the tile is blank, mine or their
def get_tile_state(square, input_int):
    assert len(input_int.shape) == 1
    input_int = input_int.tolist()
    input_str = [to_string(i) for i in input_int]
    tile_index = to_string(to_int(square))
    game_state = seq_to_state_stack(input_str)
    tile_state_list_w_b = game_state[:, tile_index // 8, tile_index % 8].copy() # 0 is blank, -1 is white, 1 is black
    # change dtype to int
    tile_state_list_w_b = tile_state_list_w_b.astype(int)
    assert len(tile_state_list_w_b.shape) == 1
    assert tile_state_list_w_b.shape[0] == len(input_int)
    tile_state_list_w_b[0::2] *= -1 # 0 blank, 1 mine, -1 theirs
    tile_state_list_w_b[tile_state_list_w_b == 1] = MINE # 2
    tile_state_list_w_b[tile_state_list_w_b == -1] = YOURS # 1
    assert set(tile_state_list_w_b).issubset(set([0, 1, 2]))
    return list(tile_state_list_w_b) # 0 empty, 1 yours, 2 mine


# In[13]:


def tile_state_to_str(tile_state):
    return "EMPTY" if tile_state == EMPTY else "YOURS" if tile_state == YOURS else "MINE"


# In[14]:


def activation_patching_from_inputs(args: Arguments):
    # Create tile_state_clean and tile_state_corrupted instead of tile_state_clean
    tile_state_list_clean = get_tile_state(args.square, args.clean_input)
    tiel_state_list_corrupt = get_tile_state(args.square, args.corrupted_input)
    print(tile_state_list_clean)
    print(tiel_state_list_corrupt)
    if args.start_move is None:
        args.start_move = args.corrupted_move

    for move in range(args.start_move, args.end_move):
        args.move = move
        args.tile_state_clean = tile_state_list_clean[move]
        args.tile_state_corrupt = tiel_state_list_corrupt[move]
        clean_input_short = args.clean_input[:move+1].clone()
        corrupted_input_short = args.corrupted_input[:move+1].clone()

        _, clean_cache = model.run_with_cache(clean_input_short)
        _, corrupted_cache = model.run_with_cache(corrupted_input_short)

        clean_logit_diff = cache_to_logit(clean_cache, args)
        corrupted_logit_diff = cache_to_logit(corrupted_cache, args)
        # Output Tile State Clean and Corrupt as String
        print(f"Tile State Clean: {tile_state_to_str(args.tile_state_clean)}")
        print(f"Tile State Corrupt: {tile_state_to_str(args.tile_state_corrupt)}")
        # Output the logit diff for the clean and corrupted input
        print(f"Clean logit Diff of {args.square} at move {args.move}: {clean_logit_diff.item()}")
        print(f"Corrupted logit Diff of {args.square} at move {args.move}: {corrupted_logit_diff.item()}")

        if args.denoising:
            patching_results = get_act_patch_resid_pre(model, corrupted_input_short, clean_cache, patching_metric, corrupted_logit_diff, clean_logit_diff, args)
        else:
            patching_results = get_act_patch_resid_pre(model, clean_input_short, corrupted_cache, patching_metric, corrupted_logit_diff, clean_logit_diff, args)

        line_labels = ["attn", "mlp"]
        if args.include_resid:
            line_labels += ["resid_pre", "resid_post"]
        if args.include_heads:
            line_labels += [f"head_{head}" for head in range(8)]
        assert patching_results.shape[0] == len(line_labels)

        line(patching_results, title=f"Layer Output Patching Effect on {args.square} Logit Diff", line_labels=line_labels, width=750)


# In[15]:


# clean_input_int     = t.Tensor(to_int([37, 43, 42, 29, 19, 41, 44, 21, 30, 39, 14, 38, 51, 26, 45, 10, 1, 22, 46, 12, 23, 7, 18, 15, 3, 47, 20, 31])).to(t.int64)
# corrupted_input_int = t.Tensor(to_int([37, 43, 42, 29, 19, 34, 41, 21, 30, 39, 14, 38, 51, 26, 45, 10, 1, 22, 46, 12, 23, 7, 18, 15, 3, 47, 20, 31])).to(t.int64)
# plot_game(to_string(clean_input_int.unsqueeze(0)), game_index=0, end_move = 16)
# plot_game(to_string(corrupted_input_int.unsqueeze(0)), game_index=0, end_move = 16)

# OLD ARGS
'''
clean_input = to lazy ...
# length = 28
# GAME_INDEX = 0


args.clean_input     = t.Tensor(to_int([37, 43, 42, 29, 19, 41, 44, 21, 30, 39, 14, 38, 51, 26, 45, 10, 1, 22, 46, 12, 23, 7, 18, 15, 3, 47, 20, 31])).to(t.int64)
args.corrupted_input = t.Tensor(to_int([37, 43, 42, 29, 19, 34, 41, 21, 30, 39, 14, 38, 51, 26, 45, 10, 1, 22, 46, 12, 23, 7, 18, 15, 3, 47, 20, 31])).to(t.int64)
args.corrupted_move = 6-1
# length = 28
# GAME_INDEX = 0
corrupted_square = 41
args.square = to_board_label(corrupted_square)
args.end_move = 16
args.include_resid = False
args.include_heads = True
args.denoising = False
'''


# In[16]:


from generate_patches import generate_patch
from pprint import pprint


# In[17]:


args = Arguments()
'''patch_infor = generate_patch(min_length = 40, min_flipped_clean = 1, max_flipped_corrupt = 100, min_tries = 10000, same_flips = True, max_first_flip = 30)
patch_info = generate_patch(max_flipped_corrupt = 100)
pprint(patch_info, compact=True)
args.clean_input = t.Tensor(to_int(patch_info["board_history_clean"])).to(t.int64)
args.corrupted_input = t.Tensor(to_int(patch_info["board_history_corrupt"])).to(t.int64)
args.corrupted_move = patch_info["corrupted_move"] - 1
args.square = to_board_label(patch_info["corrupted_square"])'''
# {'board_history_clean': [44, 29, 21, 43, 37, 53, 42, 30, 39, 31, 23, 13, 51, 59, 26, 19, 5, 25, 24, 38, 34, 22, 58, 20, 60, 46, 11, 10, 18, 12, 9, 6, 50, 17, 54, 32, 7, 1, 40, 52, 8, 49, 14, 47, 55, 62, 33, 0, 61, 57, 48, 41, 16, 2, 4, 3], 'board_history_corrupt': [44, 29, 21, 43, 37, 45, 53, 30, 39, 31, 23, 13, 51, 59, 26, 19, 5, 25, 24, 38, 34, 22, 58, 20, 60, 46, 11, 10, 18, 12, 9, 6, 50, 17, 54, 32, 7, 1, 40, 52, 8, 49, 14, 47, 55, 62, 33, 0, 61, 57, 48, 41, 16, 2, 4, 3], 'corrupted_square': 53, 'corrupted_move': 6, 'length': 56, 'flipped_list_clean': [41, 50, 51], 'flipped_list_corrupt': [41, 50, 51]}
# {'board_history_clean': [26, 18, 19, 34, 9, 10, 1, 11, 44, 45, 46, 54, 42, 25, 37, 53, 24, 8, 62, 63, 3, 38, 39, 20, 55, 43, 52, 4, 13, 32, 16, 22, 40, 50, 6, 2, 5, 61, 31, 47, 58, 49, 59, 51, 0, 12, 41, 23, 21, 29, 48, 14, 7, 30, 60, 57], 'board_history_corrupt': [26, 18, 19, 34, 17, 9, 1, 11, 44, 45, 46, 54, 42, 25, 37, 53, 24, 8, 62, 63, 3, 38, 39, 20, 55, 43, 52, 4, 13, 32, 16, 22, 40, 50, 6, 2, 5, 61, 31, 47, 58, 49, 59, 51, 0, 12, 41, 23, 21, 29, 48, 14, 7, 30, 60, 57], 'corrupted_square': 9, 'corrupted_move': 5, 'length': 56, 'flipped_list_clean': [6, 17, 40], 'flipped_list_corrupt': []}


# In[18]:


patch_info = {'board_history_clean': [44, 29, 21, 43, 37, 53, 42, 30, 39, 31, 23, 13, 51, 59, 26, 19, 5, 25, 24, 38, 34, 22, 58, 20, 60, 46, 11, 10, 18, 12, 9, 6, 50, 17, 54, 32, 7, 1, 40, 52, 8, 49, 14, 47, 55, 62, 33, 0, 61, 57, 48, 41, 16, 2, 4, 3], 'board_history_corrupt': [44, 29, 21, 43, 37, 45, 53, 30, 39, 31, 23, 13, 51, 59, 26, 19, 5, 25, 24, 38, 34, 22, 58, 20, 60, 46, 11, 10, 18, 12, 9, 6, 50, 17, 54, 32, 7, 1, 40, 52, 8, 49, 14, 47, 55, 62, 33, 0, 61, 57, 48, 41, 16, 2, 4, 3], 'corrupted_square': 53, 'corrupted_move': 6, 'length': 56, 'flipped_list_clean': [41, 50, 51], 'flipped_list_corrupt': [41, 50, 51]}
# patch_info = {'board_history_clean': [26, 18, 19, 34, 9, 10, 1, 11, 44, 45, 46, 54, 42, 25, 37, 53, 24, 8, 62, 63, 3, 38, 39, 20, 55, 43, 52, 4, 13, 32, 16, 22, 40, 50, 6, 2, 5, 61, 31, 47, 58, 49, 59, 51, 0, 12, 41, 23, 21, 29, 48, 14, 7, 30, 60, 57], 'board_history_corrupt': [26, 18, 19, 34, 17, 9, 1, 11, 44, 45, 46, 54, 42, 25, 37, 53, 24, 8, 62, 63, 3, 38, 39, 20, 55, 43, 52, 4, 13, 32, 16, 22, 40, 50, 6, 2, 5, 61, 31, 47, 58, 49, 59, 51, 0, 12, 41, 23, 21, 29, 48, 14, 7, 30, 60, 57], 'corrupted_square': 9, 'corrupted_move': 5, 'length': 56, 'flipped_list_clean': [6, 17, 40], 'flipped_list_corrupt': []}
pprint(patch_info, compact=True)
args.clean_input = t.Tensor(to_int(patch_info["board_history_clean"])).to(t.int64)
args.corrupted_input = t.Tensor(to_int(patch_info["board_history_corrupt"])).to(t.int64)
args.corrupted_move = patch_info["corrupted_move"] - 1
args.square = to_board_label(patch_info["corrupted_square"])


# In[19]:


args.square


# In[20]:


clean_input_str = patch_info["board_history_clean"]
corrupt_input_str = patch_info["board_history_corrupt"]
# plot_board(clean_input_str)


# In[21]:


# plot_board(corrupt_input_str)


# In[22]:


print(clean_input_str)
print(corrupt_input_str)


# In[23]:


plot_game(to_string(args.clean_input.unsqueeze(0)), game_index=0, end_move = 50)
plot_game(to_string(args.corrupted_input.unsqueeze(0)), game_index=0, end_move = 50)


# In[ ]:





# In[24]:


args.end_move = patch_info["length"] - 1
args.include_resid = False
args.include_heads = True
args.denoising = False
args.end_move = 50
args.start_move = 47

activation_patching_from_inputs(args)


# In[25]:


_, clean_cache = model.run_with_cache(args.clean_input)
_, corrupted_cache = model.run_with_cache(args.corrupted_input)


# In[26]:


# load flipped_0

flipped_probe_L0 = t.load("probes/flipped/resid_0_flipped.pth")
corrupt_resid_0 = corrupted_cache["resid_post", 0][0]
sqaure_tuple = square_tuple_from_square(args.square)
flipped_0 = einops.einsum(corrupt_resid_0, flipped_probe_L0, 'pos d_model, modes d_model rows cols options -> modes pos rows cols options')[0, 48, sqaure_tuple[0], sqaure_tuple[1]]

flipped_probe_L5 = t.load("probes/flipped/resid_5_flipped.pth")
corrupt_resid_5 = corrupted_cache["resid_post", 5][0]
sqaure_tuple = square_tuple_from_square(args.square)
flipped_5 = einops.einsum(corrupt_resid_5, flipped_probe_L5, 'pos d_model, modes d_model rows cols options -> modes pos rows cols options')[0, 48, sqaure_tuple[0], sqaure_tuple[1]]

print(flipped_0, flipped_5)
print(t.softmax(flipped_0, dim=-1)[0])
print(t.softmax(flipped_5, dim=-1)[0])
# What the hell? Why is flipped_5 less confident, then flipped_0 am I dumb


# - How does L0_Head Produce the Flip direction for G5?
#     - Head 5 seems to do something.
#     - Which Heads are involved?

# In[27]:


def heads_involved_in_flipping(square_str, clean_input, corrupt_input, layer, num_heads, position):
    clean_input = clean_input.clone()[:position + 1]
    corrupt_input = corrupt_input.clone()[:position + 1]
    square_tuple = square_tuple_from_square(square_str)
    # flipped_probe = t.load(f"probes/flipped/resid_{layer}_flipped.pth")
    _, corrupt_cache = model.run_with_cache(corrupt_input)
    # enumerate over all tuples of lenth of num_heads in [0, 1, 2, 3, 4, 5, 6, 7]
    # layer_name = utils.get_act_name("attn_out", layer)
    layer_name = utils.get_act_name("z", layer)
    for heads in [[1, 4, 5], [1], [4], [5]]:
        # for heads in itertools.combinations(range(8), num_heads):
        # for heads in itertools.combinations(range(1), num_heads):
        hooks = []
        for head in heads:
            hook_fn = partial(patch_final_move_output, clean_cache=corrupt_cache, head=head)
            # hook_fn = partial(patch_final_move_output, clean_cache=corrupt_cache)
            hooks += [(layer_name, hook_fn)]
        model.reset_hooks()
        patched_cache = model.add_caching_hooks()
        _ = model.run_with_hooks(
            clean_input,
            fwd_hooks = hooks,
        )
        patched_cache = ActivationCache(patched_cache, model)
        resid = patched_cache["resid_post", layer][0]
        flipped = einops.einsum(resid, flipped_probe_L0, 'pos d_model, modes d_model rows cols options -> modes pos rows cols options')[0, position, square_tuple[0], square_tuple[1]]
        print(flipped)
        flipped = (flipped[0] - flipped[1]).item()
        # flipped = t.softmax(flipped, dim=-1)[0].item()
        # if flipped > 0.65:
        print(f"Flipped: {flipped} with heads {heads}")

heads_involved_in_flipping(args.square, args.clean_input, args.corrupted_input, 0, 1, 48)
# 1, 5: 0.56
# 4, 5: 0.61
# 1, 4, 5: 0.81
# 0, 4, 5: 0.71
# 4, 5, 7: 0.71
# 5 > 4 > 1 > 0 > 7
# adding the single contributions of 1, 4, 5 gives 0.3


# ### Look at Attn Patterns

# In[28]:


# import circuitsvis as cv
from circuitsvis.attention import attention_patterns
from circuitsvis.tokens import colored_tokens
import webbrowser
colored_tokens(["hey"], [1, 3])


# In[29]:


# webbrowser.Opera
attention_pattern = corrupted_cache["pattern", 0][0]
print(attention_pattern.shape)
str_tokens = to_int(corrupt_input_str)
print(str_tokens)
str_tokens = [str(i) for i in str_tokens]
print(str_tokens)

print("Layer 0 Head Attention Patterns:")
display(attention_patterns(
    tokens=str_tokens,
    attention=attention_pattern,
    attention_head_names=[f"L0H{i}" for i in range(8)],
))

'''html = cv.attention.attention_patterns(
    tokens=str_tokens,
    attention=attention_pattern,
    attention_head_names=[f"L0H{i}" for i in range(8)],
)
code = html.show_code()
f = open("attention_pattern.html", "w")
f.write(code)
f.close()
filename = 'file://'+os.path.realpath('attention_pattern.html')
print(filename)
webbrowser.open(filename) '''


# In[30]:


from IPython.display import HTML
s = """

Header 1
Header 2


row 1, cell 1
row 1, cell 2


row 2, cell 1
row 2, cell 2

"""
h = HTML(s); h


# ### Firgure out model, resid_pre, resid_post

# In[31]:


model.cfg.positional_embedding_type


# In[32]:


first_move = focus_games_int[0, 0]
print(first_move)
W_E = model.W_E
W_pos = model.W_pos
# print(W_E[first_move] + W_pos[first_move])
# resid_pre_maybe = W_E[first_move] + W_pos[first_move]
resid_pre_maybe = W_E[first_move] + W_pos[0]
model.reset_hooks()
logits, cache = model.run_with_cache(focus_games_int[0, :-1])
# print(cache, cache.keys())
resid_pre = cache["resid_pre", 0][0, 0]
print(resid_pre.shape)
print(resid_pre[:10])
print(resid_pre_maybe[:10])
t.testing.assert_close(resid_pre, resid_pre_maybe)


# ### Old

# In[33]:


'''model.reset_hooks()
hook_fn = partial(patch_final_move_output, clean_cache=clean_cache)
cache = model.add_caching_hooks()
_ = model.run_with_hooks(
    corrupted_input,
    fwd_hooks = [
        (utils.get_act_name("attn_out", 0), hook_fn),
        (utils.get_act_name("attn_out", 4), hook_fn)]
)
cache = ActivationCache(cache, model)
print(cache.keys())
metric = patching_metric(cache)
print(metric)'''
# print(cache)'''


# In[ ]:


def patch_final_move_output(
    activation: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    clean_cache: ActivationCache,
) -> Float[Tensor, "batch seq d_model"]:
    '''
    Hook function which patches activations at the final sequence position.

    Note, we only need to patch in the final sequence position, because the
    prior moves in the clean and corrupted input are identical (and this is
    an autoregressive model).
    '''
    # SOLUTION
    activation[0, -1, :] = clean_cache[hook.name][0, -1, :]
    return activation


def get_act_patch_resid_pre(
    model: HookedTransformer,
    corrupted_input: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[[ActivationCache], Float[Tensor, ""]],
    corrupted_logit_diff: float,
    clean_logit_diff: float
) -> Float[Tensor, "2 n_layers"]:
    '''
    Returns an array of results, corresponding to the results of patching at
    each (attn_out, mlp_out) for all layers in the model.
    '''
    # SOLUTION
    model.reset_hooks()
    cache = model.add_caching_hooks()
    results = t.zeros(2, model.cfg.n_layers, device=device, dtype=t.float32)
    hook_fn = partial(patch_final_move_output, clean_cache=clean_cache)

    for i, activation in enumerate(["attn_out", "mlp_out"]):
        for layer in tqdm(range(model.cfg.n_layers)):
            _ = model.run_with_hooks(
                corrupted_input,
                fwd_hooks = [(utils.get_act_name(activation, layer), hook_fn)],
            )
            cache = ActivationCache(cache, model)
            results[i, layer] = patching_metric(cache, corrupted_logit_diff, clean_logit_diff)

    return results


# In[ ]:




