# %%
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
# from generate_patches import generate_patch
from pprint import pprint
from utils import plot_game
from training_utils import get_state_stack_num_flipped
from utils import plot_probe_outputs
from utils import seq_to_state_stack
from utils import VisualzeBoardArguments
from utils import visualize_game

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import plot_boards_general
import numpy as np
import pickle

# import part6_othellogpt.tests as tests

t.manual_seed(42)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

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

start = 0
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

rows = 8
cols = 8
options = 3

black_to_play_index = 0
white_to_play_index = 1
blank_index = 0
their_index = 1
my_index = 2

# Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")

from utils import *

# %%
def get_estimated_attention_pattern(num_games):
    estimated_attention_pattern : Float[Tensor, "layer head pos_from pos_to"] = t.zeros((8, 8, 59, 59)).to(device)
    estimated_attention_pattern_variance = t.zeros((8, 8)).to(device)
    for layer in range(8):
        _, cache = model.run_with_cache(
            board_seqs_int[:num_games, :-1].to(device),
            return_type=None,
            names_filter = lambda name : name in [utils.get_act_name("pattern", layer)]
        )
        attention_pattern = cache["pattern", layer]
        estimated_attention_pattern[layer] = attention_pattern.mean(dim=0)
        estimated_attention_pattern_variance[layer] = attention_pattern.var(dim=0).mean()
    estimated_attention_pattern_variance = estimated_attention_pattern_variance.mean()
    print(estimated_attention_pattern_variance)
    return estimated_attention_pattern

def get_avg_resid(layer, num_games):
    _, cache = model.run_with_cache(
        board_seqs_int[:num_games, :-1].to(device),
        return_type=None,
        names_filter = lambda name : name in [f"blocks.{layer}.ln1.hook_normalized"]
    )
    return cache[f"blocks.{layer}.ln1.hook_normalized"].mean(dim=0)

# %%
'''# Wenn ich so an allen stellen etwas ändere muss ich wirklich schon beim coden alles testen, so viel es geht!!!
layer = 1
tile_tuple = (4, 4)
probe_flipped_b = get_probe(layer-1, "flipped", "post")[0, :, *tile_tuple, FLIPPED].detach()
yours_probe_b = get_probe(layer, "linear", "mid")[0, :, *tile_tuple, YOURS].detach()
mine_probe_b = get_probe(layer, "linear", "mid")[0, :, *tile_tuple, MINE].detach()
probe_flipped_normalized_b = probe_flipped_b / probe_flipped_b.norm()
flipped_after_OV_b = einops.einsum(probe_flipped_normalized_b, OV.AB[layer, :], "d_model_in, head_idx d_model_in d_model_out -> head_idx d_model_out")
conversion_factors_yours_b = einops.einsum(flipped_after_OV_b, yours_probe_b, "head_idx d_model, d_model -> head_idx")
conversion_factors_mine_b = einops.einsum(flipped_after_OV_b, mine_probe_b, "head_idx d_model, d_model -> head_idx")

avg_resids_b = focus_cache[f"blocks.{layer}.ln1.hook_normalized"].mean(dim=0)
projection_b = einops.repeat(avg_resids_b @ probe_flipped_b, "pos -> pos d_model", d_model=512) / (probe_flipped_b @ probe_flipped_b) * probe_flipped_b
avg_resids_without_flipped_b = avg_resids_b - projection_b
avg_resids_after_OV_b = einops.einsum(avg_resids_without_flipped_b, OV.AB[layer, :], "pos d_model_in, head_idx d_model_in d_model_out -> head_idx pos d_model_out")
avg_resids_yours_bias_b = einops.einsum(avg_resids_after_OV_b, yours_probe_b, "head_idx pos d_model, d_model -> head_idx pos")
avg_resids_mine_bias_b = einops.einsum(avg_resids_after_OV_b, mine_probe_b, "head_idx pos d_model, d_model -> head_idx pos")'''

# All the Things needed later for all layers head row col
# Bei den layout von dimensionen sollte man immer darauf achten, je wahrscheinlicher es ist, dass eine dimension für mul oder add benutz wird, desto weiter nach rechts sollte sie sein
# TODO: Does it make more sense if this is normalized? NO Because we want the Logit and thats not normalized! (DONE)
num_games = 200

OV = model.OV
flipped_probe : Float[Tensor, "d_model layer row col"] = t.Tensor(size=(512, 8, 8, 8)).to(device)
flipped_probe_normalized : Float[Tensor, "d_model layer row col"] = t.Tensor(size=(512, 8, 8, 8)).to(device)
yours_probe : Float[Tensor, "d_model layer row col"] = t.Tensor(size=(512, 8, 8, 8)).to(device)
mine_probe : Float[Tensor, "d_model layer row col"] = t.Tensor(size=(512, 8, 8, 8)).to(device)
empty_probe : Float[Tensor, "d_model layer row col"] = t.Tensor(size=(512, 8, 8, 8)).to(device)
conversion_factors_mine : Float[Tensor, "layer head_idx row col"] = t.Tensor(size=(8, 8, 8, 8)).to(device)
conversion_factors_yours : Float[Tensor, "layer head_idx row col"] = t.Tensor(size=(8, 8, 8, 8)).to(device) 
avg_resids_mine_bias : Float[Tensor, "layer head_idx pos_to row col"] = t.Tensor(size=(8, 8, 59, 8, 8)).to(device)
avg_resids_yours_bias : Float[Tensor, "layer head_idx pos_to row col"] = t.Tensor(size=(8, 8, 59, 8, 8)).to(device)
avg_resids_without_flipped : Float[Tensor, "layer d_model pos_to row col"] = t.Tensor(size=(8, 512, 59, 8, 8)).to(device)

for layer in range(1, 8):
    flipped_probe_s = get_probe(layer-1, "flipped", "post")[0, :, :, :, FLIPPED].detach()
    flipped_probe[:, layer, :, :] = flipped_probe_s
    flipped_probe_normalized = flipped_probe / flipped_probe.norm(dim=0)
    yours_probe_s = get_probe(layer, "linear", "mid")[0, :, :, :, YOURS].detach()
    yours_probe[:, layer, :, :] = yours_probe_s
    mine_probe_s = get_probe(layer, "linear", "mid")[0, :, :, :, MINE].detach()
    mine_probe[:, layer, :, :] = mine_probe_s
    empty_probe_s = get_probe(layer, "linear", "mid")[0, :, :, :, EMPTY].detach()
    empty_probe[:, layer, :, :] = empty_probe_s

    # avg_resids = focus_cache[f"blocks.{layer}.ln1.hook_normalized"].mean(dim=0)
    avg_resids = get_avg_resid(layer, 200)
    projection = einops.einsum(avg_resids, flipped_probe_s, "p d, d r c -> p r c") / einops.einsum(flipped_probe_s, flipped_probe_s, "d r c, d r c -> r c") * einops.repeat(flipped_probe_s, "d r c -> d p r c", p=59)
    avg_resids_without_flipped_s = einops.repeat(avg_resids, "pos_to d_model -> d_model pos_to row col", row = 8, col = 8) - projection
    avg_resids_without_flipped[layer] = avg_resids_without_flipped_s
    avg_resids_after_OV = einops.einsum(avg_resids_without_flipped_s, OV.AB[layer, :], "d_model_in pos_to row col, head_idx d_model_in d_model_out -> d_model_out head_idx pos_to row col")
    avg_resids_yours_bias[layer] = einops.einsum(avg_resids_after_OV, yours_probe_s, "d_model head_idx pos_to row col, d_model row col -> head_idx pos_to row col")
    avg_resids_mine_bias[layer] = einops.einsum(avg_resids_after_OV, mine_probe_s, "d_model head_idx pos_to row col, d_model row col -> head_idx pos_to row col")

flipped_after_OV = einops.einsum(flipped_probe_normalized, OV.AB, "d_model_in layer row col, layer head_idx d_model_in d_model_out -> d_model_out layer head_idx row col")
conversion_factors_yours = einops.einsum(flipped_after_OV, yours_probe, "d_model layer head_idx row col, d_model layer row col -> layer head_idx row col")
conversion_factors_mine = einops.einsum(flipped_after_OV, mine_probe, "d_model layer head_idx row col, d_model layer row col -> layer head_idx row col")

# TODO: Make this work (DONE)
def get_probe_dir2(resid : Float[Tensor, "batch pos d_model"], layer : int, row, col):
    flipped_probe_normalized_small = flipped_probe_normalized[:, layer, row, col]
    avg_resids_without_flipped_small = avg_resids_without_flipped[layer, :, :, row, col]
    flipped_in_resid : Float[Tensor, "batch pos"] = resid @ flipped_probe_normalized_small
    dir = einops.repeat(avg_resids_without_flipped_small, "d_model pos -> batch pos d_model", batch=200) + einops.repeat(flipped_in_resid, "batch pos -> batch pos d_model", d_model=512) * flipped_probe_normalized_small
    return dir

estimated_attention_pattern = get_estimated_attention_pattern(200)

# %%
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from typing import List, Union, Optional, Tuple, Callable, Dict


# %%
DEBUG = False

def add_bias(yours_logits_pred, mine_logits_pred, layer, tile_tuple):
    yours_probe_s = yours_probe[:, layer, *tile_tuple]
    mine_probe_s = mine_probe[:, layer, *tile_tuple]
    bias = model.b_O[layer]
    yours_logit_bias = einops.einsum(bias, yours_probe_s, "d_model, d_model -> ")
    yours_logits_pred += yours_logit_bias
    mine_logit_bias = einops.einsum(bias, mine_probe_s, "d_model, d_model -> ")
    mine_logits_pred += mine_logit_bias
    return yours_logits_pred, mine_logits_pred

def get_attn_pattern(layer, use_attn_pattern_approx, cache = None) -> Float[Tensor, "head pos_from pos_to"]:
    assert cache is not None or use_attn_pattern_approx
    if use_attn_pattern_approx:
        attention_pattern = einops.repeat(estimated_attention_pattern[layer, :, :], "head pos_from pos_to -> batch head pos_from pos_to", batch=200)
    else:
        attention_pattern : Float[Tensor, "head_idx pos_from pos_to"] = cache["pattern", layer][:, :, :]
    return attention_pattern

def get_yours_and_mine_pred_old(
        resid_real : Float[Tensor, "batch pos d_model"],
        layer : int, 
        tile_tuple : Tuple[int, int],
        cache,
        use_attn_pattern_approx : bool = True
    ):
    yours_logits_pred = t.zeros((200, 59)).to(device)
    mine_logits_pred = t.zeros((200, 59)).to(device)
    for head in range(8):
        yours_probe_s = yours_probe[:, layer, *tile_tuple]
        mine_probe_s = mine_probe[:, layer, *tile_tuple]
        resid = get_probe_dir2(resid_real, layer, *tile_tuple) # TODO: this could be wrong i guess
        # resid = resid_real
        # print(f"resid shape: {resid.shape}")
        # test_probe_dir(resid_real, resid)
        head_V = model.W_V[layer, head]
        head_v = einops.einsum(resid, head_V, "batch pos d_model, d_model d_head -> batch pos d_head")
        attention_pattern = get_attn_pattern(layer, use_attn_pattern_approx, cache)[:, head]
        # attention_pattern : Float[Tensor, "pos"] = focus_cache["pattern", layer][:, head, pos_from]
        z = einops.repeat(head_v, "batch pos_to d_head -> batch pos_from pos_to d_head", pos_from = 59) * einops.repeat(attention_pattern, "batch pos_from pos_to -> batch pos_from pos_to d_head", d_head=64)
        z = einops.reduce(z, "batch pos_from pod_to d_head -> batch pos_from d_head", "sum")
        # z = focus_cache["z", layer][:, pos_from, head] # TODO: Remove
        result = einops.einsum(z, model.W_O[layer, head], "batch pos_from d_head, d_head d_model -> batch pos_from d_model")
        yours_logit_head = einops.einsum(result, yours_probe_s, "batch pos_from d_model, d_model -> batch pos_from")
        yours_logits_pred += yours_logit_head
        mine_logit_head = einops.einsum(result, mine_probe_s, "batch pos_from d_model, d_model -> batch pos_from")
        mine_logits_pred += mine_logit_head
        # attn_out_fake += result
    yours_logits_pred, mine_logits_pred = add_bias(yours_logits_pred, mine_logits_pred, layer, tile_tuple)
    return yours_logits_pred, mine_logits_pred
        

def get_yours_and_mine_pred_math2(
        resid_real : Float[Tensor, "batch pos d_model"],
        layer : int,
        tile_tuple : tuple[int, int],
        cache,
        use_attn_pattern_approx=True,
    ):
    # flipped_probe_normalized[layer] acutally means the flipped probe of the previous layer
    flipped_probe_normalized_s : Float[Tensor, "d_model"] = flipped_probe_normalized[:, layer, *tile_tuple]
    conversion_factors_mine_s : Float[Tensor, "head_idx"] = conversion_factors_mine[layer, :, *tile_tuple]
    conversion_factors_yours_s : Float[Tensor, "head_idx"] = conversion_factors_yours[layer, :, *tile_tuple]
    avg_resids_mine_bias_s : Float[Tensor, "head_idx pos_to"] = avg_resids_mine_bias[layer, :, :, *tile_tuple]
    avg_resids_yours_bias_s : Float[Tensor, "head_idx pos_to"] = avg_resids_yours_bias[layer, :, :, *tile_tuple]
    # "layer head_idx pos_to row col"
    # TODO: Use Avg Resid, Dont Remove negative Flipped Logits
    attention_pattern : Float[Tensor, "head pos_from pos_to"] = get_attn_pattern(layer, use_attn_pattern_approx, cache)
    flipped_logit = einops.einsum(resid_real, flipped_probe_normalized_s, "batch pos_to d_model, d_model -> batch pos_to")
    # not_flipped_logit = einops.einsum(resid_real, probe_not_flipped_normalized, "batch pos d_model, d_model -> batch pos")
    # Negative Logits are doing a lot work ..
    # flipped_logit = flipped_logit * ((flipped_logit > not_flipped_logit) & (flipped_logit > 0)).to(device)
    # flipped_logit = t.max(flipped_logit, t.zeros_like(flipped_logit).to(device))
    flipped_logit = einops.repeat(flipped_logit, "batch pos_to -> batch head_idx pos_to", head_idx=8)
    yours_logits_pred = einops.repeat(flipped_logit * einops.repeat(conversion_factors_yours_s, "head_idx -> head_idx pos_to", pos_to=59) + avg_resids_yours_bias_s, "batch head_idx pos_to -> batch head_idx pos_from pos_to", pos_from=59) * attention_pattern
    yours_logits_pred = einops.reduce(yours_logits_pred, "batch head_idx pos_from pos_to -> batch pos_from", "sum")
    mine_logits_pred = einops.repeat(flipped_logit * einops.repeat(conversion_factors_mine_s, "head_idx -> head_idx pos_to", pos_to=59) + avg_resids_mine_bias_s, "batch head_idx pos_to -> batch head_idx pos_from pos_to", pos_from=59) * attention_pattern
    mine_logits_pred = einops.reduce(mine_logits_pred, "batch head_idx pos_from pos_to -> batch pos_from", "sum")
    yours_logits_pred, mine_logits_pred = add_bias(yours_logits_pred, mine_logits_pred, layer, tile_tuple)
    return yours_logits_pred, mine_logits_pred

def get_logits_real(cache, layer, tile_tuple):
    attn_out = cache["attn_out", layer]
    yours_probe_s = yours_probe[:, layer, *tile_tuple]
    mine_probe_s = mine_probe[:, layer, *tile_tuple]
    empty_probe_s = empty_probe[:, layer, *tile_tuple]
    yours_logits = einops.einsum(attn_out, yours_probe_s, "batch pos_from d_model, d_model -> batch pos_from")
    mine_logits = einops.einsum(attn_out, mine_probe_s, "batch pos_from d_model, d_model -> batch pos_from")
    empty_logits = einops.einsum(attn_out, empty_probe_s, "batch pos_from d_model, d_model -> batch pos_from")
    return yours_logits, mine_logits, empty_logits

def get_mind_change_mask(cache, layer, tile_tuple):
    resid_real = cache[f"blocks.{layer}.ln1.hook_normalized"]
    resid_mid = cache[f"blocks.{layer}.hook_resid_mid"]
    yours_probe_mid_layer = yours_probe[:, layer, *tile_tuple]
    mine_probe_mid_layer = mine_probe[:, layer, *tile_tuple]
    yours_probe_prev_layer = get_probe(layer-1, "linear", "post")[0, :, *tile_tuple, YOURS].detach()
    mine_probe_prev_layer = get_probe(layer-1, "linear", "post")[0, :, *tile_tuple, MINE].detach()
    yours_logits_prev_layer = einops.einsum(resid_real, yours_probe_prev_layer, "batch pos d_model, d_model -> batch pos")
    mine_logits_prev_layer = einops.einsum(resid_real, mine_probe_prev_layer, "batch pos d_model, d_model -> batch pos")
    yours_logits_mid_layer = einops.einsum(resid_mid, yours_probe_mid_layer, "batch pos d_model, d_model -> batch pos")
    mine_logits_mid_layer = einops.einsum(resid_mid, mine_probe_mid_layer, "batch pos d_model, d_model -> batch pos")
    mask = t.ones(size=(200, 59)).to(device)
    mask[(yours_logits_mid_layer < mine_logits_mid_layer) & (yours_logits_prev_layer < mine_logits_prev_layer)] = 0
    mask[(mine_logits_mid_layer < yours_logits_mid_layer) & (mine_logits_prev_layer < yours_logits_prev_layer)] = 0
    return mask.to(dtype=t.int)

# Input: resid_real, Outpu: logits pred and real
def get_yours_and_mine_pred(cache, layer, tile_label, use_attn_pattern_approx, func_to_evaluate, only_mind_changes=False):
    # TODO: Output Cool Ass Dataframe
    # resid_real is correct. I thought it should be layer -1 but NO!
    resid_real = cache[f"blocks.{layer}.ln1.hook_normalized"]
    tile_tuple = label_to_tuple(tile_label)
    yours_logits_pred, mine_logits_pred = func_to_evaluate(resid_real, layer, tile_tuple, cache, use_attn_pattern_approx)
    yours_logits, mine_logits, empty_logits = get_logits_real(cache, layer, tile_tuple)
    all_logits = t.stack([empty_logits, mine_logits, yours_logits], dim=-1)
    mask = all_logits.argmax(dim=-1) != 0
    logits_diff = yours_logits - mine_logits
    logits_pred_diff = yours_logits_pred - mine_logits_pred
    # TOOD: Evaluate only on not empty tiles
    # correct = (logits_pred_diff > 0) == (logits_diff > 0)
    if only_mind_changes:
        mind_change_mask = get_mind_change_mask(cache, layer, tile_tuple)
        mask = mask * mind_change_mask
    # correct = correct * mask
    # return correct.float().sum(dim=0) / mask.sum(dim=0)

    # remove everything where mask is 0
    # logits_diff = logits_diff[mask]
    # logits_pred_diff = logits_pred_diff[mask]
    return logits_diff, logits_pred_diff, mask

def get_scores(logit_diff, logit_diff_preds, mask):
    # calculate tp, tn, fp, fn
    if DEBUG:
        mask = t.ones_like(mask).to(device)
    tp = einops.reduce((logit_diff > 0) & (logit_diff_preds > 0) & mask, "batch pos -> pos", "sum")
    tn = einops.reduce((logit_diff < 0) & (logit_diff_preds < 0) & mask, "batch pos -> pos", "sum")
    fp = einops.reduce((logit_diff < 0) & (logit_diff_preds > 0) & mask, "batch pos -> pos", "sum")
    fn = einops.reduce((logit_diff > 0) & (logit_diff_preds < 0) & mask, "batch pos -> pos", "sum")
    return tp, tn, fp, fn

def get_yours_and_mine_pred_results(num_batches, batch_size, use_attn_pattern_approx, func_to_evaluate, only_mind_changes=False, start=200):
    # TODO: Seperate the batches used for attention approximation and the rest
    results = {
        "TP" : t.zeros((8, 59, 8, 8)).to(device),
        "TN" : t.zeros((8, 59, 8, 8)).to(device),
        "FP" : t.zeros((8, 59, 8, 8)).to(device),
        "FN" : t.zeros((8, 59, 8, 8)).to(device),
    }

    for layer in range(1, 8):
        for batch in range(num_batches):
            indeces = t.arange(start + batch * batch_size, start + (batch + 1) * batch_size).to(dtype=t.int)
            _, cache = model.run_with_cache(
                board_seqs_int[indeces, :-1].to(device),
                return_type=None,
                names_filter=lambda name: name in [f"blocks.{layer}.ln1.hook_normalized", f"blocks.{layer}.hook_resid_mid", utils.get_act_name("attn_out", layer), f"blocks.{layer}.attn.hook_pattern"]
            )
            for row in range(8):
                for col in range(8):
                    if DEBUG:
                        row, col = 3, 3
                    tile_label = tuple_to_label((row, col))
                    # get logits pred, real
                    logits_diff, logits_diff_pred, mask = get_yours_and_mine_pred(cache, layer, tile_label, use_attn_pattern_approx, func_to_evaluate, only_mind_changes)
                    tp, tn, fp, fn = get_scores(logits_diff, logits_diff_pred, mask)
                    if DEBUG:
                        print(tp[10], tn[10], fp[10], fn[10])
                    results["TP"][layer, :, row, col] += tp
                    results["TN"][layer, :, row, col] += tn
                    results["FP"][layer, :, row, col] += fp
                    results["FN"][layer, :, row, col] += fn
                    if DEBUG:
                        break
                if DEBUG:
                    break
        if DEBUG:
            break
    return results

def get_score_from_results(results : dict[str, Tensor], dimensions : list[str]):
    # TODO: Also do Weighted F1 (mhhh idk. I have to think about useful metrics here ...)
    assert all([dimension in ["layer", "pos", "row", "col"] for dimension in dimensions])
    scores = {}
    # compress results to the specified dimensions
    results_compressed = {}
    for key in results.keys():
        results_compressed[key] = einops.reduce(results[key], f"layer pos row col -> {' '.join(dimensions)}", "sum")
    # calculate scores
    tp = results_compressed["TP"]
    tn = results_compressed["TN"]
    fp = results_compressed["FP"]
    fn = results_compressed["FN"]
    scores["Accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    scores["Precision"] = tp / (tp + fp)
    scores["Recall"] = tp / (tp + fn)
    scores["F1"] = 2 * (scores["Precision"] * scores["Recall"]) / (scores["Precision"] + scores["Recall"])
    return scores

def evaluate_yours_and_mine_pred(num_batches, batch_size, use_attn_pattern_approx, func_to_evaluate, only_mind_changes=False):
    results = get_yours_and_mine_pred_results(num_batches, batch_size, use_attn_pattern_approx, func_to_evaluate, only_mind_changes)
    scores = get_score_from_results(results, ["layer", "pos", "row", "col"])
    return scores

DEBUG = False
batches = 1000
batch_size = 200
only_mind_changes = False
for use_attn_pattern_approx in [True, False]:
    if use_attn_pattern_approx:
        print("Using Attention Pattern Approximation")
        approx_str = "approx"
    else:
        print("Using Attention Pattern")
        approx_str = "real"
    print("Math")
    results_math = get_yours_and_mine_pred_results(batches, batch_size, use_attn_pattern_approx, get_yours_and_mine_pred_math2, only_mind_changes)
    scores_math = get_score_from_results(results_math, ["layer", "pos", "row", "col"])
    print(f"Math Acc: {scores_math['Accuracy'][1, 10, *label_to_tuple('D3')].item():.4f}")
    with open(f"results_math_{approx_str}.pkl", "wb") as file:
        pickle.dump(results_math, file)
    print("Test")
    results_test = get_yours_and_mine_pred_results(batches, batch_size, use_attn_pattern_approx, get_yours_and_mine_pred_old, only_mind_changes)
    scores_test = get_score_from_results(results_test, ["layer", "pos", "row", "col"])
    print(f"Real Acc: {scores_test['Accuracy'][1, 10, *label_to_tuple('D3')].item():.4f}")
    with open(f"results_test_{approx_str}.pkl", "wb") as file:
        pickle.dump(results_test, file)

# %%
# TODO: First get results for math version and test version, then make score with interesting two dimensions, then create heatmap ...
# TODO: Übelegen was dann kommt
# TODO: Nur Falls es langsam ist ... Alle Tiles gleichzeitig (Könnte maybe keinen Sinn machen, vielleicht will ich ja andere Tiles beim input mit rein nehmen so)
# save results_math and results_test using pickle
