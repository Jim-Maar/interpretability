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
    OthelloBoardState
)

# Load board data as ints (i.e. 0 to 60)
board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)

assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
assert board_seqs_int.max() == 60

num_games, length_of_game = board_seqs_int.shape

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

module = "post" # Eigentlich "mid" ...

for layer in range(1, 8):
    flipped_probe_s = get_probe(layer-1, "flipped", "post")[0, :, :, :, FLIPPED].detach()
    flipped_probe[:, layer, :, :] = flipped_probe_s
    flipped_probe_normalized = flipped_probe / flipped_probe.norm(dim=0)
    yours_probe_s = get_probe(layer, "linear", module)[0, :, :, :, YOURS].detach()
    yours_probe[:, layer, :, :] = yours_probe_s
    mine_probe_s = get_probe(layer, "linear", module)[0, :, :, :, MINE].detach()
    mine_probe[:, layer, :, :] = mine_probe_s
    empty_probe_s = get_probe(layer, "linear", module)[0, :, :, :, EMPTY].detach()
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

def get_attn_pattern(layer, use_attn_pattern_approx, batch_size, cache = None) -> Float[Tensor, "head pos_from pos_to"]:
    assert cache is not None or use_attn_pattern_approx
    if use_attn_pattern_approx:
        attention_pattern = einops.repeat(estimated_attention_pattern[layer, :, :], "head pos_from pos_to -> batch head pos_from pos_to", batch=batch_size)
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

def get_logits_real(layer, tile_tuple, cache=None, attn_out=None):
    if attn_out is None:
        attn_out = cache["attn_out", layer]
    # attn_out = cache["attn_out", layer]
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

EPSILON = 1e-6
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
    scores["Accuracy"] = (tp + tn) / (tp + tn + fp + fn + EPSILON)
    scores["Precision"] = tp / (tp + fp + EPSILON)
    scores["Recall"] = tp / (tp + fn + EPSILON)
    scores["F1"] = 2 * (scores["Precision"] * scores["Recall"]) / (scores["Precision"] + scores["Recall"])
    return scores

def evaluate_yours_and_mine_pred(num_batches, batch_size, use_attn_pattern_approx, func_to_evaluate, only_mind_changes=False):
    results = get_yours_and_mine_pred_results(num_batches, batch_size, use_attn_pattern_approx, func_to_evaluate, only_mind_changes)
    scores = get_score_from_results(results, ["layer", "pos", "row", "col"])
    return scores

def orthogonalize_vectors(a, B, normalize=True):
    """Orthogonalizes vector a against a list of vectors B without in-place modification using PyTorch"""
    orthogonal_a = a.clone()  # Create a copy of a to avoid in-place modification
    B_prev = []
    for b in B:
        if not all([b @ b_prev < 1e-6 for b_prev in B_prev]):
            b = orthogonalize_vectors(b, B_prev)
        # Project orthogonal_a onto b
        projection = einops.repeat(einops.einsum(a, b, "... d_model, d_model -> ...") / t.dot(b, b), "... -> ... d_model", d_model = b.shape[0]) * b
        # Update orthogonal_a by subtracting the projection
        orthogonal_a = orthogonal_a - projection
        B_prev += [b]
    
    # Normalize the resulting vector orthogonal_a
    if normalize:
        orthogonal_a = orthogonal_a / t.norm(orthogonal_a)
    
    return orthogonal_a

def get_activation(act_names, num_games, start=0):
    # TODO: If this takes to long or something, Make a filter step!
    act_name_results = {act_name : [] for act_name in act_names}
    inference_size = 1000
    for batch in range(start, start+num_games, inference_size):
        with t.inference_mode():
            _, cache = model.run_with_cache(
                board_seqs_int[batch:batch+inference_size, :-1].to(device),
                return_type=None,
                names_filter=lambda name: name in act_names
                # names_filter=lambda name: name == f"blocks.{layer}.hook_resid_mid" or name == f"blocks.{layer}.mlp.hook_post"
                # names_filter=lambda name: name == f"blocks.{layer}.hook_resid_pre" or name == f"blocks.{layer}.mlp.hook_post"
            )
        for act_name in act_names:
            act_name_results[act_name] += [cache[act_name]]
    for act_name in act_names:
        act_name_results[act_name] = t.cat(act_name_results[act_name], dim=0)
        act_name_results[act_name] = act_name_results[act_name].detach()[:num_games]
    return act_name_results

# This function is weird but okay
def get_probe2(direction_str : str, layer, tile : tuple):
    if direction_str == "mine":
        return mine_probe[:, layer, *tile]
    elif direction_str == "yours":
        return yours_probe[:, layer, *tile]
    elif direction_str == "flipped":
        return flipped_probe_normalized[:, layer+1, *tile]

def get_neuron_out_direction_scaled(pos_to, neuron, layer, game=None, no_mean=True):
    if game is not None:
        neruon_activation = focus_cache["mlp_post", layer-1][game, pos_to, neuron]
    else:
        if pos_to is not None:
            neruon_activations = focus_cache["mlp_post", layer-1][:, pos_to, neuron]
        else:
            neruon_activations = focus_cache["mlp_post", layer-1][:, :, neuron]
        neruon_activations_positive = neruon_activations[neruon_activations > 0]
        neruon_activation = neruon_activations_positive.mean()
    if no_mean:
        neruon_activation = t.Tensor([1]).to(device)
    direction = model.W_out[layer-1, neuron].detach()
    direction_scaled = neruon_activation * direction
    return direction_scaled, neruon_activation.item()

def get_logit_after_ov(layer, head, direction_scaled, direction_str, tile):
    flipped_dir = get_probe2("flipped", layer-1, tile)
    mine_dir = get_probe2("mine", layer-1, tile)
    yours_dir = get_probe2("yours", layer-1, tile)
    dir_next= get_probe2(direction_str, layer, tile)
    direction_scaled = orthogonalize_vectors(direction_scaled, [flipped_dir, yours_dir, mine_dir], normalize=False)
    direction_after_OV = einops.einsum(direction_scaled, OV.AB[layer, head], "d_model_in, d_model_in d_model_out -> d_model_out")
    logit = einops.einsum(direction_after_OV, dir_next, "d_model, d_model -> ")
    return logit.item()

def get_logit_before_ov(layer, head, direction_scaled, direction_str, tile):
    # flipped_dir = flipped_probe_normalized[:, layer, *tile]
    # mine_dir = mine_probe[:, layer-1, *tile]#
    dir = get_probe2(direction_str, layer-1, tile)
    # yours_dir = yours_probe[:, layer-1, *tile]
    logit = einops.einsum(direction_scaled, dir, "d_model, d_model -> ")
    return logit.item()

def get_activation_for_all_neurons(pos_to, layer, head, tile, game=None, get_logit_function=get_logit_after_ov, direction_str="yours"):
    activations_list = []
    logits_list = []
    for neuron in range(2048):
        direction_scaled, neuron_activation = get_neuron_out_direction_scaled(pos_to, neuron, layer, game)
        logit = get_logit_function(layer, head, direction_scaled, direction_str, tile)
        activations_list.append(neuron_activation)
        logits_list.append(logit)
    return activations_list, logits_list

def get_neuron_weights_as_probes(layer, tile_tuple, how_many=20):
    final_neurons = []
    direction_strs = ["yours", "mine", "flipped"]
    # direction_strs = ["yours", "mine"]
    for direction_str in direction_strs:
        logits_list_result = [0] * 2048
        for head in range(8):
            activations_list, logits_list = get_activation_for_all_neurons(None, layer, head, tile_tuple, game=None, get_logit_function=get_logit_before_ov, direction_str=direction_str)
            logits_list_result = [max(logits_list_result[i], logits_list[i]) for i in range(2048)]
        # get the top 40 neuron, ideces
        neuron_with_logits_list = zip(logits_list_result, range(2048))
        neuron_with_logits_list = sorted(neuron_with_logits_list, key=lambda x: x[0], reverse=True)
        top_neurons = [neuron for _, neuron in neuron_with_logits_list[:how_many]]
        final_neurons += top_neurons
    final_neurons = list(set(final_neurons))
    final_directions = [model.W_out[layer, neuron].detach() for neuron in final_neurons]
    return final_directions

from tqdm import tqdm

def get_probes_curr():
    yours_probe_s_curr = get_probe(layer, "linear", "post")[0, :, *tile_tuple, YOURS].detach()
    mine_probe_s_curr = get_probe(layer, "linear", "post")[0, :, *tile_tuple, MINE].detach()
    return yours_probe_s_curr, mine_probe_s_curr

def get_logits_pred_diff(probes, layer, num_games, resid_real, attention_pattern):
    yours_probe_s_curr, mine_probe_s_curr = get_probes_curr()
    avg_resids = get_avg_resid(layer, 200)
    probes += [avg_resids]

    # resid_real = cache[f"blocks.{layer}.ln1.hook_normalized"]
    # act_name = f"blocks.{layer}.ln1.hook_normalized"
    # resid_real = get_activation([act_name], num_games, cache_start)[act_name]
    new_probes = []
    for probe in probes:
        probe = orthogonalize_vectors(probe, new_probes)
        new_probes += [probe]
    probes = new_probes
    # attention pattern
    # TODO: don't use the old cache!
    # attention_pattern : Float[Tensor, "head pos_from pos_to"] = get_attn_pattern(layer, use_attn_pattern_approx, num_games, cache)
    yours_logits_pred_head_pos_sum = t.zeros((num_games, 8, 59)).to(device)
    mine_logits_pred_head_pos_sum = t.zeros((num_games, 8, 59)).to(device)
    for probe in probes:
        if probe.shape[0] == 59:
            logit = einops.einsum(resid_real, probe, "batch pos_to d_model, pos_to d_model -> batch pos_to")
            probe_scales = logit * einops.repeat(probe, "pos_to d_model -> d_model batch pos_to", batch=logit.shape[0])
        else:
            logit = einops.einsum(resid_real, probe, "batch pos_to d_model, d_model -> batch pos_to")
            probe_scales = logit * einops.repeat(probe, "d_model -> d_model batch pos_to", batch=logit.shape[0], pos_to=logit.shape[1])
        probe_after_OV = einops.einsum(probe_scales, OV.AB[layer, :], "d_model batch pos_to, head_idx d_model d_model_out -> d_model_out batch head_idx pos_to")
        yours_logits_pred_head_pos = einops.einsum(probe_after_OV, yours_probe_s_curr, "d_model batch head_idx pos_to, d_model -> batch head_idx pos_to")
        mine_logits_pred_head_pos = einops.einsum(probe_after_OV, mine_probe_s_curr, "d_model batch head_idx pos_to, d_model -> batch head_idx pos_to")
        yours_logits_pred_head_pos_sum += yours_logits_pred_head_pos
        mine_logits_pred_head_pos_sum += mine_logits_pred_head_pos
    yours_logits_pred_head_pos_sum = einops.repeat(yours_logits_pred_head_pos_sum, "batch head_idx pos_to -> batch head_idx pos_from pos_to", pos_from=59) * attention_pattern
    yours_logits_pred = einops.reduce(yours_logits_pred_head_pos_sum, "batch head_idx pos_from pos_to -> batch pos_from", "sum")
    mine_logits_pred_head_pos_sum = einops.repeat(mine_logits_pred_head_pos_sum, "batch head_idx pos_to -> batch head_idx pos_from pos_to", pos_from=59) * attention_pattern
    mine_logits_pred = einops.reduce(mine_logits_pred_head_pos_sum, "batch head_idx pos_from pos_to -> batch pos_from", "sum")
    yours_logits_pred, mine_logits_pred = add_bias(yours_logits_pred, mine_logits_pred, layer, tile_tuple)
    logits_pred_diff = yours_logits_pred - mine_logits_pred
    return logits_pred_diff, yours_logits_pred_head_pos_sum, mine_logits_pred_head_pos_sum

def get_results_yours_mine_pred(layer, tile_tuple, get_probes_function, cache_start=50, num_games = 200, use_attn_pattern_approx = True, inference_size=1000):
    mask_list = []
    correct_list = []
    for batch in range(cache_start, cache_start + num_games, inference_size):
        fake_cache = get_activation([utils.get_act_name("attn_out", layer), f"blocks.{layer}.ln1.hook_normalized", utils.get_act_name("pattern", layer)], num_games=inference_size, start=batch)
        resid_real = fake_cache[f"blocks.{layer}.ln1.hook_normalized"]
        attn_out = fake_cache[utils.get_act_name("attn_out", layer)]
        attention_pattern = fake_cache[utils.get_act_name("pattern", layer)]
        if use_attn_pattern_approx:
            attention_pattern = get_attn_pattern(layer, use_attn_pattern_approx, batch_size=inference_size)
        probes = get_probes_function(layer, tile_tuple)
        yours_logits, mine_logits, empty_logits = get_logits_real(layer, tile_tuple, attn_out=attn_out)
        all_logits = t.stack([empty_logits, mine_logits, yours_logits], dim=-1)
        mask = all_logits.argmax(dim=-1) != 0
        logits_diff = yours_logits - mine_logits
        logits_pred_diff, yours_logits_pred_head_pos, mine_logits_pred_head_pos = get_logits_pred_diff(probes, layer, inference_size, resid_real, attention_pattern)
        correct = (logits_pred_diff > 0) == (logits_diff > 0)
        mask_list.append(mask)
        correct_list.append(correct)
    # if only_mind_changes:
    #     mind_change_mask = get_mind_change_mask(cache, layer, tile_tuple)
    #     mask = mask * mind_change_mask
    mask = t.cat(mask_list, dim=0)
    correct = t.cat(correct_list, dim=0)
    return mask, correct, yours_logits_pred_head_pos, mine_logits_pred_head_pos


def get_mean_of_yours_mine_pred(mask, correct):
    mean_result = t.zeros((59, )).to(device)
    not_blank_count = t.zeros((59, )).to(device)
    correct_count = t.zeros((59, )).to(device)
    for i in range(59):
        mean_result[i] = correct[:, i][mask[:, i] == 1].float().mean()
        not_blank_count[i] = mask[:, i].float().sum()
        correct_count[i] = correct[:, i][mask[:, i] == 1].float().sum()
    return mean_result, not_blank_count, correct_count


def get_probes_flipped_yours_mine(layer, tile_tuple):
    flipped_probe_normalized_s : Float[Tensor, "d_model"] = flipped_probe_normalized[:, layer, *tile_tuple]
    yours_probe_s_prev = get_probe(layer-1, "linear", "post")[0, :, *tile_tuple, YOURS].detach()
    mine_probe_s_prev = get_probe(layer-1, "linear", "post")[0, :, *tile_tuple, MINE].detach()
    probes = [flipped_probe_normalized_s, yours_probe_s_prev, mine_probe_s_prev]
    return probes

def get_probes_flipped(layer, tile_tuple):
    flipped_probe_normalized_s : Float[Tensor, "d_model"] = flipped_probe_normalized[:, layer, *tile_tuple]
    probes = [flipped_probe_normalized_s]
    return probes


def all_probes(layer, tile_tuple):
    probes = []
    probes += get_probes_flipped_yours_mine(layer, tile_tuple)
    probes += get_neuron_weights_as_probes(layer, tile_tuple)
    return probes

if __name__ == "__main__":
    num_games = 50000
    inference_size=2000
    script_name = sys.argv[0]
    if len(sys.argv) < 2:
        print(f"Usage: {script_name} <dataset>")
        sys.exit(1)
    layer = int(sys.argv[1])
    # layer = 1
    cache_start = 200
    probe_function_names = ["get_probes_flipped_yours_mine", "get_probes_flipped", "all_probes"]
    for probe_idx, get_probes_function in enumerate([get_probes_flipped_yours_mine, get_probes_flipped, all_probes]):
        final_scores = t.zeros((8, 8, 2, 59))
        final_correct_count = t.zeros((8, 8, 2, 59))
        final_not_blank_count = t.zeros((8, 8, 2, 59))
        probe_function_name = probe_function_names[probe_idx]
        for row in range(8):
            for col in range(8):
                label = tuple_to_label((row, col))
                for attn_pattern_approx in [True, False]:
                    tile_tuple = (row, col)
                    tile_label = tuple_to_label(tile_tuple)
                    mask, correct, yours_logits_pred_head_pos, mine_logits_pred_head_pos = get_results_yours_mine_pred(
                        layer,
                        tile_tuple,
                        get_probes_function,
                        cache_start=cache_start,
                        num_games=num_games,
                        use_attn_pattern_approx=attn_pattern_approx,
                        inference_size=inference_size
                    )
                    mean, not_blank_count, correct_count = get_mean_of_yours_mine_pred(mask, correct)
                    final_scores[row, col, int(attn_pattern_approx)] = mean
                    final_not_blank_count[row, col, int(attn_pattern_approx)] = not_blank_count
                    final_correct_count[row, col, int(attn_pattern_approx)] = correct_count
                    print(f"Layer: {layer}, Label: {label}, Attn Pattern Approx: {attn_pattern_approx}, Mean at pos 10: {mean[10]}")
        # save final_scores
        with open(f"attn_approx_results/{probe_function_name}/attn_approx_mean_L{layer}.pkl", "wb") as file:
            pickle.dump(final_scores, file)
        # save not_blank_count
        with open(f"attn_approx_results/{probe_function_name}/attn_approx_not_blank_count_L{layer}.pkl", "wb") as file:
            pickle.dump(final_not_blank_count, file)
        # save correct_count
        with open(f"attn_approx_results/{probe_function_name}/attn_approx_correct_count_L{layer}.pkl", "wb") as file:
            pickle.dump(final_correct_count, file)
    print("Done!")