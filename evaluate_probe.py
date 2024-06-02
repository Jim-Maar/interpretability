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
import part6_othellogpt.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)

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
assert section_dir.name == "interpretability"

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

num_games = 50
focus_games_int = board_seqs_int[:num_games]
focus_games_string = board_seqs_string[:num_games]

focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))
focus_logits.shape

full_linear_probe = t.load(OTHELLO_MECHINT_ROOT / "main_linear_probe.pth", map_location=device)

rows = 8
cols = 8
options = 3
assert full_linear_probe.shape == (3, cfg.d_model, rows, cols, options)

black_to_play_index = 0
white_to_play_index = 1
blank_index = 0
their_index = 1
my_index = 2

# Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")
linear_probe = t.zeros(cfg.d_model, rows, cols, options, device=device)
linear_probe[..., blank_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 0] + full_linear_probe[white_to_play_index, ..., 0])
linear_probe[..., their_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 1] + full_linear_probe[white_to_play_index, ..., 2])
linear_probe[..., my_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 2] + full_linear_probe[white_to_play_index, ..., 1])

layer = 6
game_index = 0
move = 29

BLANK = 0
FILLED_UNAFFECTED = 1
FLIPPED = 2
PLAYED = 3
UNAFFECTED = 4

from training_utils import get_state_stack_one_hot_flipped, get_state_stack_one_hot_num_flipped, get_state_stack_one_hot_even_odd_flipped
from training_utils import build_state_stack_flipped, state_stack_to_one_hot_flipped
# from othello_world.mechanistic_interpretability.mech_interp_othello_utils import load_hooked_model

# function loars a linear_probe from the models folder as a pytorch model
def load_linear_probe(model_path: str, device: t.device) -> t.Tensor:
    return t.load(model_path, map_location=device)

def evaluate(probe_dir):
    """
    Evaluate probe model.
    """
    global board_seqs_int
    global board_seqs_string
    start_of_test_size = 30000
    test_size = 10000
    board_seqs_int = board_seqs_int[start_of_test_size : start_of_test_size + test_size]
    board_seqs_string = board_seqs_string[start_of_test_size : start_of_test_size + test_size]

    games_int = board_seqs_int
    games_str = board_seqs_string
    all_indices = t.arange(test_size)
    batch_size = 128
    orig_state_stack = build_state_stack_flipped(games_str)

    pos_start = 0
    pos_end = model.cfg.n_ctx - 0

    for layer in range(8):
        linear_probe = t.load(
            os.path.join(
                probe_dir,
                f"flipped_L{layer}.pth",
            )
        )
        accs = []
        per_timestep_num_correct = t.zeros((59, 8, 8))
        all_preds = []
        all_groundtruths = []
        for idx in range(0, test_size, batch_size):
            indices = all_indices[idx : idx + batch_size]
            _games_int = games_int[indices]

            state_stack = orig_state_stack[indices, pos_start:pos_end, :, :]
            state_stack_one_hot = state_stack_to_one_hot_flipped(state_stack).cuda()

            with t.inference_mode():
                logits, cache = model.run_with_cache(
                    _games_int.cuda()[:, :-1], return_type="logits"
                )
                resid_post = cache["resid_post", layer][:, pos_start:pos_end]
            probe_out = einops.einsum(
                resid_post.clone(),
                linear_probe,
                "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
            )

            # [256, 51, 8, 8]
            flipped_preds = probe_out[0].argmax(-1)
            groundtruth = state_stack_one_hot.argmax(-1)
            test_results = flipped_preds == groundtruth
            test_acc = (test_results.sum() / test_results.numel()).item()
            per_timestep_num_correct += test_results[0].sum(0).cpu()
            all_preds.append(flipped_preds)
            all_groundtruths.append(groundtruth)
            accs.append(test_acc * indices.shape[0])

        _all_preds = t.cat(all_preds, dim=0)
        _all_gt = t.cat(all_groundtruths, dim=0)
        f1_score = multiclass_f1_score(
            _all_gt.cpu().view(-1), _all_preds.cpu().view(-1)
        )
        print(f"Layer: {layer} F1_score: {f1_score}")


evaluate("models")