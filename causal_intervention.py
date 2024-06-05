'''import os, sys
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
import utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

import wandb

from training_utils import get_state_stack_one_hot_flipped, get_state_stack_one_hot_num_flipped, get_state_stack_one_hot_even_odd_flipped

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

model = utils.load_model(device)

section_dir = Path.cwd()
assert section_dir.name == "interpretability"

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

sys.path.append(str(OTHELLO_MECHINT_ROOT))

# Load board data as ints (i.e. 0 to 60)
board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)

assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
assert board_seqs_int.max() == 60

layer = 6
game_index = 30001
move = 13


imshow(
    focus_states[game_index, :16],
    facet_col=0,
    facet_col_wrap=8,
    facet_labels=[f"Move {i}" for i in range(1, 17)],
    title="First 16 moves of first game",
    color_continuous_scale="Greys",
)'''
# plot_probe_outputs(layer, game_index, move)
# plot_probe_outputs_intervention(layer, game_index, move)






'''cell_r = 5
cell_c = 4
print(f"Flipping the color of cell {'ABCDEFGH'[cell_r]}{cell_c}")

board = OthelloBoardState()
board.update(moves.tolist())
board_state = board.state.copy()
valid_moves = board.get_valid_moves()
flipped_board = copy.deepcopy(board)
flipped_board.state[cell_r, cell_c] *= -1
flipped_valid_moves = flipped_board.get_valid_moves()

newly_legal = [string_to_label(move) for move in flipped_valid_moves if move not in valid_moves]
newly_illegal = [string_to_label(move) for move in valid_moves if move not in flipped_valid_moves]
print("newly_legal", newly_legal)
print("newly_illegal", newly_illegal)


def apply_scale(
    resid: Float[Tensor, "batch=1 seq d_model"],
    flip_dir: Float[Tensor, "d_model"],
    scale: int,
    pos: int
) -> Float[Tensor, "batch=1 seq d_model"]:
    # SOLUTION
    flip_dir_normed = flip_dir / flip_dir.norm()

    alpha = resid[0, pos] @ flip_dir_normed
    resid[0, pos] -= (scale+1) * alpha * flip_dir_normed

    return resid


tests.test_apply_scale(apply_scale)

flip_dir = my_probe[:, cell_r, cell_c]

big_flipped_states_list = []
layer = 4
scales = [0, 1, 2, 4, 8, 16]

# Iterate through scales, generate a new facet plot for each possible scale
for scale in scales:

    # Hook function which will perform flipping in the "F4 flip direction"
    def flip_hook(resid: Float[Tensor, "batch=1 seq d_model"], hook: HookPoint):
        return apply_scale(resid, flip_dir, scale, pos)

    # Calculate the logits for the board state, with the `flip_hook` intervention
    # (note that we only need to use :pos+1 as input, because of causal attention)
    flipped_logits: Tensor = model.run_with_hooks(
        focus_games_int[game_index:game_index+1, :pos+1],
        fwd_hooks=[
            (utils.get_act_name("resid_post", layer), flip_hook),
        ]
    ).log_softmax(dim=-1)[0, pos]

    flip_state = t.zeros((64,), dtype=t.float32, device=device) - 10.
    flip_state[stoi_indices] = flipped_logits.log_softmax(dim=-1)[1:]
    big_flipped_states_list.append(flip_state)

flip_state_big = t.stack(big_flipped_states_list)
state_big = einops.repeat(state.flatten(), "d -> b d", b=6)
color = t.zeros((len(scales), 64)).cuda() + 0.2
for s in newly_legal:
    color[:, to_string(s)] = 1
for s in newly_illegal:
    color[:, to_string(s)] = -1

scatter(
    y=state_big,
    x=flip_state_big,
    title=f"Original vs Flipped {string_to_label(8*cell_r+cell_c)} at Layer {layer}",
    # labels={"x": "Flipped", "y": "Original"},
    xaxis="Flipped",
    yaxis="Original",

    hover=[f"{r}{c}" for r in "ABCDEFGH" for c in range(8)],
    facet_col=0, facet_labels=[f"Translate by {i}x" for i in scales],
    color=color, color_name="Newly Legal", color_continuous_scale="Geyser"
)'''