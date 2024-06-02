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

from training_utils import get_state_stack_one_hot_flipped, get_state_stack_one_hot_num_flipped, get_state_stack_one_hot_even_odd_flipped

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

def plot_probe_outputs(layer, game_index, move, **kwargs):
    residual_stream = focus_cache["resid_post", layer][game_index, move]
    # print("residual_stream", residual_stream.shape)
    probe_out = einops.einsum(residual_stream, linear_probe, "d_model, d_model row col options -> row col options")
    probabilities = probe_out.softmax(dim=-1)
    plot_square_as_board(probabilities, facet_col=2, facet_labels=["P(Empty)", "P(Their's)", "P(Mine)"], **kwargs)

import wandb

# We train different Probes on the Model
# Probe1. Detect if a Tile has been Flipped this turn TODO: Look at the Code from Andrew Lee (e. g. what exactly was the loss)
# Probe2. Detect how many times each Tile has been Flipped from 1 to 20 or something => Look at cosine similiarity of input weights of the output neuron to see if it's the same direction scaled
# Probe3. Detect if a Tile has been flipped an odd or even number of times
# Probe4. Detect what the first played tile on each square was (Mine vs. Theirs)
# Probe5. Detect what the first played tile on each square was (Black vs. White)
# Probe6. The Piece on the Tile is the same or different to what was first played?

@dataclass
class ProbeTrainingArgs():

    # Which layer, and which positions in a game sequence to probe
    layer: int = 6
    pos_start: int = 5
    pos_end: int = model.cfg.n_ctx - 5
    length: int = pos_end - pos_start
    alternating: Tensor = t.tensor([1 if i%2 == 0 else -1 for i in range(length)], device=device)

    # Game state (options are blank/mine/theirs)
    options: int = 3
    rows: int = 8
    cols: int = 8

    # Standard training hyperparams
    max_epochs: int = 8
    num_games_train: int = 30000
    num_games_val: int = 10000
    num_games_test: int = 10000

    # Hyperparams for optimizer
    batch_size: int = 256
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    wd: float = 0.01

    # Misc.
    probe_name: str = "main_linear_probe"

    # The modes are "black to play / odd moves", "white to play / even moves", and "all moves"
    modes = 3

    # wnadb
    wandb_project: str = "Othello-GPT-Probes"

    # probe_kind
    get_state_stack_one_hot = None
    relevant_mode = 2
    

    # Code to get randomly initialized probe
    def setup_linear_probe(self, model: HookedTransformer):
        linear_probe = t.randn(
            self.modes, model.cfg.d_model, self.rows, self.cols, self.options, requires_grad=False, device=device
        ) / np.sqrt(model.cfg.d_model)
        linear_probe.requires_grad = True
        return linear_probe


# set torch seed to 42
t.manual_seed(42)

'''n_indices = 4
full_train_indices = t.randperm(4)[:n_indices]
full_train_indices = einops.rearrange(full_train_indices, "(batch_idx game_idx) -> batch_idx game_idx", game_idx=2)
indeces = full_train_indices[0]
games_str = board_seqs_string[indeces.cpu()]
state_stack = seq_to_state_stack(games_str[0].tolist())
imshow(
    state_stack[:16],
    facet_col=0,
    facet_col_wrap=8,
    facet_labels=[f"Move {i}" for i in range(1, 17)],
    title="First 16 moves of first game",
    color_continuous_scale="Greys",
)
args = ProbeTrainingArgs()
get_state_stack_one_hot_num_flipped(games_str, args)'''


class LinearProbeTrainer:
    def __init__(self, model: HookedTransformer, args: ProbeTrainingArgs):
        self.model = model
        self.args = args
        self.linear_probe = args.setup_linear_probe(model)

    def training_step(self, indices: Int[Tensor, "game_idx"], train_or_val="train") -> t.Tensor:

        # Get the game sequences and convert them to state stacks
        games_int = board_seqs_int[indices.cpu()]
        games_str = board_seqs_string[indices.cpu()]
        # state_stack = t.stack([t.tensor(seq_to_state_stack(game_str.tolist())) for game_str in games_str])
        # state_stack = state_stack[:, self.args.pos_start: self.args.pos_end, :, :]
        batch_size = self.args.batch_size
        game_len = self.args.length
        options = self.args.options

        state_stack_one_hot = args.get_state_stack_one_hot(games_str, self.args).to(device)

        # games_int = tensor of game sequences, each of length 60
        # This is the input to our model
        assert isinstance(games_int, Int[Tensor, f"batch={batch_size} full_game_len=60"])

        # state_stack_one_hot = tensor of one-hot encoded states for each game
        # We'll multiply this by our probe's estimated log probs along the `options` dimension, to get probe's estimated log probs for the correct option
        assert isinstance(state_stack_one_hot, Int[Tensor, f"batch={batch_size} game_len={game_len} rows=8 cols=8 options={options}"])

        # SOLUTION
        with t.inference_mode():
            _, cache = model.run_with_cache(
                games_int[:, :-1].to(device),
                return_type=None,
                names_filter=lambda name: name.endswith("resid_post")
            )
            resid_post = cache["resid_post", self.args.layer][:, self.args.pos_start: self.args.pos_end]

        probe_out = einops.einsum(
            resid_post,
            self.linear_probe,
            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
        )

        probe_log_probs = probe_out.log_softmax(-1)
        probe_correct_log_probs = einops.reduce(
            probe_log_probs * state_stack_one_hot,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean"
        ) * self.args.options # Multiply to correct for the mean over options
        loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum() # note that "even" means odd in the game framing, since we offset by 5 moves lol
        loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
        loss_all = -probe_correct_log_probs[2, :].mean(0).sum()
        loss = loss_even + loss_odd + loss_all

        if train_or_val == "train":
            wandb.log({f"{train_or_val}_loss_even": loss_even.item(), f"{train_or_val}_loss_odd": loss_odd.item(), f"{train_or_val}_loss_all": loss_all.item(), f"{train_or_val}_loss": loss.item()}, step=self.step)
            self.step += 1

        return loss
        

    def shuffle_indices(self, start, num_games):
        '''
        Helper function
        '''
        n_indices = num_games - (num_games % self.args.batch_size)
        full_train_indices = t.randperm(num_games)[:n_indices] + start
        full_train_indices = einops.rearrange(full_train_indices, "(batch_idx game_idx) -> batch_idx game_idx", game_idx=self.args.batch_size)
        return full_train_indices
    
    def shuffle_training_indices(self):
        '''
        Returns the tensors you'll use to index into the training data.
        '''
        return self.shuffle_indices(0, self.args.num_games_train)

    def shuffle_val_indices(self):
        '''
        Returns the tensors you'll use to index into the validation data.
        '''
        return self.shuffle_indices(self.args.num_games_train, self.args.num_games_val)

    def save_linear_probe(self, path):
        t.save(self.linear_probe, path)


    def train(self):

        self.step = 0
        # wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.probe_name, config=args)

        optimizer = t.optim.AdamW([self.linear_probe], lr=self.args.lr, betas=self.args.betas, weight_decay=self.args.wd)

        for epoch in range(args.max_epochs):
            full_train_indices = trainer.shuffle_training_indices()
            progress_bar_train = tqdm(full_train_indices)
            for indices in progress_bar_train:
                loss = trainer.training_step(indices)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress_bar_train.set_description(f"Train_Loss = {loss:.4f}")

            val_loss = 0
            count = 0
            full_val_indices = trainer.shuffle_val_indices()
            progress_bar_val = tqdm(full_val_indices)
            with t.inference_mode():
                for indices in progress_bar_val:
                    count += 1
                    val_loss += trainer.training_step(indices, "val")
                    progress_bar_val.set_description(f"Test_Loss = {loss:.4f}")
            
            wandb.log({"val_loss": val_loss / count}, step=self.step)

        wandb.finish()
        self.save_linear_probe(f"models/{args.probe_name}.pth")
        print("Probe Trained and Saved")


# Train Flipped Probe
'''for probe in ["flipped", "num_flipped", "even_odd_flipped"]:
    for layer in range(8):
        args = ProbeTrainingArgs()
        args.layer = layer
        args.wandb_project = "Othello-GPT-Probes"
        args.probe_name = probe
        if probe == "flipped":
            args.get_state_stack_one_hot = get_state_stack_one_hot_flipped
            args.num_outputs = 2
        elif probe == "num_flipped":
            args.get_state_stack_one_hot = get_state_stack_one_hot_num_flipped
            args.num_outputs = 18
        elif probe == "even_odd_flipped":
            args.get_state_stack_one_hot = get_state_stack_one_hot_even_odd_flipped
            args.num_outputs = 2
        trainer = LinearProbeTrainer(model, args)
        trainer.train()'''

if __name__ == "__main__":
    # TODO: Add Probes for the other 5 Probes
    for probe in ["num_flipped", "even_odd_flipped", "flipped"]:
        for layer in range(8):
            args = ProbeTrainingArgs()
            args.layer = layer
            args.wandb_project = "Othello-GPT-Probes-real-test"
            args.probe_name = f"{probe}_L{layer}"
            args.modes = 3
            if probe == "flipped":
                args.get_state_stack_one_hot = get_state_stack_one_hot_flipped
                args.options = 2
                args.relevant_mode = 0
            elif probe == "num_flipped":
                args.get_state_stack_one_hot = get_state_stack_one_hot_num_flipped
                args.options = 18
                args.relevant_mode = 0
            elif probe == "even_odd_flipped":
                args.get_state_stack_one_hot = get_state_stack_one_hot_even_odd_flipped
                args.options = 2
                args.relevant_mode = 0
            trainer = LinearProbeTrainer(model, args)
            trainer.train()