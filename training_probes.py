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
import utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import dataclasses
from dataclasses import dataclass

from tqdm import tqdm

import wandb

from training_utils import (
    get_state_stack_one_hot_flipped,
    get_state_stack_one_hot_num_flipped,
    get_state_stack_one_hot_even_odd_flipped,
    get_state_stack_one_hot_first_tile_places_black_white,
    get_state_stack_one_hot_first_tile_places_mine_theirs,
    get_state_stack_one_hot_empty_yours_mine,
    get_state_stack_one_hot_placed_and_flipped,
    get_state_stack_one_hot_placed_and_flipped_stripe,
    get_state_stack_one_hot_placed)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"device: {device}")

MAIN = __name__ == "__main__"

model = utils.load_model(device)

section_dir = Path.cwd()
assert section_dir.name == "interpretability"

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

sys.path.append(str(OTHELLO_MECHINT_ROOT))

DEBUG = False
if DEBUG:
    DATASET = "small"
else:
    DATASET = "big"

if DATASET == "small":
    # Load board data as ints (i.e. 0 to 60)
    board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
    # Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
    board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)
elif DATASET == "big":
    # Load board data as ints (i.e. 0 to 60)
    board_seqs_int = t.load(
        os.path.join(
            section_dir,
            "data/board_seqs_int_train.pth",
        )
    )
    # Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
    board_seqs_string = t.load(
        os.path.join(
            section_dir,
            "data/board_seqs_string_train.pth",
        )
    )
else:
    raise ValueError("Invalid DATASET")

assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
assert board_seqs_int.max() == 60

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
    pos_start: int = 0
    pos_end: int = 59
    length: int = pos_end - pos_start
    alternating: Tensor = t.tensor([1 if i%2 == 0 else -1 for i in range(length)], device=device)

    # Game state (options are blank/mine/theirs)
    options: int = 3
    rows: int = 8
    cols: int = 8

    # Standard training hyperparams
    max_epochs: int = 1
    if DATASET == "small":
        num_games_train: int = 6
        num_games_val: int = 0
        num_games_test: int = 0
    else:
        num_games_train: int = 80000
        num_games_val: int = 0
        num_games_test: int = 0

    # Hyperparams for optimizer
    if DEBUG:
        batch_size: int = 2
    else:
        batch_size: int = 256
    lr: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.99)
    wd: float = 0.01

    # Misc.
    probe_name: str = "main_linear_probe"
    full_probe_name: str = "main_linear_probe_L6"

    # The modes are "black to play / odd moves", "white to play / even moves", and "all moves"
    modes = 3

    # wnadb
    wandb_project: str = "Othello-GPT-Probes"

    # probe_kind
    get_state_stack_one_hot = None

    # training
    has_bias = False

    cache_position = "resid_post"

    multi_label = False
    

    # Code to get randomly initialized probe
    def setup_linear_probe(self, model: HookedTransformer):
        linear_probe = t.randn(
            self.modes, model.cfg.d_model, self.rows, self.cols, self.options, requires_grad=False, device=device
        ) / np.sqrt(model.cfg.d_model)
        linear_probe.requires_grad = True
        return linear_probe

    def setup_bias(self, model: HookedTransformer):
        bias = t.randn(
            self.modes, self.rows, self.cols, self.options, requires_grad=False, device=device
        ) / np.sqrt(model.cfg.d_model)
        bias.requires_grad = True
        return bias


# set torch seed to 42
t.manual_seed(42)

class LinearProbeTrainer:
    def __init__(self, model: HookedTransformer, args: ProbeTrainingArgs):
        self.model = model
        self.args = args
        self.linear_probe = args.setup_linear_probe(model)
        if args.has_bias:
            self.bias = args.setup_bias(model)

    def training_step(self, indices: Int[Tensor, "game_idx"], train_or_val="train") -> t.Tensor:

        # Get the game sequences and convert them to state stacks
        games_int = board_seqs_int[indices.cpu()]
        games_str = board_seqs_string[indices.cpu()]
        # state_stack = t.stack([t.tensor(seq_to_state_stack(game_str.tolist())) for game_str in games_str])
        # state_stack = state_stack[:, self.args.pos_start: self.args.pos_end, :, :]
        batch_size = self.args.batch_size
        game_len = self.args.length
        options = self.args.options

        state_stack_one_hot = args.get_state_stack_one_hot(games_str).to(device)
        assert state_stack_one_hot.shape == (batch_size, 60, 8, 8, options)
        state_stack_one_hot = state_stack_one_hot[:, args.pos_start:args.pos_end, :, :, :]
        state_stack_one_hot = state_stack_one_hot.to(dtype=t.long)

        # games_int = tensor of game sequences, each of length 60
        # This is the input to our model
        assert isinstance(games_int, Int[Tensor, f"batch={batch_size} full_game_len=60"])

        # state_stack_one_hot = tensor of one-hot encoded states for each game
        # We'll multiply this by our probe's estimated log probs along the `options` dimension, to get probe's estimated log probs for the correct option
        assert isinstance(state_stack_one_hot, Int[Tensor, f"batch={batch_size} game_len={args.pos_end - args.pos_start} rows=8 cols=8 options={options}"])

        # SOLUTION
        with t.inference_mode():
            _, cache = model.run_with_cache(
                games_int[:, :-1].to(device),
                return_type=None,
                names_filter=lambda name: name.endswith(args.cache_position)
            )
            resid = cache[args.cache_position, self.args.layer][:, self.args.pos_start: self.args.pos_end]

        probe_out = einops.einsum(
            resid.clone().to(device),
            self.linear_probe,
            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
        )
        if self.args.has_bias:
            probe_out += einops.rearrange(self.bias, 'modes rows cols options -> modes 1 1 rows cols options')
        
        if not args.multi_label:
            probe_log_probs = probe_out.log_softmax(-1)
        else:
            probe_log_probs = probe_out.sigmoid().log()
        probe_correct_log_probs = einops.reduce(
            probe_log_probs * state_stack_one_hot,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean"
        ) * self.args.options # Multiply to correct for the mean over options
        # loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum() # note that "even" means odd in the game framing, since we offset by 5 moves lol
        # loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
        loss = -probe_correct_log_probs[0, :].mean(0).sum()
        # loss = loss_even + loss_odd + loss_all

        # if train_or_val == "train" and self.step % 10 == 0 and not DEBUG:
        #     wandb.log({f"{train_or_val}_loss_even": loss_even.item(), f"{train_or_val}_loss_odd": loss_odd.item(), f"{train_or_val}_loss_all": loss_all.item(), f"{train_or_val}_loss": loss.item()}, step=self.step)
        if train_or_val == "train" and self.step % 10 == 0 and not DEBUG:
            wandb.log({f"{train_or_val}_loss": loss.item()}, step=self.step)
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
        folder_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        t.save(self.linear_probe, path)

    def save_bias(self, path):
        assert(self.args.has_bias)
        t.save(self.bias, path)


    def train(self):
        print(f"Training Probe: {self.args.full_probe_name}")
        self.step = 0
        # wandb.login(key=os.environ["WANDB_API_KEY"])
        if not DEBUG:
            wandb.login()
            wandb.init(project=args.wandb_project, name=args.full_probe_name, config=args)

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

            '''val_loss = 0
            count = 0
            full_val_indices = trainer.shuffle_val_indices()
            progress_bar_val = tqdm(full_val_indices)
            with t.inference_mode():
                for indices in progress_bar_val:
                    count += 1
                    val_loss += trainer.training_step(indices, "val")
                    progress_bar_val.set_description(f"Test_Loss = {loss:.4f}")
            
            wandb.log({"val_loss": val_loss / count}, step=self.step)'''

        if not DEBUG:
            wandb.finish()
        module_name = args.cache_position.split("_")[1]
        self.save_linear_probe(f"probes/{module_name}/{args.probe_name}/{args.full_probe_name}.pth")
        if args.has_bias:
            self.save_bias(f"probes/{module_name}/{args.probe_name}/{args.full_probe_name}_bias.pth")
        print("Probe Trained and Saved")

if __name__ == "__main__":
    # TODO: Add Probes for the other 5 Probes
    # TODO: Change position of where first and last part are removed
    # TODO: Use big dataset instead of small dataset
    '''for probe in ["mine_theirs_first_tile", "num_flipped_with_bias", "even_odd_flipped", "num_flipped", "flipped", "black_white_first_tile"]:
        for layer in range(8):
            args = ProbeTrainingArgs()
            args.layer = layer
            args.wandb_project = "Othello-GPT-Probes"
            args.probe_name = probe
            args.full_probe_name = f"{probe}_L{layer}"
            args.modes = 3
            if probe == "flipped":
                args.get_state_stack_one_hot = get_state_stack_one_hot_flipped
                args.options = 2
                args.relevant_mode = 0
            elif probe == "num_flipped":
                args.get_state_stack_one_hot = get_state_stack_one_hot_num_flipped
                args.options = 18
                args.relevant_mode = 0
            elif probe == "num_flipped_with_bias":
                args.get_state_stack_one_hot = get_state_stack_one_hot_num_flipped
                args.options = 18
                args.relevant_mode = 0
                args.has_bias = True
            elif probe == "even_odd_flipped":
                args.get_state_stack_one_hot = get_state_stack_one_hot_even_odd_flipped
                args.options = 2
                args.relevant_mode = 0
            elif probe == "black_white_first_tile":
                args.get_state_stack_one_hot = get_state_stack_one_hot_first_tile_places_black_white
                args.options = 3
                args.relevant_mode = 0
            elif probe == "mine_theirs_first_tile":
                args.get_state_stack_one_hot = get_state_stack_one_hot_first_tile_places_mine_theirs
                args.options = 3
                args.relevant_mode = 0
            trainer = LinearProbeTrainer(model, args)
            trainer.train()'''
    args = ProbeTrainingArgs()
    for probe in ["placed_and_flipped", "placed_and_flipped_stripe", "placed"]:
        for cache_position in ["resid_mid", "resid_post"]:
            for layer in range(8):
                args = ProbeTrainingArgs()
                args.layer = layer
                args.wandb_project = "Othello-GPT-Placed-and-Flipped-Probes"
                args.cache_position = cache_position
                args.probe_name = probe
                args.full_probe_name = f"resid_{layer}_{probe}"
                args.modes = 1
                if probe == "placed_and_flipped":
                    args.multi_label = True
                    args.get_state_stack_one_hot = get_state_stack_one_hot_placed_and_flipped
                    args.options = 8
                elif probe == "placed_and_flipped_stripe":
                    args.multi_label = True
                    args.get_state_stack_one_hot = get_state_stack_one_hot_placed_and_flipped_stripe
                    args.options = 8
                elif probe == "placed":
                    args.get_state_stack_one_hot = get_state_stack_one_hot_placed
                    args.options = 2
                trainer = LinearProbeTrainer(model, args)
                trainer.train()