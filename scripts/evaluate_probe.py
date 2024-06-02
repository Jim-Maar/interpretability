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

import wandb

from training_utils import get_state_stack_one_hot_flipped, get_state_stack_one_hot_num_flipped, get_state_stack_one_hot_even_odd_flipped

device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)

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

# import f1 score
from sklearn.metrics import f1_score as multiclass_f1_score

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