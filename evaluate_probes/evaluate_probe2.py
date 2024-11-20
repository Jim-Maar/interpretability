from training_utils import (
    get_state_stack_one_hot_empty_yours_mine,
    get_state_stack_one_hot_flipped,
    get_state_stack_one_hot_placed,
    get_state_stack_one_hot_placed_and_flipped,
    get_state_stack_one_hot_placed_and_flipped_stripe,
    get_state_stack_num_flipped,
    get_state_stack_one_hot_legal,
    get_state_stack_one_hot_accesible,
    get_state_stack_one_hot_first_tile_places_black_white,
    get_state_stack_one_hot_even_odd_flipped,
    get_state_stack_one_hot_num_flipped,
    get_state_stack_one_hot_first_tile_places_mine_theirs,
)

import os
import torch as t
from torch import Tensor
from typing import Tuple, Callable
from einops import rearrange
import einops
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import pandas as pd
from tqdm import tqdm
import wandb

device = t.device("cuda" if t.cuda.is_available() else "cpu")

from utils import *

def get_probes(probe_type : str = "linear", probe_module : str = "post", probe_directory : str = "probes") -> Tensor:
    probes = []
    for layer in range(8):
        path = f"{probe_directory}/{probe_module}/{probe_type}/resid_{layer}_{probe_type}.pth"
        if os.path.exists(path):
            probes += [t.load(path).to(device)]
            continue
        path = f"{probe_directory}/{probe_module}/{probe_type}/{probe_type}_L{layer}.pth"
        if os.path.exists(path):
            probes += [t.load(path).to(device)]
            continue
    return probes

flip_lables = ["flipped", "legal", "placed", "accesible"]

def get_f1score_and_accuracy_detailed(
        probe_type : str,
        probe_module : str,
        probe_directory : str,
        len_data : Int,
        get_state_stack_one_hot_function : Callable,
        batch_size : Int = 128,
        multi_label : bool = True,
        options : Int = 8,
        multi_label_threshold : Float = 0.5
    ) -> Tuple[Tensor, Tensor]:

    f1_scores_int = t.zeros((8)).to(device)
    accuracy_scores_int = t.zeros((8)).to(device)
    recall_scores_int = t.zeros((8)).to(device)
    precision_scores_int = t.zeros((8)).to(device)

    probes = get_probes(probe_type, probe_module, probe_directory)

    state_stacks_flat_list = [[] for _ in range(8)]
    pred_flat_list = [[] for _ in range(8)]
    for i in tqdm(range(0, len_data, batch_size)):
        indeces = t.arange(i, min(i + batch_size, len_data))
        input_seqs = board_seqs_int[indeces]
        input_seqs = input_seqs[:, :-1]
        state_stacks = get_state_stack_one_hot_function(t.Tensor(to_string(input_seqs)).to(dtype=t.long))
        if not multi_label:
            state_stacks = state_stacks.argmax(dim=-1)
        if probe_type in flip_lables:
            state_stacks = (state_stacks == 0).to(dtype=t.long)
        _, cache= model.run_with_cache(
            input_seqs.to(device),
            return_type=None,
            names_filter=lambda name: name in [utils.get_act_name(f"resid_{probe_module}", l) for l in range(8)],
        )
        for layer in range(8):
            probe = probes[layer]
            resid = cache[f"resid_{probe_module}", layer].to(device)
            result = einops.einsum(probe, resid, "modes d_model rows cols options, batch pos d_model -> modes batch pos rows cols options")[0]
            if not multi_label:
                pred = result.argmax(dim=-1)
                if probe_type in flip_lables:
                    pred = (pred == 0).to(dtype=t.long)
                # pred = t.nn.functional.one_hot(pred, num_classes=options)
                state_stacks_flat = einops.rearrange(state_stacks, "batch pos rows cols -> (batch pos rows cols)")
                pred_flat = einops.rearrange(pred, "batch pos rows cols -> (batch pos rows cols)")
            else:
                pred = result
                pred = t.nn.functional.sigmoid(pred)
                pred = (pred > multi_label_threshold).int()
                state_stacks_flat = einops.rearrange(state_stacks, "batch pos rows cols options -> (batch pos rows cols options)")
                pred_flat = einops.rearrange(pred, "batch pos rows cols options -> (batch pos rows cols options)")
            # Now calculate the f1 score over all dimensions
            state_stacks_flat = state_stacks_flat.to(dtype=t.long).cpu()
            pred_flat = pred_flat.to(dtype=t.long).cpu()
            state_stacks_flat_list[layer].append(state_stacks_flat)
            pred_flat_list[layer].append(pred_flat)
    for layer in range(8):
        state_stacks_flat = t.cat(state_stacks_flat_list[layer])
        pred_flat = t.cat(pred_flat_list[layer])
        f1_score_int = f1_score(state_stacks_flat, pred_flat, average="macro")
        accuracy_score_int = accuracy_score(state_stacks_flat, pred_flat)
        # recall_score_int = recall_score(state_stacks_flat, pred_flat, average="macro")
        # precision_score_int = precision_score(state_stacks_flat, pred_flat, average="macro")
        f1_scores_int[layer] = f1_score_int
        accuracy_scores_int[layer] = accuracy_score_int
        # recall_scores_int[layer] = recall_score_int
        # precision_scores_int[layer] = precision_score_int
    return f1_scores_int, accuracy_scores_int # , recall_scores_int, precision_scores_int
            
'''get_f1score_detailed(
    "placed",
    "mid",
    100,
    get_state_stack_one_hot_placed,
    batch_size=90,
    multi_label=False,
    options=2,
    multi_label_threshold=0.999999
)'''


state_stack_one_hot_functions = {
    "linear" : get_state_stack_one_hot_empty_yours_mine,
    "flipped" : get_state_stack_one_hot_flipped,
    "placed" : get_state_stack_one_hot_placed,
    "placed_and_flipped" : get_state_stack_one_hot_placed_and_flipped,
    "placed_and_flipped_stripe" : get_state_stack_one_hot_placed_and_flipped_stripe,
    "legal" : get_state_stack_one_hot_legal,
    "accesible" : get_state_stack_one_hot_accesible,
    "black_white_first_tile" : get_state_stack_one_hot_first_tile_places_black_white,
    "even_odd_flipped" : get_state_stack_one_hot_even_odd_flipped,
    "num_flipped" : get_state_stack_one_hot_num_flipped,
    "mine_theirs_first_tile" : get_state_stack_one_hot_first_tile_places_mine_theirs,
}

probe_option_count = {
    "linear" : 3,
    "flipped" : 2,
    "placed" : 2,
    "placed_and_flipped" : 8,
    "placed_and_flipped_stripe" : 8,
    "legal" : 2,
    "accesible" : 2,
    "black_white_first_tile" : 3,
    "mine_theirs_first_tile" : 3,
    "even_odd_flipped" : 2,
    "num_flipped" : 18,
}

probe_ignore_dimensions = {
    "linear" : [0],
    "flipped" : [1],
    "placed" : [1],
    "placed_and_flipped" : [],
    "placed_and_flipped_stripe" : [],
}

probe_multi_label = {
    "linear" : False,
    "flipped" : False,
    "placed" : False,
    "placed_and_flipped" : True,
    "placed_and_flipped_stripe" : True,
    "legal" : False,
    "accesible" : False,
    "black_white_first_tile" : False,
    "mine_theirs_first_tile" : False,
    "even_odd_flipped" : False,
    "num_flipped" : False,
}

eval_kind = {
    "linear" : "accuracy",
    "flipped" : "f1",
    "placed" : "f1",
    "placed_and_flipped" : "f1",
    "placed_and_flipped_stripe" : "f1",
    "legal" : "f1",
    "accesible" : "f1",
    "black_white_first_tile" : "accuray",
    "mine_theirs_first_tile" : "accuracy",
    "even_odd_flipped" : "accuracy",
    "num_flipped" : "f1",
}

# First_tile_places keine Ahnung ob wie viele optionen es gibt
len_data = 1000
WANDB = True

# Get Scores for all probes, put the results (either accuracy or recally) into a seperate Dataframe and upload to wandb
# Also save tables in Memory and convert to latex
table_dicts = {}
tables = {}

if WANDB:
    wandb.login()
    wandb.init(project="Othello-GPT probe evaluation test", name="Othello-GPT probe evaluation")

for probe_directory in ["probes", "probes_failed"]:
    probe_modules = os.listdir(probe_directory)
    for probe_module in probe_modules:
        probe_types = os.listdir(f"{probe_directory}/{probe_module}")
        # probe_types = ["flipped"]
        for probe_type in probe_types:
            if probe_type not in table_dicts:
                table_keys = ["probe_module"] + [f"layer_{i}" for i in range(8)]
                table_dict = {k : [] for k in table_keys}
                table_dicts[probe_type] = table_dict
            else:
                table_dict = table_dicts[probe_type]
            if probe_type not in state_stack_one_hot_functions:
                continue
            print(probe_type)
            f1_scores_int, accuracy_scores_int = get_f1score_and_accuracy_detailed(
                probe_type,
                probe_module,
                probe_directory,
                len_data,
                state_stack_one_hot_functions[probe_type],
                batch_size=50,
                multi_label=probe_multi_label[probe_type],
                options=probe_option_count[probe_type],
            )
            if eval_kind[probe_type] == "accuracy":
                scores = accuracy_scores_int
            elif eval_kind[probe_type] == "f1":
                scores = f1_scores_int
            # round to 2 decimal places
            scores = t.round(scores, decimals=4)
            table_dict["probe_module"].append(probe_module)
            for i in range(8):
                table_dict[f"layer_{i}"].append(scores[i].item())

for probe_type, table_dict in table_dicts.items():
    table = pd.DataFrame(table_dict)
    tables[probe_type] = table
    if WANDB:
        wandb.log({probe_type : wandb.Table(dataframe=table)})
    print(probe_type)
    print(table)
if WANDB:
    wandb.finish()

for probe_type, table in tables.items():
    print("\label{tab:" + probe_type + "}")
    print(table.to_latex(index=False))