import os, sys
chapter = "chapter1_transformer_interp"
repo = "ARENA_3.0"
chapter_dir = r"./" if chapter in os.listdir() else os.getcwd().split(chapter)[0]
sys.path.append(chapter_dir + f"{chapter}/exercises")

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import torch as t
from torch import Tensor
import numpy as np
import einops
from ipywidgets import interact
import plotly.express as px
from pathlib import Path
import itertools
import random
from IPython.display import display
from typing import List, Union, Optional, Tuple, Callable, Dict
import typeguard
from functools import partial
# from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import f1_score as multiclass_f1_score
import dataclasses
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm.notebook import tqdm
from dataclasses import dataclass
from rich import print as rprint
import pandas as pd

from plotly_utils import imshow
from pathlib import Path
from typing import List, Union, Optional, Tuple, Callable, Dict
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# os.chdir(section_dir)
section_dir = Path.cwd()
assert section_dir.name == "interpretability"

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

sys.path.append(str(OTHELLO_MECHINT_ROOT))
from mech_interp_othello_utils import (
    OthelloBoardState,
    to_int,
    to_string,
    string_to_label,
    str_to_int,
)

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

board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
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
# start = 30000
start = 0
num_games = 200
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

BLANK1 = 0
BLACK = 1
WHITE = -1

EMPTY = 0
YOURS = 1
MINE = 2

FLIPPED = 0
NOT_FLIPPED = 1

PLACED = 0
NOT_PLACED = 1

FLIPPED_TOP = 0
FLIPPED_TOP_RIGHT = 1
FLIPPED_RIGHT = 2
FLIPPED_BOTTOM_RIGHT = 3
FLIPPED_BOTTOM = 4
FLIPPED_BOTTOM_LEFT = 5
FLIPPED_LEFT = 6
FLIPPED_TOP_LEFT = 7

# Load Model
def load_model(device):
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

    sd = transformer_lens.utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
    # champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
    model.load_state_dict(sd)
    return model

alpha = "ABCDEFGH"

# Load Probes

probes = dict()
probe_modules = os.listdir("probes")
for probe_module in probe_modules:
    probe_types = os.listdir(f"probes/{probe_module}")
    for probe_type in probe_types:
        for layer in range(8):
            if device.type == "cpu":
                probe = t.load(f"probes/{probe_module}/{probe_type}/resid_{layer}_{probe_type}.pth", map_location=device).detach()
            else:
                probe = t.load(f"probes/{probe_module}/{probe_type}/resid_{layer}_{probe_type}.pth").to(device).detach()
            probes[(probe_module, probe_type, layer)] = probe

def get_probe(layer : Int = 5, probe_type : str = "linear", probe_module : str = "post"):
    # assert probe_module in ["post", "mid"]
    # assert probe_type in ["linear", "flipped"]
    return probes[(probe_module, probe_type, layer)]

probe_directions = {
    "linear": {
        "empty" : EMPTY,
        "yours" : YOURS,
        "mine" : MINE, 
    },
    "flipped": {
        "flipped" : FLIPPED,
        "not_flipped" : NOT_FLIPPED,
    },
    "placed" : {
        "placed" : PLACED,
        "not_placed" : NOT_PLACED,
    },
    "placed_and_flipped" : {
        "top" : FLIPPED_TOP,
        "top_right" : FLIPPED_TOP_RIGHT,
        "right" : FLIPPED_RIGHT,
        "bottom_right" : FLIPPED_BOTTOM_RIGHT,
        "bottom" : FLIPPED_BOTTOM,
        "bottom_left" : FLIPPED_BOTTOM_LEFT,
        "left" : FLIPPED_LEFT,
        "top_left" : FLIPPED_TOP_LEFT,
    },
    "placed_and_flipped_stripe" : {
        "top" : FLIPPED_TOP,
        "top_right" : FLIPPED_TOP_RIGHT,
        "right" : FLIPPED_RIGHT,
        "bottom_right" : FLIPPED_BOTTOM_RIGHT,
        "bottom" : FLIPPED_BOTTOM,
        "bottom_left" : FLIPPED_BOTTOM_LEFT,
        "left" : FLIPPED_LEFT,
        "top_left" : FLIPPED_TOP_LEFT,
    }
}

probe_directions_list = {
    k : list(v.keys()) for k, v in probe_directions.items()
}

short_cuts = {
    "empty" : "E",
    "yours" : "Y",
    "mine" : "M",
    "flipped" : "F",
    "not_flipped" : "NF",
    "placed" : "P",
    "not_placed" : "NP",
    "top" : "T",
    "top_right" : "TR",
    "right" : "R",
    "bottom_right" : "BR",
    "bottom" : "B",
    "bottom_left" : "BL",
    "left" : "L",
    "top_left" : "TL",
    "linear" : "L",
    "placed_and_flipped" : "PF",
    "placed_and_flipped_stripe" : "PFS",
}

def get_short_cut(name):
    return short_cuts[name]

def get_probe_names():
    return list(probe_directions.keys())

def get_direction_str(direction_int):
    for probe_name in probe_directions:
        for direction_str in probe_directions[probe_name]:
            if probe_directions[probe_name][direction_str] == direction_int:
                return direction_str
    assert(False)

def get_direction_int(directions_str):
    directions_str = directions_str.lower()
    for probe_name in probe_directions:
        if directions_str in probe_directions[probe_name]:
            return probe_directions[probe_name][directions_str]
    assert(False)

def seq_to_state_stack(str_moves):
    """
    Takes a sequence of moves and returns a stack of states for each move with dimensions (num_moves, rows, cols)
    -1 white, 0 blank, 1 black
    """
    board = OthelloBoardState()
    states = []
    for move in str_moves:
        board.umpire(move)
        states.append(np.copy(board.state))
    states = np.stack(states, axis=0)
    return states


# Ploting Functions

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

def plot_probe_outputs(focus_cache, linear_probe, layer, game_index, move, **kwargs):
    residual_stream = focus_cache["resid_post", layer][game_index, move]
    # print("residual_stream", residual_stream.shape)
    # probe_out = einops.einsum(residual_stream, linear_probe, "d_model, d_model row col options -> row col options")
    probe_out = einops.einsum(residual_stream, linear_probe, "d_model, modes d_model row col options -> modes row col options")[0]
    '''if move % 2 == 0:
        probe_out = probe_out[0]
    else:
        probe_out = probe_out[1]'''
    probabilities = probe_out.softmax(dim=-1)
    plot_square_as_board(probabilities, facet_col=2, facet_labels=["P(EMPTY)", "P(YOURS)", "P(MINE)"], **kwargs)

def plot_game(games_str, game_index=0, end_move=16):
    '''
    This shows the game the 0'th move is the first move the display shows the board after the move was made
    '''
    focus_states = seq_to_state_stack(games_str[game_index])
    imshow(
        focus_states[:end_move],
        facet_col=0,
        facet_col_wrap=8,
        facet_labels=[f"Move {i}" for i in range(0, end_move)],
        title="First 16 moves of first game",
        color_continuous_scale="Greys",
        y = [i for i in alpha],
    )

def square_to_tuple(square, is_int=False):
    if is_int:
        square = to_string(square)
    row = square // 8
    col = square % 8
    return (row, col)

def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"

def get_focus_games(model = None, device = "cpu"):
    # Load board data as ints (i.e. 0 to 60)
    board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
    # Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
    board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)

    assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
    assert board_seqs_int.max() == 60

    num_games, length_of_game = board_seqs_int.shape
    start = 30000
    num_games = 50
    focus_games_int = board_seqs_int[start : start + num_games]
    focus_games_string = board_seqs_string[start: start + num_games]

    if model is not None:
        focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))
        return focus_games_int, focus_games_string, focus_logits, focus_cache
    return focus_games_int, focus_games_string

def square_tuple_from_square(square : str):
    return (alpha.index(square[0]), int(square[1])) 

reverse_alpha = ["H", "G", "F", "E", "D", "C", "B", "A"]

def save_plotly_to_html(fig, filename):
    TEMPLATE_PATH = "interactive_plots/template.html"
    assert os.path.exists(TEMPLATE_PATH)
    plotly_jinja_data = {"fig":fig.to_html(full_html=False)}
    with open(filename, "w", encoding="utf-8") as output_file:
        with open(TEMPLATE_PATH) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render(plotly_jinja_data))

def save_plotly_to_png(fig, filename):
    fig.write_image(filename)

def save_plotly(fig, name):
    save_plotly_to_html(fig, f"interactive_plots/{name}.html")
    save_plotly_to_png(fig, f"plots/{name}.png")

def plot_boards_general(x_labels : List[str],
                        y_labels : List[str],
                        boards : Float[Tensor, "x y rows cols"],
                        size_of_board : Int = 200,
                        margin_t : Int = 100,
                        title_text : str = "",
                        static_image : bool = False,
                        save : bool = False):
    # TODO: add attn/mlp only
    # TODO: Change Width and Height accordingly
    boards = boards.flip(2)
    x_len, y_len, rows, cols = boards.shape
    subplot_titles = [f"{y_label}, {x_label}" for y_label in y_labels for x_label in x_labels]
    # subplot_titles = [f"P: {i}, T: {label_list[i]}, L: {j}" for i in range(vis_args.start_pos, vis_args.end_pos) for j in range(vis_args.layers)]
    width = x_len * size_of_board
    height = y_len * size_of_board + margin_t
    vertical_spacing = 70 / height
    fig = make_subplots(rows=y_len, cols=x_len, subplot_titles=subplot_titles, vertical_spacing=vertical_spacing)
    boards_min = boards.min().item()
    boards_max = boards.max().item()
    abs_max = max(abs(boards_min), abs(boards_max))
    for x in range(x_len):
        for y in range(y_len):
            heatmap = go.Heatmap(
                z=boards[x, y].cpu(),
                x=list(range(0, rows)),
                y=reverse_alpha,
                hoverongaps = False,
                zmin=-abs_max,
                zmax=abs_max,
                colorscale="RdBu",
            )
            fig.add_trace(
                heatmap,
                row=y + 1,
                col=x + 1
            )
    fig.layout.update(width=width, height=height, margin_t=margin_t, title_text=title_text) 
    if static_image:
        # count the number of images in the last_plot directory
        num_images = len(list(Path("last_plot").glob("*.png")))
        fig.write_image(f'last_plot/last_plot{num_images+1}.png')
    else:
        fig.show()
    if save:
        save_plotly(fig, title_text)

def get_color(val : float):
    val = min(int(val * 5), 4)
    # Define the gradient characters from darkest to lightest
    gradient_chars = [" ", "░", "▒", "▓", "█"]
    return gradient_chars[val]

@dataclass
class VisualzeBoardArguments:
    include_attn_only = False
    include_mlp_only = False
    include_pre_resid = False
    include_layer_norm = False
    start_pos=0
    end_pos=59
    layers=8
    static_image=False#
    size_of_board = 225
    margin_t = 100
    mode = "linear"

def get_score_from_resid(resid, layer):
    # assert probe_name in ["linear", "flipped"]
    linear_probe = get_probe(layer, "linear", "post")
    flipped_probe = get_probe(layer, "flipped", "post")
    assert len(resid.shape) == 2
    seq_len, d_model = resid.shape
    logits = einops.einsum(resid, linear_probe, 'pos d_model, modes d_model rows cols options -> modes pos rows cols options')[0]
    probs = logits.softmax(dim=-1)
    flipped_logits = einops.einsum(resid, flipped_probe, 'pos d_model, modes d_model rows cols options -> modes pos rows cols options')[0]
    flipped_probs = flipped_logits.softmax(dim=-1)
    probs_copy = probs.clone()
    # Convert Back to Balck/White
    for i in range(0, seq_len, 2):
        probs[i, :, :, 1], probs[i, :, :, 2] = probs_copy[i, :, :, 2], probs_copy[i, :, :, 1]
    color_score = 0.5 + (probs[:, :, :, 2] - probs[:, :, :, 1])/2
    # Flip the color score on the rows dimension
    # TODO: Add Flips as Labels...
    color_score = color_score.flip(1)
    flip_score = flipped_probs[:, :, :, [0]].flip(1).squeeze(dim=-1)
    # flip_score = flipped_probs[:, :, :, [0]].squeeze(dim=-1)
    return color_score, flip_score

def get_boards(input_int : Float[Tensor, "pos"], vis_args : VisualzeBoardArguments, model: HookedTransformer):
    _, cache = model.run_with_cache(input_int)
    boards = []
    flip_boards = []
    for layer in range(vis_args.layers):
        color_scores = []
        flip_scores = []
        resid = cache["resid_post", layer][0].detach()
        color_score, flip_score = get_score_from_resid(resid, layer)
        color_scores += [color_score]
        flip_scores += [flip_score]
        if vis_args.include_pre_resid:
            resid = cache["resid_pre", layer][0].detach()
            color_score, flip_score = get_score_from_resid(resid, layer)
            color_scores += [color_score]
            flip_scores += [flip_score]
        if vis_args.include_attn_only:
            resid = cache["resid_post", layer][0].detach() - t.stack([cache["mlp_out", l][0].detach() for l in range(layer, layer + 1)]).sum(dim=0) - cache["resid_pre", layer][0].detach()
            color_score, flip_score = get_score_from_resid(resid, layer)
            color_scores += [color_score]
            flip_scores += [flip_score]
        if vis_args.include_mlp_only:
            resid = cache["resid_post", layer][0].detach() - t.stack([cache["attn_out", l][0].detach() for l in range(layer, layer + 1)]).sum(dim=0) - cache["resid_pre", layer][0].detach()
            color_score, flip_score = get_score_from_resid(resid, layer)
            color_scores += [color_score]
            flip_scores += [flip_score]
        if vis_args.include_layer_norm:
            resid = cache[f"blocks.{layer+1}.ln1.hook_normalized"][0].detach()
            color_score, flip_score = get_score_from_resid(resid, layer)
            color_scores += [color_score]
            flip_scores += [flip_score]
        if vis_args.include_layer_norm:
            resid = cache[f"blocks.{layer+1}.ln2.hook_normalized"][0].detach()
            color_score, flip_score = get_score_from_resid(resid, layer)
            color_scores += [color_score]
            flip_scores += [flip_score]
        color_score = t.stack(color_scores, dim=0)
        # color_score = color_score.transpose(0, 1)
        # color_score = color_score.reshape(-1, 8, 8)
        flip_score = t.stack(flip_scores, dim=0)
        # flip_score = flip_score.transpose(0, 1)
        # flip_score = flip_score.reshape(-1, 8, 8)
        # color_score, flip_score = get_score_from_resid(resid, layer)
        boards += [color_score]
        flip_boards += [flip_score]
    boards = t.stack(boards)
    flip_boards = t.stack(flip_boards)
    return boards, flip_boards

def plot_boards(label_list: List[str], boards : Float[Tensor, "layers mode pos rows cols"], flip_boards : Float[Tensor, "layers mode pos rows cols"], vis_args: VisualzeBoardArguments):
    # TODO: add attn/mlp only
    # TODO: Change Width and Height accordingly
    _, _, _, rows, cols = boards.shape
    print(boards.shape)
    seq_len = vis_args.end_pos - vis_args.start_pos
    modes = ["N"]
    if vis_args.include_pre_resid:
        modes += ["P"]
    if vis_args.include_attn_only:
        modes += ["A"]
    if vis_args.include_mlp_only:
        modes += ["M"]
    if vis_args.include_layer_norm:
        modes += ["L1"]
        modes += ["L2"]
    subplot_titles = [f"P: {i}, T: {label_list[i]}, L: {j}, M: {mode}" for i in range(vis_args.start_pos, vis_args.end_pos) for mode in modes for j in range(vis_args.layers)]
    # subplot_titles = [f"P: {i}, T: {label_list[i]}, L: {j}" for i in range(vis_args.start_pos, vis_args.end_pos) for j in range(vis_args.layers)]
    width = vis_args.layers * vis_args.size_of_board
    height = vis_args.margin_t + seq_len * len(modes) * vis_args.size_of_board
    vertical_spacing = 70 / height
    fig = make_subplots(rows=seq_len * len(modes), cols=vis_args.layers, subplot_titles=subplot_titles, vertical_spacing=vertical_spacing)
    for pos_idx, pos in enumerate(range(vis_args.start_pos, vis_args.end_pos)):
        for layer in range(vis_args.layers):
            for mode_idx, mode in enumerate(modes):
                text_data = [[get_color(flip_boards[layer, mode_idx, pos, i, j]) for j in range(cols)] for i in range(rows)]
                if vis_args.mode == "linear":
                    heatmap = go.Heatmap(
                        z=boards[layer, mode_idx, pos].cpu(),
                        text=text_data,
                        x=list(range(0, rows)),
                        y=reverse_alpha,
                        hoverongaps = False,
                        zmin=0.0,
                        zmax=1.0,
                        colorscale="RdBu",
                        texttemplate="%{text}",
                        # textfont_color="green",
                    )
                elif vis_args.mode == "flipped":
                    heatmap = go.Heatmap(
                        z=flip_boards[layer, mode_idx, pos].cpu(),
                        x=list(range(0, rows)),
                        y=reverse_alpha,
                        hoverongaps = False,
                        zmin=0.0,
                        zmax=1.0,
                        colorscale="Greens", # Green color scale
                    )
                else:
                    raise ValueError("Invalid Mode")
                fig.add_trace(
                    heatmap,
                    row=pos_idx * len(modes) + mode_idx + 1,
                    col=layer + 1
                )
    fig.layout.update(width=width, height=height, margin_t=vis_args.margin_t, title_text=f"Probe Results per Position per Layer, Mode: {modes[0]}") 
    if vis_args.static_image:
        # count the number of images in the last_plot directory
        num_images = len(list(Path("last_plot").glob("*.png")))
        fig.write_image(f'last_plot/last_plot{num_images+1}.png')
    else:
        fig.show()


def visualize_game(input_str, vis_args: VisualzeBoardArguments, model: HookedTransformer):
    # 1. Get the cache
    # 2. Get Board States from the cache using the Pobes
    # 3. Plot the Board States
    # assert not (vis_args.include_attn_only and vis_args.include_mlp_only)
    if len(input_str) > 59:
        input_str = input_str[:59]
    label_list = string_to_label(input_str)
    boards, flip_boards = get_boards(t.Tensor(to_int(input_str)).to(t.int32), vis_args, model)
    plot_boards(label_list, boards, flip_boards, vis_args)


def label_to_tuple(label):
    # return f"{alpha[label // 8]}{label % 8}" This but reverse
    alhpha_ind = alpha.find(label[0])
    return (alhpha_ind, int(label[1]))

def label_to_string(label):
    tup = label_to_tuple(label)
    return tup[0] * 8 + tup[1]

def label_to_int(label):
    st =  label_to_string(label)
    return str_to_int(st)



if __name__ == "__main__":
    vis_args = VisualzeBoardArguments()
    vis_args.start_pos = 0
    vis_args.end_pos = 20
    vis_args.layers = 6
    vis_args.include_attn_only = False
    vis_args.include_mlp_only = False
    vis_args.include_layer_norm = True
    vis_args.mode = "flipped"
    vis_args.static_image = True

    model = load_model("cuda")
    _, focus_games_str = get_focus_games()

    clean_input_str = focus_games_str[0][:30]
    visualize_game(clean_input_str, vis_args, model)
    '''
    print(label_to_int("B3"))
    print(label_to_string("B3"))
    print(label_to_tuple("B3"))'''


    