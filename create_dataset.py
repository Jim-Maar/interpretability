import os, sys
import torch as t
from torch import Tensor
import numpy as np
import einops
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import pandas as pd
from typing import List, Union, Optional, Tuple, Callable, Dict
from jaxtyping import Int, Float

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

'''cfg = HookedTransformerConfig(
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
model = HookedTransformer(cfg)'''

# from utils import board_seqs_int
from utils import get_probe
from utils import probe_directions_list
from utils import tuple_to_label
from utils import model

board_seqs_int = t.load(
    "data/board_seqs_int_train.pth"
)

def run_inference(games_int : Int[Tensor, "batch pos"], layer) -> ActivationCache:
    with t.inference_mode():
        _, cache = model.run_with_cache(
            games_int[:, :-1].to(device),
            return_type=None,
            names_filter=lambda name: name == f"blocks.{layer}.ln2.hook_normalized" or name == f"blocks.{layer}.mlp.hook_post"
            # names_filter=lambda name: name == f"blocks.{layer}.hook_resid_mid" or name == f"blocks.{layer}.mlp.hook_post"
            # names_filter=lambda name: name == f"blocks.{layer}.hook_resid_pre" or name == f"blocks.{layer}.mlp.hook_post"
        )
    return cache

def get_variables(resid_mid : Float[Tensor, "batch d_model"], layer, use_softmax = True) -> Float[Tensor, "batch variables"]:
    probe_results_all = []
    for probe_name in ["linear", "flipped", "placed"]:
        probe = get_probe(layer, probe_name, "mid")[0]
        probe_result = einops.einsum(resid_mid, probe, "batch d_model, d_model rows cols options -> batch rows cols options")
        if use_softmax:
            probe_result = probe_result.softmax(dim=-1)
        probe_result = einops.rearrange(probe_result, "batch rows cols options -> batch (rows cols options)")
        probe_results_all.append(probe_result)
    probe_results_all = t.cat(probe_results_all, dim=-1)
    return probe_results_all

def get_variable_names():
    variable_names = []
    for probe_name in ["linear", "flipped", "placed"]:
        for row in range(8):
            for col in range(8):
                for option in probe_directions_list[probe_name]:
                    label = tuple_to_label((row, col))
                    variable_names.append(f"{label} {option}")
    return variable_names

def extract_in_and_output(cache : ActivationCache, layer, neuron, pos_start = 0, pos_end = 60, use_softmax = True) -> Tuple[Float[Tensor, "batch variables"], Float[Tensor, "batch neurons"]]:
    resid_layer_norm = cache[f"blocks.{layer}.ln2.hook_normalized"][:, pos_start : pos_end]
    # resid_layer_norm = cache[f"blocks.{layer}.hook_resid_mid"][:, pos_start : pos_end]
    # resid_layer_norm = cache[f"blocks.{layer}.hook_resid_pre"][:, pos_start : pos_end]
    resid_layer_norm = einops.rearrange(resid_layer_norm, "batch pos d_model -> (batch pos) d_model")
    neuron_activations : Float[Tensor, "batch pos neurons"] = cache["mlp_post", layer][:, pos_start : pos_end]
    neuron_activations = einops.rearrange(neuron_activations, "batch pos neurons -> (batch pos) neurons")
    if neuron is not None:
        neuron_activations = neuron_activations[:, neuron]
    input_variables : Float[Tensor, "batch variables"] = get_variables(resid_layer_norm, layer, use_softmax)
    return input_variables, neuron_activations

# Rework everything in here than copy it to .py file
# TODO: Make this function swappable
# TODO: This function currently only works for a single neuron
def overfit(input_variables : Float[Tensor, "batch variables"], neuron_activations_single : Float[Tensor, "batch"], overfitting_strength = 1) -> Tuple[Float[Tensor, "batch variables"], Float[Tensor, "batch"]]:
    with t.inference_mode():
        # TODO: Make undersampling work for generell case
        # Make the hole thing generel so that it works for all kinds of features not just neuron
        # TODO: Do the Bottom Part for each neuron
        # Calculate indeces
        positive_indeces = t.nonzero(neuron_activations_single > 0).squeeze()
        count_positive = len(positive_indeces)
        negative_indeces = t.nonzero(neuron_activations_single <= 0).squeeze()[:count_positive * overfitting_strength]
        indeces = t.cat([positive_indeces, negative_indeces])
        # Do the rest
        neuron_activations_single = neuron_activations_single[indeces]
        input_variables_single = input_variables[indeces]
        return input_variables_single, neuron_activations_single


# TODO: Add start for evaluation
# TODO: Cut the data to the right size
# TODO: randomize at the end
# TODO: Change import to be non expensive
import pandas as pd
import numpy as np
import torch as t

np.random.seed(42)

def create_dataset(layer, neuron, num_samples, dataset_name, start=0, use_softmax=True, overfitting_strength=1, inference_size=1000, pos_start=0, pos_end=60):
    """
    Creates a pandas DataFrame with the input variables and the neuron activations for a given layer and neuron.
    It has one column for each input variable and one column for the neuron activations.
    """
    column_names = get_variable_names()
    column_names.append("neuron_activation")
    
    # Initialize an empty list to collect dataframes for each batch
    dataframes = []
    
    batch = start
    while len(dataframes) == 0 or sum(len(df) for df in dataframes) < num_samples:
        indices = t.arange(batch, min(batch + inference_size, num_samples))
        games_int = board_seqs_int[indices.cpu()]

        assert isinstance(games_int, Int[Tensor, f"batch={inference_size} full_game_len=60"])
        
        cache = run_inference(games_int, layer)
        input_variables, neuron_activations = extract_in_and_output(cache, layer, neuron, pos_start, pos_end, use_softmax)
        input_variables, neuron_activations = overfit(input_variables, neuron_activations, overfitting_strength)
        
        input_variables = input_variables.cpu().numpy()
        neuron_activations = neuron_activations.cpu().numpy()
        
        # Concatenate input variables and neuron activations along the second axis
        data = np.concatenate([input_variables, neuron_activations[:, None]], axis=1)
        
        # Append the resulting dataframe to the list
        dataframes.append(pd.DataFrame(data, columns=column_names))
        
        batch += inference_size

    # Concatenate all the dataframes into one and trim it to the correct size
    df = pd.concat(dataframes, ignore_index=True).iloc[:num_samples]
    
    # Shuffle the rows of the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save the dataframe to a CSV file
    df.to_csv(f"data/neuron_datasets/{dataset_name}_L{layer}_N{neuron}.csv", index=False)


def get_neuron_names(layer, neurons = list(range(2048))):
    return [f"L{layer}_N{i}" for i in neurons]


def create_big_dataset(layer, num_samples, dataset_name, start=0, use_softmax=True, overfitting_strength=1, inference_size=1000, pos_start=0, pos_end=60):
    """
    Creates a pandas DataFrame with the input variables and the neuron activations for a given layer and neuron.
    It has one column for each input variable and one column for the neuron activations.
    """
    neuron = None
    input_names = get_variable_names()
    output_names = get_neuron_names(layer)
    column_names = input_names + output_names
    
    # Initialize an empty list to collect dataframes for each batch
    dataframes = []
    
    batch = start
    while len(dataframes) == 0 or sum(len(df) for df in dataframes) < num_samples:
        indices = t.arange(batch, min(batch + inference_size, start + num_samples))
        games_int = board_seqs_int[indices.cpu()]

        assert isinstance(games_int, Int[Tensor, f"batch={inference_size} full_game_len=60"])
        
        cache = run_inference(games_int, layer)
        # Float[Tensor, "batch variables"], Float[Tensor, "batch neurons"]
        input_variables, neuron_activations = extract_in_and_output(cache, layer, neuron, pos_start, pos_end, use_softmax)
        
        input_variables = input_variables.cpu().numpy()
        neuron_activations = neuron_activations.cpu().numpy()
        
        # Concatenate input variables and neuron activations along the second axis
        data = np.concatenate([input_variables, neuron_activations], axis=1)
        
        # Append the resulting dataframe to the list
        dataframes.append(pd.DataFrame(data, columns=column_names))
        
        batch += inference_size

    # Concatenate all the dataframes into one and trim it to the correct size
    df = pd.concat(dataframes, ignore_index=True).iloc[:num_samples]
    
    # Shuffle the rows of the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save the dataframe to a CSV file
    df.to_csv(f"data/neuron_datasets/{dataset_name}_L{layer}.csv", index=False)


def get_filtered_dataset(big_dataset : pd.DataFrame, layer, neuron : int, size = 5000, overfitting_strength : int = 1) -> pd.DataFrame:
    """
    Returns a new DataFrame with only the input variables and the neuron activations for the given neuron.
    The Big Dataset is filered such that the number of positive and negative samples is equal.
    """
    input_names = get_variable_names()
    output_names = get_neuron_names(layer)
    column_names = input_names + ["neuron activation"]
    
    # Get the input variables and the neuron activations for the given neuron
    input_variables = big_dataset[input_names].to_numpy()
    neuron_activations = big_dataset[output_names[neuron]].to_numpy()
    
    if overfitting_strength is not None:
        # Calculate the number of positive samples
        positive_samples = np.sum(neuron_activations > 0)
        
        # Calculate the number of negative samples
        negative_samples = np.sum(neuron_activations <= 0)
        
        # Calculate the number of samples to keep
        num_samples = min(positive_samples, negative_samples) * overfitting_strength
        
        # Get the indeces of the positive samples
        positive_indeces = np.where(neuron_activations > 0)[0]
        
        # Get the indeces of the negative samples
        negative_indeces = np.where(neuron_activations <= 0)[0][:num_samples]
        
        # Concatenate the indeces
        indeces = np.concatenate([positive_indeces, negative_indeces])
        
        # shuffle the indeces
        np.random.shuffle(indeces)

        # Cut the indeces to the correct size
        indeces = indeces[:size]
    else:
        indeces = np.arange(size)
    
    # Get the filtered input variables and neuron activations
    input_variables_filtered = input_variables[indeces]
    neuron_activations_filtered = neuron_activations[indeces]
    
    # Concatenate the filtered input variables and neuron activations along the second axis
    data = np.concatenate([input_variables_filtered, neuron_activations_filtered[:, None]], axis=1)
    
    # Create the filtered dataframe
    return pd.DataFrame(data, columns=column_names)


def save_filtered_dataset_for_neurons(big_dataset : pd.DataFrame, dataset_name :str, layer : int, neurons : list[int], num_samples : int = 100000):
    input_names = get_variable_names()
    output_names = get_neuron_names(layer, neurons)
    column_names = input_names + output_names
    big_dataset_filtered = big_dataset[column_names]
    # save big_dataset_filtered to csv
    big_dataset_filtered.to_csv(f"data/neuron_datasets/{dataset_name}_L{layer}.csv", index=False)


if __name__ == "__main__":
    layer = 1
    # neuron = 421# 
    # create_dataset(layer, neuron, 10000, "logic")
    # print("Done")
    create_big_dataset(layer, 10000, "logic_testing", use_softmax=False)
    create_big_dataset(layer, 10000, "logic_testing_no_softmax", use_softmax=False)