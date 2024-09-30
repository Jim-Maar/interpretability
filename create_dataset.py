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

# import abstract base class
from abc import ABC, abstractmethod
from functools import lru_cache

board_seqs_int = t.load(
    "data/board_seqs_int_train.pth"
)

def run_inference(games_int : Int[Tensor, "batch pos"], modules : list[str]) -> ActivationCache:
    with t.inference_mode():
        _, cache = model.run_with_cache(
            games_int[:, :-1].to(device),
            return_type=None,
            names_filter=lambda name: name in modules
            # names_filter=lambda name: name == f"blocks.{layer}.hook_resid_mid" or name == f"blocks.{layer}.mlp.hook_post"
            # names_filter=lambda name: name == f"blocks.{layer}.hook_resid_pre" or name == f"blocks.{layer}.mlp.hook_post"
        )
    return cache

def get_variables(resid_mid : Float[Tensor, "batch d_model"], layer, use_softmax = True, use_argmax=False, module:str="mid") -> Float[Tensor, "batch variables"]:
    probe_results_all = []
    # TODO: Add Accesible and Legal
    for probe_name in ["linear", "flipped", "placed", "accesible", "legal"]:
        probe = get_probe(layer, probe_name, module)[0]
        probe_result = einops.einsum(resid_mid, probe, "batch d_model, d_model rows cols options -> batch rows cols options")
        options = probe_result.shape[-1]
        if use_softmax:
            probe_result = probe_result.softmax(dim=-1)
        if use_argmax:
            probe_result = probe_result.argmax(dim=-1)
            # one hot encode the argmax
            probe_result = t.nn.functional.one_hot(probe_result, num_classes=options)
        probe_result = einops.rearrange(probe_result, "batch rows cols options -> batch (rows cols options)")
        probe_results_all.append(probe_result)
    probe_results_all = t.cat(probe_results_all, dim=-1)
    return probe_results_all

@lru_cache(maxsize=2)
def get_variable_names(remove_negative_features : bool = False) -> List[str]:
    variable_names = []
    # TODO: Add Accesible and Legal
    for probe_name in ["linear", "flipped", "placed", "accesible", "legal"]:
        for row in range(8):
            for col in range(8):
                for feature_name in probe_directions_list[probe_name]:
                    if feature_name[:3] == "not" and remove_negative_features:
                        continue
                    label = tuple_to_label((row, col))
                    variable_names.append(f"{label} {feature_name}")
    return variable_names

def extract_in_and_output(cache : ActivationCache, layer, neuron, pos_start = 0, pos_end = 60, use_softmax = True, use_argmax = False) -> Tuple[Float[Tensor, "batch variables"], Float[Tensor, "batch neurons"]]:
    resid_layer_norm = cache[f"blocks.{layer}.ln2.hook_normalized"][:, pos_start : pos_end]
    # resid_layer_norm = cache[f"blocks.{layer}.hook_resid_mid"][:, pos_start : pos_end]
    # resid_layer_norm = cache[f"blocks.{layer}.hook_resid_pre"][:, pos_start : pos_end]
    resid_layer_norm = einops.rearrange(resid_layer_norm, "batch pos d_model -> (batch pos) d_model")
    neuron_activations : Float[Tensor, "batch pos neurons"] = cache["mlp_post", layer][:, pos_start : pos_end]
    neuron_activations = einops.rearrange(neuron_activations, "batch pos neurons -> (batch pos) neurons")
    if neuron is not None:
        neuron_activations = neuron_activations[:, neuron]
    input_variables : Float[Tensor, "batch variables"] = get_variables(resid_layer_norm, layer, use_softmax, use_argmax)
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
    column_names = get_variable_names(False)
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


def get_big_dataset(layer, num_samples, start=0, use_softmax=True, overfitting_strength=None, inference_size=1000, pos_start=0, pos_end=60, use_argmax=False):
    """
    Creates a pandas DataFrame with the input variables and the neuron activations for a given layer and neuron.
    It has one column for each input variable and one column for the neuron activations.
    """
    neuron = None
    input_names = get_variable_names(False)
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
        input_variables, neuron_activations = extract_in_and_output(cache, layer, neuron, pos_start, pos_end, use_softmax, use_argmax)
        
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
    return df


def create_big_dataset(layer, num_samples, dataset_name, start=0, use_softmax=True, overfitting_strength=None, inference_size=1000, pos_start=0, pos_end=60, use_argmax=False):
    """
    Creates a pandas DataFrame with the input variables and the neuron activations for a given layer and neuron.
    It has one column for each input variable and one column for the neuron activations.
    """
    # Get the big dataset
    df = get_big_dataset(layer, num_samples, start, use_softmax, overfitting_strength, inference_size, pos_start, pos_end, use_argmax)
    # Save the dataframe to a CSV file
    df.to_csv(f"data/neuron_datasets/{dataset_name}_L{layer}.csv", index=False)



class variableExtractor(ABC):
    @abstractmethod
    def get_module_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_dataframe_from_cache(self, cache : ActivationCache) -> pd.DataFrame:
        pass

class featureExtractor(variableExtractor):
    def __init__(self, module : str, layer : int, use_softmax : bool = True, use_argmax : bool = True, in_or_out : str = None):
        self.use_softmax = use_softmax
        self.use_argmax = use_argmax
        if in_or_out is None:
            self.input_names = get_variable_names(False)
        else:
            self.input_names = [name + " " + in_or_out for name in get_variable_names(False)]
        self.module = module
        self.layer = layer
        if self.module == "ln1":
            self.full_module_name = f"blocks.{self.layer}.ln1.hook_normalized"
        elif self.module == "ln2":
            self.full_module_name = f"blocks.{self.layer}.ln2.hook_normalized"
        elif self.module == "mid":
            self.full_module_name = f"blocks.{self.layer}.hook_resid_mid"
        elif self.module == "post":
            self.full_module_name = f"blocks.{self.layer}.mlp.hook_post"

    def get_module_names(self) -> List[str]:
        return [self.full_module_name]

    def get_dataframe_from_cache(self, cache : ActivationCache) -> pd.DataFrame:
        assert self.full_module_name is not None
        resid_layer_norm = cache[self.full_module_name]
        resid_layer_norm = einops.rearrange(resid_layer_norm, "batch pos d_model -> (batch pos) d_model")
        if self.module == "mid" or self.module == "ln2":
            probe_module = "mid"
        else:
            probe_module = "post"
        input_variables : Float[Tensor, "batch variables"] = get_variables(resid_layer_norm, layer, self.use_softmax, self.use_argmax, probe_module)
        input_variables = input_variables.cpu().numpy()
        return pd.DataFrame(input_variables, columns=self.input_names)
    
class neuronActivationExtractor(variableExtractor):
    def __init__(self, layer, neurons = list(range(2048))):
        self.neurons = neurons
        self.output_names = get_neuron_names(layer, neurons)
        self.layer = layer
        self.full_module_name = f"blocks.{layer}.mlp.hook_post"
    
    def get_module_names(self) -> List[str]:
        return [self.full_module_name]

    def get_dataframe_from_cache(self, cache : ActivationCache) -> pd.DataFrame:
        neuron_activations : Float[Tensor, "batch pos neurons"] = cache["mlp_post", self.layer]
        neuron_activations = einops.rearrange(neuron_activations, "batch pos neurons -> (batch pos) neurons")
        neuron_activations = neuron_activations[:, self.neurons]
        neuron_activations = neuron_activations.cpu().numpy()
        return pd.DataFrame(neuron_activations, columns=self.output_names)

def create_big_dataset_2(layer, num_samples, dataset_name, variable_extractor_in : variableExtractor, variable_extractor_out : variableExtractor, start=0, inference_size=1000, pos_start=0, pos_end=60):
    batch = start
    dataframes = []

    while len(dataframes) == 0 or sum(len(df) for df in dataframes) < num_samples:
        real_inference_size = min(inference_size, num_samples)
        indices = t.arange(batch, min(batch + inference_size, start + num_samples))
        games_int = board_seqs_int[indices.cpu()]

        assert isinstance(games_int, Int[Tensor, f"batch={real_inference_size} full_game_len=60"])

        modules = variable_extractor_in.get_module_names() + variable_extractor_out.get_module_names()
        cache = run_inference(games_int, modules)

        input_variables = variable_extractor_in.get_dataframe_from_cache(cache)
        output_variables = variable_extractor_out.get_dataframe_from_cache(cache)

        data = pd.concat([input_variables, output_variables], axis=1)
        dataframes.append(data)
        batch += inference_size
    
    df = pd.concat(dataframes, ignore_index=True).iloc[:num_samples]
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(f"data/neuron_datasets/{dataset_name}_L{layer}.csv", index=False)
    return df
    

'''def create_big_dataset_from_cache(layer, num_samples, start = 0, use_softmax=True, pos_start=0, pos_end=60):
    df = get_big_dataset(layer, num_samples, start=start, use_softmax=use_softmax, inference_size=num_samples, pos_start=pos_start, pos_end=pos_end)
    return df'''


def get_filtered_dataset(big_dataset : pd.DataFrame, layer, neuron : int, size = 100000, overfitting_strength : int = None, remove_negative_features = True) -> pd.DataFrame:
    """
    Returns a new DataFrame with only the input variables and the neuron activations for the given neuron.
    The Big Dataset is filered such that the number of positive and negative samples is equal.
    Input:
    - big_dataset: The big dataset containing all the input variables and neuron activations
    - layer: The layer of the neuron
    - neuron: The neuron index
    - size: The number of samples to keep
    - overfitting_strength: The factor by which the number of negative samples is multiplied
    - remove_negative_features: Whether to remove the negative features e.g. (not_placed, not_flipped, not_accesible, not_legal)
    """
    input_names = get_variable_names(remove_negative_features)
    output_names = get_neuron_names(layer)
    column_names = input_names + ["neuron activation"]
    
    # Get the input variables and the neuron activations for the given neuron
    input_variables = big_dataset[input_names].to_numpy()
    neuron_activations = big_dataset[output_names[neuron]].to_numpy()

    cutoff = min(size, len(input_variables), len(neuron_activations))
    
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
        indeces = indeces[:cutoff]
    else:
        indeces = np.arange(cutoff)
    
    # Get the filtered input variables and neuron activations
    input_variables_filtered = input_variables[indeces]
    neuron_activations_filtered = neuron_activations[indeces]
    
    # Concatenate the filtered input variables and neuron activations along the second axis
    data = np.concatenate([input_variables_filtered, neuron_activations_filtered[:, None]], axis=1)
    
    # Create the filtered dataframe
    return pd.DataFrame(data, columns=column_names)


def save_filtered_dataset_for_neurons(big_dataset : pd.DataFrame, dataset_name :str, layer : int, neurons : list[int], num_samples : int = 100000):
    input_names = get_variable_names(False)
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
    # create_big_dataset(layer, 10000, "logic_testing", use_softmax=False)
    # create_big_dataset(layer, 10000, "logic_testing_no_softmax", use_softmax=False)
    # create_big_dataset(layer, 100000, "big_argmax_train", use_argmax=True)
    # create_big_dataset(layer, 50000, "big_argmax_eval", start=100000, use_argmax=True)
    # feature_extractor = featureExtractor()
    # neuron_extractor = neuronActivationExtractor(layer, [0, 1, 2])
    # create_big_dataset_2(layer, 2000, "new_dataset_argmax_test", feature_extractor, neuron_extractor, start=0)
    # create_big_dataset_2(layer, 1000000, "dataset_test_test_big", feature_extractor_in, feature_extractor_out, start=0)
    # create_big_dataset_2(layer, 1000000, "dataset_test_test_big", feature_extractor_in, feature_extractor_out, start=0)
    train_size = 100000
    test_size = 50000
    for layer in range(8):
        print(f"Layer {layer}")
        feature_extractor_in = featureExtractor(module = "ln2", layer = layer, use_softmax=False, use_argmax=True, in_or_out=None)
        feature_extractor_out = neuronActivationExtractor(layer = layer)
        for train_or_test in ["train", "test"]:
            if layer <= 1 and train_or_test == "train":
                continue
            print(f"Train or Test: {train_or_test}")
            print(f"Dataset Size: big")
            if train_or_test == "train":
                start = 0
            else:
                start = train_size
            if train_or_test == "train":
                size = train_size
            else:
                size = test_size
            big_dataset = create_big_dataset_2(layer, size, f"big_argmax_{train_or_test}", feature_extractor_in, feature_extractor_out, start=start)
            # Now convert the big dataset to a filtered dataset for a small and medium set of neurons
            for dataset_size in ["small", "medium"]:
                print(f"Dataset Size: {dataset_size}")
                if dataset_size == "small":
                    neurons = list(range(10))
                elif dataset_size == "medium":
                    neurons = list(range(100))
                save_filtered_dataset_for_neurons(big_dataset, f"{dataset_size}_argmax_{train_or_test}", layer, neurons, num_samples=size)

