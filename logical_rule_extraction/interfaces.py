"""
This File contains the interfaces for the different classes in the logical_rule_extraction package
"""
from torch import Tensor
from typing import List, Union, Optional, Tuple, Callable, Dict
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped

class EvaluaterInterface:
    def __init__(self) -> None:
        pass


# Trainer und Model sind sehr hart linked, deshalb weiß ich nicht, ob die Interfaces so sinnvoll sind
class ModelInterface:
    def __init__(self) -> None:
        pass

    def forward(self, variables : Float[Tensor, "batch neurons variables"]) -> Float[Tensor, "batch neurons"]:
        pass


class TrainerInterface:
    def __init__(self) -> None:
        pass

class ArgsInterface:
    def __init__(self, layer : int, run_number : int, run_name: str, single_neuron = None) -> None:
        self.undersampling_factor = 1
        self.debug = False
        self.single_neuron = single_neuron
        self.neurons_count = 2048 if single_neuron is None else 1
        self.variables_count = 64 * (3 + 2 + 2) # TODO: Train more Probes and update this (Legal, Almost Legal)
        self.num_games_train: int = 250
        self.layer = layer
        self.run_number = run_number
        self.set_run_name(run_name)
        
    def set_run_name(self, run_name: str):
        self.run_name = f"{run_name}_{self.run_number}"
        self.full_run_name = f"{run_name}_L{self.layer}_{self.run_number}"
        self.wandb_project = f"{run_name}"

class Experimenter:
    def __init__(self) -> None:
        pass