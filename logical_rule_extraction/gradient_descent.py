"""
This File contains the implementation for the Gradient Descent Technique for the logical rule extraction
"""
from interfaces import ModelInterface, TrainerInterface, ArgsInterface

from torch import Tensor
from typing import List, Union, Optional, Tuple, Callable, Dict
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped

import torch as t
import einops

device = t.device("cuda" if t.cuda.is_available() else "cpu")

EPSILON = 1e-6

debug = False

class GradientDescentArgs(ArgsInterface):
    def __init__(self, layer : int, run_number : int, run_name: str, single_neuron = None) -> None:
        super().__init__(layer = 1, run_number = 1, run_name = "GradientDescent")
        self.pos_start: int = 0
        self.pos_end: int = 59
        self.length: int = self.pos_end - self.pos_start
        self.max_epochs = 1
        self.single_neuron = single_neuron
        self.rules_count = 10
        self.neurons_count = 2048 if single_neuron is None else 1
        self.manual = False
        self.fuzzy_and = "mul"
        self.fuzzy_or = "max"
        self.spacity_factor = 10
        self.only_0_1_factor = 0
        self.initialization = "uniform"
        self.use_rule_weights = True
        self.manual_rules = None
        self.manual_weights = 10
        self.weight_decay_loss_func = "l1"
        self.learn_rules = False
        self.neuron_membership_funcs = None
        self.lr: float = 1e-2
        self.betas: Tuple[float, float] = (0.9, 0.99)
        self.batch_size: int = 2


class GradientDescentTrainer(TrainerInterface):
    pass


def get_centroid_individual(a, b, c, d, trapezoid_height):
    # This is the Center of Sums Method. Theres also the center of Gravity Method
    # b_new = a + trapezoid_height * (b-a)
    # c_new = d - trapezoid_height * (d-c)
    b_new = a + trapezoid_height * (b-a)
    c_new = d - trapezoid_height * (d-c)
    long_base = d - a
    short_base = c_new - b_new
    # individual_centroid = ((long_base + 2 * short_base) * trapezoid_height) / (3*(long_base + short_base)) + a
    # individual_centroid = a + ((b_new - a)**2 + 3*(c_new - b_new)**2 + (d - c_new)**2) / (3 * (- a - b_new + c_new + d))
    d_1 = b_new-a
    d_2 = c_new - b_new
    d_3 = d - c_new
    individual_centroid = ((a * d_1 / 2) + (d_1**2 / 6) + (b_new * d_2) + (d_2**2 / 2) + (d * d_3 / 2) - (d_3**2 / 6)) / ((d_1 / 2) + d_2 + (d_3 / 2))
    # individual_centroid = ((a + d_1/3) * (d_1 * trapezoid_height)/2 + (b_new + d_2/2) * (d_2 * trapezoid_height) + (d - d_3/3) * (d_3 * trapezoid_height)/2) / ((d_1 * trapezoid_height)/2 + d_2 * trapezoid_height + (d_3 * trapezoid_height)/2)
    individual_area = (long_base + short_base) / 2 * trapezoid_height
    return individual_centroid, individual_area


def get_centroid(trapezoid_metadata : List[Tuple[float, float, float, float]], trapezoid_heights : Float[Tensor, "batch neurons rules"], rule_trap_weights : Float[Tensor, "rules trap_params"]):
    # This is the Center of Sums Method. Theres also the center of Gravity Method
    # TODO: get rid of the for loop, vectorize this
    if rule_trap_weights is not None:
        batches, neurons, rules = trapezoid_heights.shape
    else:
        trapezoids, batches, neurons = trapezoid_heights.shape
    member_ship_funcs = []
    numerrator = t.zeros((batches, neurons)).to(device)
    denomenator = t.zeros((batches, neurons)).to(device)
    if rule_trap_weights is None:
        for i, (a, b, c, d) in enumerate(trapezoid_metadata):
            trapezoid_height = trapezoid_heights[i]
            trapezoid_height = t.max(trapezoid_height, t.tensor(EPSILON).to(device))
            # print(trapezoid_height.shape, trapezoid_height[:10], a, b, c, d)
            individual_centroid, individual_area = get_centroid_individsual(a, b, c, d, trapezoid_height)
            numerrator += individual_centroid * individual_area
            denomenator += individual_area
    else:
        for i in range(trapezoid_heights.shape[-1]):
            a = rule_trap_weights[i, 0]# .item()
            b = rule_trap_weights[i, 1]# .item()
            c = rule_trap_weights[i, 2]# .item()
            d = rule_trap_weights[i, 3]# .item()
            trapezoid_height = trapezoid_heights[:, :, i]
            trapezoid_height = t.max(trapezoid_height, t.tensor(EPSILON).to(device))
            # print(trapezoid_height.shape, trapezoid_height[:10], a, b, c, d)
            # This currently throws an error, it's really weird
            individual_centroid, individual_area = get_centroid_individual(a, b, c, d, trapezoid_height)
            numerrator += individual_centroid * individual_area
            denomenator += individual_area
    return numerrator / denomenator

# %%
import collections, gc, resource, torch
def debug_memory():
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter(
        (str(o.device), str(o.dtype), tuple(o.shape))
        for o in gc.get_objects()
        if torch.is_tensor(o)
    )
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

# %%
# TODO: First make funcitons narrow, then gerenerell
def variables_to_tensor(input_variables : Dict[str, Tuple]) -> Float[Tensor, "variables"]:
    out = t.Tensor((64 * (3 + 2 + 2)))
    i = 0
    for _, value in input_variables.items():
        for j in range(len(value)):
            out[i] = value[j]
            i += 1
    return out

def compute_fuzzy_out(fuzzy_weights, input_variables, neurons, rules):
    fuzzy_weights_r = einops.repeat(fuzzy_weights, "neurons rules variables -> neurons rules variables")
    input_variables_r = einops.repeat(input_variables, "variables -> neurons rules variables", neurons=neurons, rules=rules)
    updated_variables : Float[Tensor, "neuron rules variables"] = 1 - fuzzy_weights_r * (1 - input_variables_r)
    # fuzzy_mid = einops.reduce(updated_variables, "pos neurons rules variables -> pos neurons rules", "min")  # AND Gate
    # fuzzy_out = einops.reduce(fuzzy_mid, "pos neurons rules -> pos neurons", "max")  # OR Gate
    updated_variables = updated_variables.min(dim=-1).values  # AND Gate
    updated_variables = updated_variables.max(dim=-1).values  # OR Gate
    return updated_variables

def compute_fuzzy_mid(fuzzy_weights, input_variables, v, batch, neurons, rules):
    return 1 - einops.repeat(fuzzy_weights[:, :, v], "neurons rules -> batch neurons rules", batch=batch) * (1 - einops.repeat(input_variables[:, :, v], "batch -> batch neurons rules", neurons=neurons, rules=rules))

def compute_fuzzy_out_new(fuzzy_weights, input_variables):
    neurons, rules, variables = fuzzy_weights.shape
    batch, variables = input_variables.shape
    # fuzzy_weights_r = einops.repeat(fuzzy_weights, "neurons rules variables -> pos neurons rules variables")
    # input_variables_r = einops.repeat(input_variables, "batch pos variables -> pos neurons rules variables")
    fuzzy_mid = t.zeros((batch, neurons, rules)).to(device)
    for v in range(variables):
        if debug:
            print(f"Fuzzy Logic started for variable {v}")
            debug_memory()
        fuzzy_mid = t.min(fuzzy_mid, compute_fuzzy_mid(fuzzy_weights, input_variables, v, batch, neurons, rules))
    # fuzzy_mid = einops.reduce(updated_variables, "pos neurons rules variables -> pos neurons rules", "min")  # AND Gate
    # fuzzy_out = einops.reduce(fuzzy_mid, "pos neurons rules -> pos neurons", "max")  # OR Gate
    updated_variables = updated_variables.min(dim=-1).values  # AND Gate
    updated_variables = updated_variables.max(dim=-1).values  # OR Gate
    return updated_variables


def perform_fuzzy_logic(args : GradientDescentArgs, input_variables : Float[Tensor, "batch variables"], fuzzy_weights : Float[Tensor, "neuron rules variables"], rule_weights : Float[Tensor, "rules"]) -> Dict[str, Tuple]:
    batch, variables = input_variables.shape
    neurons, rules, _ = fuzzy_weights.shape
    if neurons == 1:
        fuzzy_weights_r = einops.repeat(fuzzy_weights, "neurons rules variables -> batch neurons rules variables", batch=batch)
        input_variables_r = einops.repeat(input_variables, "batch variables -> batch neurons rules variables", neurons=neurons, rules=rules)
        updated_variables : Float[Tensor, "batch neuron rules variables"] = 1 - fuzzy_weights_r * (1 - input_variables_r)
        if args.fuzzy_and == "mul":
            fuzzy_mid = updated_variables.prod(dim=-1) * rule_weights.sigmoid() # AND Gate
        else:
            fuzzy_mid = einops.reduce(updated_variables, "batch neurons rules variables -> batch neurons rules", "min")  # AND Gate
        if args.rules_count == 1:
            fuzzy_out = fuzzy_mid.squeeze(dim=-1)
        if args.learn_rules:
            fuzzy_mid_sum = 1 - fuzzy_mid.sum(dim=-1)
            rule_0 : Float[Tensor, "batch neurons"] = t.max(t.zeros_like(fuzzy_mid_sum).to(device), fuzzy_mid_sum).unsqueeze(dim=-1)
            fuzzy_out = t.cat([rule_0, fuzzy_mid], dim=-1)
            return fuzzy_out
        else:
            if args.fuzzy_or == "max":
                fuzzy_out = einops.reduce(fuzzy_mid, "batch neurons rules -> batch neurons", "max") # OR Gate
            elif args.fuzzy_or == "prob_sum":
                fuzzy_out = 1 - (1 - fuzzy_mid).prod(dim=-1) #OR Gate
        return fuzzy_out
    fuzzy_out_complete = []
    for b in range(batch):
        if debug:
            print(f"Fuzzy Logic started for batch {b}")
            debug_memory()
        fuzzy_out = compute_fuzzy_out(fuzzy_weights, input_variables[b], neurons, rules)
        fuzzy_out_complete.append(fuzzy_out)
    fuzzy_out_complete = t.stack(fuzzy_out_complete, dim=0)
    # fuzzy_out_complete = compute_fuzzy_out_new(fuzzy_weights, input_variables)
    return fuzzy_out_complete

def perform_defuzzification(neuron_membership_funcs : List[Tuple], neuron_positive : Float[Tensor, "batch neurons"], rule_trap_weights : Float[Tensor, "rules trap_params"] = None) -> Float[Tensor, "batch neurons"]:
    # I need the fuzzy membership funcitions for the neurons and some algorithm
    if debug:
        print("Defuzzification started")
    if rule_trap_weights is None:
        neuron_negative = 1 - neuron_positive
        # batch, neurons = neuron_positive.shape
        neuron_activation = get_centroid(neuron_membership_funcs, t.stack([neuron_negative, neuron_positive], dim=0), rule_trap_weights)
    else:
        neuron_activation = get_centroid(neuron_membership_funcs, neuron_positive, rule_trap_weights)
    if debug:
        print("Defuzzification ended")
    return neuron_activation

def full_fuzzy_inference(rule_weights : Float[Tensor, "rules"], neuron_membership_funcs : List[Tuple], args : GradientDescentArgs, input_variables : Float[Tensor, "batch variables"], fuzzy_weights : Float[Tensor, "neuron rules variables"], rule_trap_weights : Float[Tensor, "rules trap_params"]) -> Dict[str, float]: # Probe Results -> Neuron Activations
    """
    Perform fuzzy inference on the input variables.
    """
    # input_variables = input_variables_to_tensor(input_variables)
    # 1. Turn Fuzzy Variable into another Fuzzy Variable using the learned logical rulesneuron
    # The shape here will change with better neuron membership functions
    fuzzy_out : Float[Tensor, "batch neurons"] | Float[Tensor, "batch neurons rules"]= perform_fuzzy_logic(args, input_variables, fuzzy_weights, rule_weights)
    # 2. Turn the Fuzzy Variable into a crisp value using defuzzification
    neuron_activations : Float[Tensor, "batch neurons"] = perform_defuzzification(neuron_membership_funcs, fuzzy_out, rule_trap_weights)
    return neuron_activations


class GradientDescentModel(ModelInterface):
    def __init__(self, args : GradientDescentArgs) -> None:
        pass

    def forward(self, variables) -> float:
        pass
        