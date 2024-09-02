# %%
from interpreting_neurons_utils import *
import wandb
import pickle

# %% [markdown]
# - Look at python fuzzy logic libraries
# - Experiment 1
#     - Make the general setup, works for
#         - different kinds of membership functions of output neurons
#         - different kind of cost functions
#         - different kind of input (Input are the fuzzy variables)
#     - Define single fuzzy membership functions for all neurons
#         - Look at histogram and do shit
#         - Just do Negative, Positive
#     - Use softmax for fuzzy input variables
#     - use L1 sparcity constraint, extra for > 1
# - ⇒
#     - How good is MSE of neuron activations
#     - Histogram of the weights
#     - How realistic are the rules learned (are they the same that I would come up with)

# %% [markdown]
# - I think I need to do all the Fuzzy Logic shit by hand ...
# - I think it makes sense to initialize all the fuzzy weights near 0.5

# %% [markdown]
# - First: not general?!
# - Training Loop
#     - Residual Stream
#     - Proberesults (Softmax)
#     - Fuzzy Inference -> Neuron Activation Pred (This is the step where things are learned)
#     - Loss
#     - Update

# %%
debug = False
EPSILON = 1e-6

# %%
"""
args.num_games_train = 500
    args.fuzzy_and = "mul"
    args.initialization = "uniform"
    args.batch_size = 1
    args.undersampling_factor = 1
    args.sparcity_factor = 10 # 0.2
    args.only_0_1_factor = 0 # 0.1
    args.lr = 1e-1
    args.rules_count = 2
    args.manual = True
    args.debug = True
"""

@dataclass
class FuzzyTrainingArgs():
    # Which layer, and which positions in a game sequence to probe
    def __init__(self, run_number : int, layer : int, single_neuron=None):
        self.undersampling = True
        self.undersampling_factor = 1
        self.debug = False
        self.run_number = run_number
        self.layer: int = layer
        self.pos_start: int = 0
        self.pos_end: int = 59
        self.length: int = self.pos_end - self.pos_start
        self.max_epochs = 1
        self.single_neuron = single_neuron
        self.rules_count = 10
        self.neurons_count = 2048 if single_neuron is None else 1
        self.variables_count = 64 * (3 + 2 + 2)
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

        self.num_games_train: int = 250
        self.batch_size: int = 2

        self.lr: float = 1e-2
        self.betas: Tuple[float, float] = (0.9, 0.99)
        self.wd: float = 0.1

        self.run_name: str = f"Fuzzy_{self.run_number}"
        self.full_run_name: str = f"Fuzzy_L{self.layer}_{self.run_number}"

        # wnadb
        self.wandb_project: str = f"Train Fuzzy Logic Gates Test"

# %%
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
t.max(t.zeros_like(t.randn((4, 5))), t.randn((4, 5))).unsqueeze(0).shape

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

def tensor_to_variables(input_tensor : Float[Tensor, "variables"]) -> Dict[str, Tuple]:
    pass

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


def perform_fuzzy_logic(args : FuzzyTrainingArgs, input_variables : Float[Tensor, "batch variables"], fuzzy_weights : Float[Tensor, "neuron rules variables"], rule_weights : Float[Tensor, "rules"]) -> Dict[str, Tuple]:
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

def full_fuzzy_inference(rule_weights : Float[Tensor, "rules"], neuron_membership_funcs : List[Tuple], args : FuzzyTrainingArgs, input_variables : Float[Tensor, "batch variables"], fuzzy_weights : Float[Tensor, "neuron rules variables"], rule_trap_weights : Float[Tensor, "rules trap_params"]) -> Dict[str, float]: # Probe Results -> Neuron Activations
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

def get_variables(resid_mid : Float[Tensor, "batch d_model"], layer) -> Float[Tensor, "batch variables"]:
    probe_results_all = []
    for probe_name in ["linear", "flipped", "placed"]:
        probe = get_probe(layer, probe_name, "mid")[0]
        probe_result = einops.einsum(resid_mid, probe, "batch d_model, d_model rows cols options -> batch rows cols options").softmax(dim=-1)
        probe_result = einops.rearrange(probe_result, "batch rows cols options -> batch (rows cols options)")
        probe_results_all.append(probe_result)
    probe_results_all = t.cat(probe_results_all, dim=-1)
    return probe_results_all

def get_manual_weights(args : FuzzyTrainingArgs, rules=[(0, "A4", "placed", "placed")]) -> Float[Tensor, "neurons rules variables"]:
    fuzzy_weights_all = []
    for probe_name, how_many in [("linear", 3), ("flipped", 2), ("placed", 2)]:
        fuzzy_weights = t.ones(size=(args.neurons_count, args.rules_count, 8, 8, how_many), requires_grad=False).to(device) * -args.manual_weights
        for rule, label, probe_name2, direction_str in rules:
            if probe_name != probe_name2:
                continue
            fuzzy_weights[0, rule, *label_to_tuple(label), get_direction_int(direction_str)] = args.manual_weights
        fuzzy_weights = einops.rearrange(fuzzy_weights, "neurons rules rows cols options -> neurons rules (rows cols options)")
        fuzzy_weights_all += [fuzzy_weights]
    fuzzy_weights_all = t.cat(fuzzy_weights_all, dim=-1)
    return fuzzy_weights_all


def init_weights(args : FuzzyTrainingArgs) -> Float[Tensor, "neurons rules variables"]:
    if args.manual:
        fuzzy_weights = get_manual_weights(args, args.manual_rules)
        fuzzy_weights.requires_grad = True
        return fuzzy_weights
    if args.initialization == "ones":
        fuzzy_weights = t.ones(size=(args.neurons_count, args.rules_count, args.variables_count), requires_grad=False).to(device)
    elif args.initialization == "zeros":
        fuzzy_weights = t.ones(size=(args.neurons_count, args.rules_count, args.variables_count), requires_grad=False).to(device) * -1
    elif args.initialization == "uniform":
        fuzzy_weights = t.randn(size=(args.neurons_count, args.rules_count, args.variables_count), requires_grad=False).to(device)
        # fuzzy_weights = fuzzy_weights.uniform_(-1, 1) + EPSILON
    else:
        raise NotImplementedError
    fuzzy_weights.requires_grad = True
    return fuzzy_weights

# Try one inference
"""variables = get_variables(focus_cache[module])
variables = variables[:2, :]
weights = init_weights()
neuron_activations = full_fuzzy_inference(variables, weights)
print(neuron_activations[0, 0, neuron])"""

# %%
# TODO: Make undersampling good so that the batchsize is constant ...

# %%
def get_rules_dict(weights : Float[Tensor, "neurons rules variables"], neuron : int = 0, thresh : float = 0.1):
    # TODO: Why did I put all these dimensions together in the first place? I guess it is easier to understand this way
    weights = t.sigmoid(weights)
    variables_expanded = {
        "linear" : einops.rearrange(weights[:, :, :64*3], "neurons rules (rows cols options) -> neurons rules rows cols options", rows=8, cols=8, options=3)[neuron],
        "flipped" : einops.rearrange(weights[:, :, 64*3:64*5], "neurons rules (rows cols options) -> neurons rules rows cols options", rows=8, cols=8, options=2)[neuron],
        "placed" : einops.rearrange(weights[:, :, 64*5:], "neurons rules (rows cols options) -> neurons rules rows cols options", rows=8, cols=8, options=2)[neuron],
    }
    out_dict = {}
    for probe_name, variables in variables_expanded.items():
        rules, rows, cols, options = variables.shape
        for rule in range(rules):
            for row in range(rows):
                for col in range(cols):
                    for option in range(options):
                        weight = variables[rule, row, col, option]
                        if weight > thresh or thresh == -1:
                            label = tuple_to_label((row, col))
                            option_str = get_direction_str(probe_name, option)
                            out_dict[(rule,label,probe_name, option_str)] = weight.item()
                            # print(f"Rule {rule}, {tuple_to_label((row, col))} is {probe_name} {option}: {weight}")
    return out_dict

def print_rules_dict(rules_dict : Dict[Tuple[int, str, str, str], float], rules_count=10):
    # Sort the rules by the rule number
    rules_dict = dict(sorted(rules_dict.items(), key=lambda item: item[0]))
    # iterate over the rule_number, get all entries for each rule_number, and print them
    for rule_number in range(rules_count):
        print(f"Rule {rule_number}")
        for key, value in rules_dict.items():
            if key[0] == rule_number:
                label = key[1]
                probe_name = key[2]
                option = key[3]
                print(f"{label} {option}: {value:.2f}")
        print("")

# %%
def get_neuron_membership_funcs_single(layer, neuron):
    if layer == 0 and neuron == 485: 
        neuron_membership_funcs = [(-0.26, -0.26, 0.04, 0.04), (2.7, 3.2, 3.2, 3.7)]
    elif layer == 1 and neuron == 421:
        # neuron_membership_funcs = [(-0.1725, -0.1725, 0, 0), (0.5, 0.5, 1.6, 1.6), (1.7, 1.7, 2.8, 2.8)]
        # neuron_membership_funcs = [(-0.1725, -0.1725, 0, 0), (0.5, 0.5, 2.8, 2.8)]
        neuron_membership_funcs = [(-0.8, -0.8, 0.5, 0.5), (1.6, 1.6, 3, 3)]
    else:
        neuron_membership_funcs = [(-0.26, -0.26, 0.04, 0.04), (2.7, 3.2, 3.2, 3.7)] # idk
    return neuron_membership_funcs


def get_neuron_membership_funcs(layer, all_neurons=True, neuron=None):
    neuron_membership_funcs_all = []
    if not all_neurons:
        neuron_membership_funcs = get_neuron_membership_funcs_single(layer, neuron)
        neuron_membership_funcs_all.append(neuron_membership_funcs)
        return neuron_membership_funcs_all
    for neuron in range(2048):
        neuron_membership_funcs = get_neuron_membership_funcs_single(layer, neuron)
        neuron_membership_funcs_all.append(neuron_membership_funcs)
    return neuron_membership_funcs_all

# %%
def get_rule_weights(args : FuzzyTrainingArgs):
    if args.initialization == "ones":
        rule_weights = t.ones(size=(args.rules_count, ), requires_grad=False).to(device)
    elif args.initialization == "zeros":
        rule_weights = t.ones(size=(args.rules_count, ), requires_grad=False).to(device) * -1
    elif args.initialization == "uniform":
        rule_weights = t.randn(size=(args.rules_count, ), requires_grad=False).to(device)
        # fuzzy_weights = fuzzy_weights.uniform_(-1, 1) + EPSILON
    else:
        raise NotImplementedError
    rule_weights.requires_grad = True
    return rule_weights
    

# %%
def get_rule_trap_weights(args : FuzzyTrainingArgs):
    assert args.single_neuron is not None
    activations : Float[Tensor, "batch pos d_mlp"] = focus_cache["mlp_post", args.layer][:, :, args.single_neuron].unsqueeze(2)
    activations = einops.rearrange(activations, "batch pos d_mlp -> (batch pos d_mlp)")
    negative_mean = activations[activations <= 0].mean()
    negative_std = activations[activations <= 0].std()
    positive_mean = activations[activations > 0].mean()
    positive_std = activations[activations > 0].std()
    rule_trap_weights = t.zeros((args.rules_count + 1, 4)).to(device)
    rule_trap_weights[0] = t.tensor([negative_mean - negative_std, negative_mean, negative_mean, negative_mean + negative_std]).to(device)
    rule_trap_weights[1:] = t.tensor([positive_mean - positive_std, positive_mean, positive_mean, positive_mean + positive_std]).to(device)
    rule_trap_weights.requires_grad = True
    return rule_trap_weights

# %%
import time

# %%
EPSILON = 1e-6
class FuzzyTrainer:
    def __init__(self, model: HookedTransformer, args: FuzzyTrainingArgs, train_base_resid):
        self.model = model
        self.args = args
        self.probe_names = {
            "linear" : ["empty", "yours", "mine"],
            "flipped" : ["flipped", "not_flipped"],
            "placed" : ["placed", "not_placed"],
        }
        self.weights : Float[Tensor, "neurons rules variables"] = init_weights(self.args)
        self.rule_weights = get_rule_weights(self.args)
        if self.args.learn_rules:
            self.rule_trap_weights = get_rule_trap_weights(self.args)
        else:
            self.rule_trap_weights = None
        if self.args.neuron_membership_funcs is None:
            self.neuron_membership_funcs = get_neuron_membership_funcs(self.args.layer, all_neurons= not self.args.single_neuron, neuron=self.args.single_neuron)
            if self.args.single_neuron is not None:
                self.neuron_membership_funcs = self.neuron_membership_funcs[0]
            else:
                # TODO: Make this work for all neurons (the Centroid calculation would not work ...)
                raise NotImplementedError
        else:
            self.neuron_membership_funcs = self.args.neuron_membership_funcs

    def training_step(self, indices: Int[Tensor, "game_idx"], train_or_val="train") -> t.Tensor:
        # Get the game sequences and convert them to state stacks
        # time_all_start = time.time()
        games_int = board_seqs_int[indices.cpu()]
        games_str = board_seqs_string[indices.cpu()]
        batch_size = self.args.batch_size
        game_len = self.args.length
        # options = self.args.options
        d_model = 512

        # games_int = tensor of game sequences, each of length 60
        # This is the input to our model
        assert isinstance(games_int, Int[Tensor, f"batch={batch_size} full_game_len=60"])

        # SOLUTION
        # time_inference_start = time.time()
        with t.inference_mode():
            _, cache = model.run_with_cache(
                games_int[:, :-1].to(device),
                return_type=None,
                names_filter=lambda name: name == f"blocks.{self.args.layer}.ln2.hook_normalized" or name == f"blocks.{self.args.layer}.mlp.hook_post"
            )
            # TODO: Make undersampling work for generell case
            resid_layer_norm = cache[f"blocks.{self.args.layer}.ln2.hook_normalized"][:, self.args.pos_start: self.args.pos_end]
            neuron_activations : Float[Tensor, "batch pos neurons"] = cache["mlp_post", self.args.layer][:, self.args.pos_start: self.args.pos_end]
            resid_layer_norm = einops.rearrange(resid_layer_norm, "batch pos d_model -> (batch pos) d_model")
            neuron_activations = einops.rearrange(neuron_activations, "batch pos neurons -> (batch pos) neurons")
            # time_inference_end = time.time()
            if self.args.single_neuron is not None:
                neuron_activations = neuron_activations[:, self.args.single_neuron]
                # neuron_activations = einops.rearrange(neuron_activations, "batch pos neurons -> (batch pos neurons)")
                if self.args.undersampling:
                    # count_positive = (neuron_activations > 0).sum()
                    # count_all = len(neuron_activations)
                    # Wait das ist übelst dumm die zu sortieren
                    # neuron_activations_ideces = neuron_activations.argsort(descending=True)
                    # positive_indeces = neuron_activations_ideces[:count_positive]
                    # positive_idences = t.where(neuron_activations > 0)[0]
                    # negative_indeces = neuron_activations_ideces[t.randperm(count_all - count_positive).to(device)[:int(count_positive * self.args.undersampling_factor)] + count_positive]
                    positive_indeces = t.nonzero(neuron_activations > 0).squeeze()
                    count_positive = len(positive_indeces)
                    negative_indeces = t.nonzero(neuron_activations <= 0).squeeze()[:count_positive]
                    indeces = t.cat([positive_indeces, negative_indeces])
                    neuron_activations = neuron_activations[indeces]
                    resid_layer_norm = resid_layer_norm[indeces]
            # resid = cache["resid_mid", layer][:, self.args.pos_start: self.args.pos_end]
        resid_layer_norm = resid_layer_norm.clone().detach().to(device)
        variables : Float[Tensor, "batch variables"] = get_variables(resid_layer_norm, self.args.layer)
        sigmoid_weights = t.sigmoid(self.weights)
        neuron_activations_pred : Float[Tensor, "batch neurons"] = full_fuzzy_inference(self.rule_weights, self.neuron_membership_funcs, self.args, variables, sigmoid_weights, self.rule_trap_weights)
        if self.args.single_neuron is not None:
            neuron_activations_pred = neuron_activations_pred[:, 0]

        acc = ((neuron_activations > 0) == (neuron_activations_pred > 0)).sum() / len(neuron_activations)

        mse : Float[Tensor, "batch neurons"] = (neuron_activations - neuron_activations_pred)**2
        mse_loss = mse.mean()

        # TODO: Create a fancy loss function that everything at ones (Forces the weights to be either 0 or 1, but allowing transitions between these two)
        # This is currently a placeholder
        if self.args.weight_decay_loss_func == "l1":
            sparcity_loss = sigmoid_weights.mean()
        elif self.args.weight_decay_loss_func == "sqrt":
            sparcity_loss = sigmoid_weights.sqrt().mean()
        else:
            raise NotImplementedError
        spacity_loss2 = (sigmoid_weights.clamp(max=0).abs() + (sigmoid_weights.clamp(min=0, max=0.5) - 0).abs() + ((sigmoid_weights - 1).clamp(min=-0.5, max=0)).abs() + ((sigmoid_weights - 1).clamp(min=0)).abs()).mean()
        loss = mse_loss + self.args.sparcity_factor * sparcity_loss + self.args.only_0_1_factor * spacity_loss2
        
        # time_all_end = time.time()

        # print(time_all_end - time_all_start, time_inference_end - time_inference_start, (time_inference_end - time_inference_start) / (time_all_end - time_all_start))

        if train_or_val == "train" and self.step % 10 == 0 and not self.args.debug:
            wandb.log({f"{train_or_val}_loss": loss.item()}, step=self.step)
        if (self.step % 50 == 0 and self.args.debug):
            rules_dict = get_rules_dict(self.weights, thresh=0.2)
            print(self.step)
            print(rules_dict)
            print_rules_dict(rules_dict, self.args.rules_count)
            print("")
            print(f"Accuracy: {acc.item()}")
            print(((neuron_activations > 0) == (neuron_activations_pred > 0))[:20])
            print(self.rule_weights)
            print(self.rule_trap_weights)
        self.step += 1

        return loss
        
    def save_weights(self, object, path):
        folder_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(path, "wb") as f:
            pickle.dump(object, f)

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

    def train(self):
        print(f"Training Probe: {self.args.full_run_name}")
        self.step = 0
        if not self.args.debug:
            wandb.login()
            wandb.init(project=self.args.wandb_project, name=self.args.full_run_name, config=self.args)
        parameters = [self.weights]
        if self.args.use_rule_weights:
            parameters.append(self.rule_weights)
        else:
            self.rule_weights = t.ones(size=(self.args.rules_count, ), requires_grad=False).to(device) * 20
        # if self.args.learn_rules:
        #     parameters.append(self.rule_trap_weights)

        if self.args.learn_rules:
            optimizer = t.optim.AdamW(parameters, lr=self.args.lr, betas=self.args.betas)
            # optimizer = t.optim.AdamW([{'params': parameters},{'params': self.rule_trap_weights, 'lr' : self.args.lr * 0.01}], lr=self.args.lr, betas=self.args.betas)
        else:
            optimizer = t.optim.AdamW(parameters, lr=self.args.lr, betas=self.args.betas)

        for epoch in range(self.args.max_epochs):
            full_train_indices = self.shuffle_training_indices()
            progress_bar_train = tqdm(full_train_indices)
            for indices in progress_bar_train:
                loss = self.training_step(indices)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress_bar_train.set_description(f"Train_Loss = {loss:.4f}")

        if not self.args.debug:
            wandb.finish()
            self.save_weights(self.weights, f"fuzzy_weights/_{self.args.full_run_name}.pth")
        print("Probe Trained and Saved")

# %%
# Example tensor a
a = torch.tensor([1.0, -2.0, 0.0, 3.0, -4.0, 5.0])

# Get indices of elements in a that are greater than 0
indices = torch.nonzero(a > 0).squeeze()

# If you need a 1D vector of indices
# indices = indices.flatten()

print(indices)

# %%
# set torch to random seed and print the seed
# It's the same every time because of the seed (What happens at 6

t.manual_seed(42)
run_number = 0
layer = 1 # 0
neuron = 421 # 485 # 368
manual = False
# layer = 0
# neuron = 485

if manual:
    args = FuzzyTrainingArgs(run_number, layer=layer, single_neuron=neuron)
    args.num_games_train = 1000
    args.fuzzy_and = "mul"
    args.initialization = "uniform"
    args.weight_decay_loss_func = "sqrt"
    args.fuzzy_or = "prob_sum"
    args.batch_size = 1000
    args.undersampling_factor = 1
    args.sparcity_factor = 10 # 0.2
    args.only_0_1_factor = 0 # 0.1
    args.lr = 1e-1
    args.rules_count = 2
    args.manual = True
    args.debug = True
    args.use_rule_weights = False
    args.manual_weights = 10
    args.neuron_membership_funcs = [(-0.1411, -0.0896, -0.0896, -0.0380), (1.1548,  1.7547,  1.7547,  2.3545)]
    # args.manual_rules = [(0, "A4", "placed", "placed"), (0, "B4", "linear", "yours"), (0, "B4", "flipped", "flipped"), (1, "B4", "placed", "placed"), (1, "C4", "linear", "yours"), (1, "C4", "flipped", "flipped")]
    # args.manual_rules = [(0, "B4", "linear", "yours"), (0, "C4", "linear", "yours"), (0, "C4", "flipped", "flipped"), (1, "A4", "placed", "placed")]
    # args.manual_rules = [(0, "B4", "linear", "yours"), (0, "C4", "linear", "yours"), (0, "C4", "flipped", "flipped"), (1, "A0", "flipped", "flipped")]
    # args.manual_rules = [(0, "B4", "linear", "yours"), (0, "C4", "linear", "yours"), (0, "C4", "flipped", "flipped")]
    args.manual_rules = [(0, "A4", "placed", "placed"), (0, "B4", "flipped", "flipped"), (1, "B4", "placed", "placed"), (1, "B4", "linear", "yours"), (1, "C4", "flipped", "flipped")]
    trainer = FuzzyTrainer(model, args, train_base_resid=False)
    trainer.train()
else:
    args = FuzzyTrainingArgs(run_number, layer=layer, single_neuron=neuron)
    args.num_games_train = 10000
    args.batch_size = 100
    args.rules_count = 2
    args.undersampling_factor = 1 # ONlY Positive counts ...
    args.sparcity_factor = 10 # 0.2
    args.only_0_1_factor = 0 # 0.1
    args.lr = 1e-1
    args.fuzzy_and = "mul"
    args.fuzzy_or = "prob_sum"
    args.initialization = "uniform"
    args.weight_decay_loss_func = "sqrt"
    args.manual = False
    args.debug = True
    args.use_rule_weights = True
    args.learn_rules = True
    # args.neuron_membership_funcs = [(-0.8, -0.8, 0.5, 0.5), (1.6, 1.6, 3, 3)]
    args.neuron_membership_funcs = [(-0.1411, -0.0896, -0.0896, -0.0380), (1.1548,  1.7547,  1.7547,  2.3545)]
    trainer = FuzzyTrainer(model, args, train_base_resid=False)
    trainer.train()
rules_dict2 = get_rules_dict(trainer.weights, neuron=0)
print(rules_dict2)

# %%
print(trainer.rule_weights)

# %%
print_rules_dict(rules_dict2, rules_count=10)
True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
True,  True, False,  True,  True,  True,  True,  True,  True,  True],

# %%
layer = 1
neuron = 421

activations : Float[Tensor, "batch pos d_mlp"] = focus_cache["mlp_post", layer][:50, :, neuron].unsqueeze(2)
activations = einops.rearrange(activations, "batch pos d_mlp -> (batch pos d_mlp)")
activations = activations[activations <= 0]
print(len(activations))
fig = px.histogram(x=activations.to("cpu"), nbins=50, range_x=[-0.2, 0])
fig.update_layout(
    title=f"Activations of Neurons in Layer {layer} of the MLP",
    xaxis_title="Activations",
    yaxis_title="Count",
)
fig.show()

# %%
layer = 1
neuron = 421

activations : Float[Tensor, "batch pos d_mlp"] = focus_cache["mlp_post", layer][:200, :, neuron].unsqueeze(2)
activations = einops.rearrange(activations, "batch pos d_mlp -> (batch pos d_mlp)")
activations = activations[activations > 0]
print(len(activations))
fig = px.histogram(x=activations.to("cpu"), nbins=50, range_x=[0, 4])
fig.update_layout(
    title=f"Activations of Neurons in Layer {layer} of the MLP",
    xaxis_title="Activations",
    yaxis_title="Count",
)
fig.show()

# %% [markdown]
# # Dimensionality Reduction Technique
# - Erst einmal drauf los
# - Dann lecture angucken und richtig machen
# - TODO: Look at how much variance is explained by the principle components
# 

# %%


# %% [markdown]
# 

# %%
layer = 1
neuron = 421

groups = 4
components = 2

activations : Float[Tensor, "batch pos"] = focus_cache["mlp_post", layer][:200, :, neuron]
activations : Float[Tensor, "batch"] = einops.rearrange(activations, "batch pos -> (batch pos)")
resid : Float[Tensor, "batch pos d_mmodel"] = focus_cache[f"blocks.{1}.ln2.hook_normalized"]
resid : Float[Tensor, "batch d_model"] = einops.rearrange(resid, "batch pos d_model -> (batch pos) d_model")
indeces = t.arange(activations.shape[0]).to("cuda")
indeces = indeces[activations > 0]
resid = resid[activations > 0]
variables : Float[Tensor, "batch variables"]= get_variables(resid, 1)

# Now we use pca and k-means to cluster the variables
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Perform PCA
pca = PCA(n_components=components)
pca_result = pca.fit_transform(variables.cpu().numpy())

# Perform K-means clustering
kmeans = KMeans(n_clusters=groups, random_state=42)
kmeans.fit(pca_result)

# Get cluster labels
cluster_labels = kmeans.labels_

# Plot the clusters
fig = plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Variable Clustering')
plt.colorbar(scatter)
plt.show()

# Print how much variance is explained by the principle compontens
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

# %%
print("hello worlld")

# %%
print("aölksdjf")

# %%
# For all each each group vizualize the board state at the group examples
how_many = 5
boards : Float[Tensor, "probe batch 8 8"] = t.Tensor(3, how_many, 8, 8).to(device)
for group in range(groups):
    print(f"Group {group}")
    count = 0
    i = 0
    while count < how_many:
        if cluster_labels[i].item() == group:
            resid_example : Float[Tensor, "d_model"] = resid[i]
            for j, probe_name in enumerate(["linear", "flipped", "placed"]):
                probe : Float[Tensor, "d_model rows cols options"]  = get_probe(layer, probe_name, "mid")[0]
                result = einops.einsum(resid_example, probe, "d_model, d_model rows cols options -> rows cols options").softmax(dim=-1)
                if probe_name == "linear":
                    result = result[:, :, YOURS] - result[:, :, MINE]
                elif probe_name == "flipped":
                    result = result[:, :, FLIPPED] - result[:, :, NOT_FLIPPED]
                elif probe_name == "placed":
                    result = result[:, :, PLACED] - result[:, :, NOT_PLACED]
                boards[j, count] = result
            count += 1
        i += 1
    plot_boards_general(
        x_labels = ["linear", "flipped", "placed"],
        y_labels = [f"Game {i}" for i in range(how_many)],
    boards = boards,
    )


