# %%
from interpreting_neurons_utils import *
import wandb
import pickle
import torch.nn as nn

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

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
    def __init__(self, run_number : int, layer : int, single_neuron :int, input : str = "feature", output : str = "neurons"):
        assert 0 <= single_neuron < 2048
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
        self.neurons_count = 1
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
        self.use_fuzzy_set_simple = False

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
# Can I reuse this for simulated annealing? Sure

class FuzzyAndMin(nn.Module):
    def __init__(self):
        super(FuzzyAndMin, self).__init__()
    
    def forward(self, updated_variables : Float[Tensor, "batch neuron rules variables"]):
        return einops.reduce(updated_variables, "batch neuron rules variables -> batch neuron rules", "min")

class FuzzyAndMul(nn.Module):
    def __init__(self):
        super(FuzzyAndMul, self).__init__()
    
    def forward(self, updated_variables : Float[Tensor, "batch neuron rules variables"]):
        return updated_variables.prod(dim=-1)

class FuzzyLogic(nn.Module):
    def __init__(self, args : FuzzyTrainingArgs):
        super(FuzzyLogic, self).__init__()
        if args.fuzzy_and == "mul":
            self.fuzzy_and = FuzzyAndMul()
        elif args.fuzzy_and == "min":
            self.fuzzy_and = FuzzyAndMin()

    def forward(self, input_variables : Float[Tensor, "batch variables"], fuzzy_weights : Float[Tensor, "neuron rules variables"], rule_weights_simoid : Float[Tensor, "rules"]) -> Float[Tensor, "batch neurons rules"]:
        batch, _ = input_variables.shape
        neurons, rules, _ = fuzzy_weights.shape
        fuzzy_weights_r = einops.repeat(fuzzy_weights, "neurons rules variables -> batch neurons rules variables", batch=batch)
        input_variables_r = einops.repeat(input_variables, "batch variables -> batch neurons rules variables", neurons=neurons, rules=rules)
        updated_variables : Float[Tensor, "batch neuron rules variables"] = 1 - fuzzy_weights_r * (1 - input_variables_r)
        fuzzy_mid = self.fuzzy_and(updated_variables) # AND Gate
        fuzzy_mid = fuzzy_mid * rule_weights_simoid
        fuzzy_mid_sum = 1 - fuzzy_mid.sum(dim=-1)
        rule_0 : Float[Tensor, "batch neurons"] = t.max(t.zeros_like(fuzzy_mid_sum).to(device), fuzzy_mid_sum).unsqueeze(dim=-1)
        fuzzy_out = t.cat([rule_0, fuzzy_mid], dim=-1)
        return fuzzy_out

class Defuzzyfication(nn.Module):
    def __init__(self):
        super(Defuzzyfication, self).__init__()

    def forward(self, fuzzy_set_membership : Float[Tensor, "batch neurons rules"], fuzzy_set_weights : Float[Tensor, "rules trap_params"]) -> Float[Tensor, "batch neurons"]:
        fuzzy_set_membership = t.max(fuzzy_set_membership, t.tensor(EPSILON).to(device))
        a : Float[Tensor, "rules"] = fuzzy_set_weights[:, 0]# .item()
        b : Float[Tensor, "rules"] = fuzzy_set_weights[:, 1]# .item()
        c : Float[Tensor, "rules"] = fuzzy_set_weights[:, 2]# .item()
        d : Float[Tensor, "rules"] = fuzzy_set_weights[:, 3]# .item()
        b_new = a + fuzzy_set_membership * (b-a)
        c_new = d - fuzzy_set_membership * (d-c)
        long_base = d - a
        short_base = c_new - b_new
        d_1 = b_new-a
        d_2 = c_new - b_new
        d_3 = d - c_new
        individual_centroids = ((a * d_1 / 2) + (d_1**2 / 6) + (b_new * d_2) + (d_2**2 / 2) + (d * d_3 / 2) - (d_3**2 / 6)) / ((d_1 / 2) + d_2 + (d_3 / 2))
        individual_areas = (long_base + short_base) / 2 * fuzzy_set_membership
        return (individual_centroids * individual_areas).sum(dim=-1) / individual_areas.sum(dim=-1)
    
class DefuzzyficationSimple(nn.Module):
    def __init__(self):
        super(DefuzzyficationSimple, self).__init__()

    def forward(self, fuzzy_set_membership : Float[Tensor, "batch neurons rules"], fuzzy_set_weights_simple : Float[Tensor, "neuron rules"]) -> Float[Tensor, "batch neurons"]:
        fuzzy_set_membership_softmax = fuzzy_set_membership.softmax(dim=-1)
        result = (fuzzy_set_membership_softmax * fuzzy_set_weights_simple).sum(dim=-1)
        return result

class FuzzyInference(nn.Module):
    def __init__(self, args : FuzzyTrainingArgs):
        super(FuzzyInference, self).__init__()
        self.fuzzy_logic = FuzzyLogic(args)
        if args.use_fuzzy_set_simple:
            self.defuzzyfication = DefuzzyficationSimple()
        else:
            self.defuzzyfication = Defuzzyfication()
    
    def forward(self, rule_weights_simoid : Float[Tensor, "rules"],  input_variables : Float[Tensor, "batch variables"], fuzzy_weights : Float[Tensor, "neuron rules variables"], fuzzy_set_weights : Float[Tensor, "rules trap_params"] | Float[Tensor, "neurons rules"]) -> Float[Tensor, "batch neurons"]:
        fuzzy_out : Float[Tensor, "batch neurons rules"] = self.fuzzy_logic(input_variables, fuzzy_weights, rule_weights_simoid)
        neuron_activations : Float[Tensor, "batch neurons"] = self.defuzzyfication(fuzzy_out, fuzzy_set_weights)
        return neuron_activations

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
    

def get_fuzzy_set_weights(args : FuzzyTrainingArgs):
    assert args.single_neuron is not None
    activations : Float[Tensor, "batch pos d_mlp"] = focus_cache["mlp_post", args.layer][:, :, args.single_neuron].unsqueeze(2)
    activations = einops.rearrange(activations, "batch pos d_mlp -> (batch pos d_mlp)")
    negative_mean = activations[activations <= 0].mean()
    negative_std = activations[activations <= 0].std()
    positive_mean = activations[activations > 0].mean()
    positive_std = activations[activations > 0].std()
    fuzzy_set_weights = t.zeros((args.rules_count + 1, 4)).to(device)
    fuzzy_set_weights[0] = t.tensor([negative_mean - negative_std, negative_mean, negative_mean, negative_mean + negative_std]).to(device)
    fuzzy_set_weights[1:] = t.tensor([positive_mean - positive_std, positive_mean, positive_mean, positive_mean + positive_std]).to(device)
    fuzzy_set_weights.requires_grad = True
    return fuzzy_set_weights

# %%
# Rework everything in here than copy it to .py file
# TODO: Make this function swappable
# TODO: This function currently only works for a single neuron
def get_input_and_output_variables(args : FuzzyTrainingArgs, games_int : Int[Tensor, "batch pos"]) -> Tuple[Float[Tensor, "batch variables"], Float[Tensor, "batch variables"]]:
    with t.inference_mode():
        _, cache = model.run_with_cache(
            games_int[:, :-1].to(device),
            return_type=None,
            names_filter=lambda name: name == f"blocks.{args.layer}.ln2.hook_normalized" or name == f"blocks.{args.layer}.mlp.hook_post"
        )
        # TODO: Make undersampling work for generell case
        # Make the hole thing generel so that it works for all kinds of features not just neurons
        resid_layer_norm = cache[f"blocks.{args.layer}.ln2.hook_normalized"][:, args.pos_start: args.pos_end]
        resid_layer_norm = einops.rearrange(resid_layer_norm, "batch pos d_model -> (batch pos) d_model")
        neuron_activations : Float[Tensor, "batch pos neurons"] = cache["mlp_post", args.layer][:, args.pos_start: args.pos_end]
        neuron_activations = einops.rearrange(neuron_activations, "batch pos neurons -> (batch pos) neurons")
        neuron_activations = neuron_activations[:, args.single_neuron]
        positive_indeces = t.nonzero(neuron_activations > 0).squeeze()
        count_positive = len(positive_indeces)
        negative_indeces = t.nonzero(neuron_activations <= 0).squeeze()[:count_positive]
        indeces = t.cat([positive_indeces, negative_indeces])
        neuron_activations = neuron_activations[indeces]
        resid_layer_norm = resid_layer_norm[indeces]
        input_variables : Float[Tensor, "batch variables"] = get_variables(resid_layer_norm, args.layer)
        output_variables : Float[Tensor, "batch variables"] = neuron_activations
        return input_variables, output_variables
    

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
        if self.args.use_fuzzy_set_simple:
            self.fuzzy_set_weights : Float[Tensor, "neurons rules"] = t.randn(size=(self.args.neurons_count, self.args.rules_count+1), requires_grad=False).to(device)
            self.fuzzy_set_weights.requires_grad = True
        else:
            self.fuzzy_set_weights = get_fuzzy_set_weights(self.args)
        # TODO: This should be initialized with reasonable values
        self.fuzzy_inference = FuzzyInference(self.args)

    def training_step(self, indices: Int[Tensor, "game_idx"], train_or_val="train") -> t.Tensor:
        # Get the game sequences and convert them to state stacks
        games_int = board_seqs_int[indices.cpu()]
        # games_int = tensor of game sequences, each of length 60
        # This is the input to our model
        assert isinstance(games_int, Int[Tensor, f"batch={self.args.batch_size} full_game_len=60"])

        variables, neuron_activations = get_input_and_output_variables(self.args, games_int)
        # print(neuron_activations.shape)
        # SOLUTION
        sigmoid_weights = t.sigmoid(self.weights)
        rule_weights_sigmoid = t.sigmoid(self.rule_weights)
        # TODO: # sigmoid_rule_weights = t.sigmoid(self.rule_weights)
        neuron_activations_pred : Float[Tensor, "batch neurons"] = self.fuzzy_inference(rule_weights_sigmoid, variables, sigmoid_weights, self.fuzzy_set_weights)
        neuron_activations_pred = neuron_activations_pred[:, 0]

        mse : Float[Tensor, "batch neurons"] = (neuron_activations - neuron_activations_pred)**2
        mse_loss = mse.mean()
        if self.args.weight_decay_loss_func == "l1":
            sparcity_loss = sigmoid_weights.mean()
        elif self.args.weight_decay_loss_func == "sqrt":
            sparcity_loss = sigmoid_weights.sqrt().mean()
        else:
            raise NotImplementedError
        spacity_loss2 = (sigmoid_weights.clamp(max=0).abs() + (sigmoid_weights.clamp(min=0, max=0.5) - 0).abs() + ((sigmoid_weights - 1).clamp(min=-0.5, max=0)).abs() + ((sigmoid_weights - 1).clamp(min=0)).abs()).mean()
        loss = mse_loss + self.args.sparcity_factor * sparcity_loss + self.args.only_0_1_factor * spacity_loss2
        
        if train_or_val == "train" and self.step % 10 == 0 and not self.args.debug:
            wandb.log({f"{train_or_val}_loss": loss.item()}, step=self.step)
        if (self.step % 50 == 0 and self.args.debug):
            acc = ((neuron_activations > 0) == (neuron_activations_pred > 0)).sum() / len(neuron_activations)
            rules_dict = get_rules_dict(self.weights, thresh=0.2)
            print(self.step)
            print(rules_dict)
            print_rules_dict(rules_dict, self.args.rules_count)
            print("")
            print(f"Accuracy: {acc.item()}")
            print(((neuron_activations > 0) == (neuron_activations_pred > 0))[:20])
            print(self.rule_weights)
            print(self.fuzzy_set_weights)
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
        if self.args.use_fuzzy_set_simple:
            parameters.append(self.fuzzy_set_weights)
        # if self.args.learn_rules:
        #     parameters.append(self.fuzzy_set_weights)

        if self.args.learn_rules:
            optimizer = t.optim.AdamW(parameters, lr=self.args.lr, betas=self.args.betas)
            # optimizer = t.optim.AdamW([{'params': parameters},{'params': self.fuzzy_set_weights, 'lr' : self.args.lr * 0.01}], lr=self.args.lr, betas=self.args.betas)
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
print(torch.__version__)
print(torch.version.cuda)

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
    # args.manual_rules = [(0, "A4", "placed", "placed"), (0, "B4", "linear", "yours"), (0, "B4", "flipped", "flipped"), (1, "B4", "placed", "placed"), (1, "C4", "linear", "yours"), (1, "C4", "flipped", "flipped")]
    # args.manual_rules = [(0, "B4", "linear", "yours"), (0, "C4", "linear", "yours"), (0, "C4", "flipped", "flipped"), (1, "A4", "placed", "placed")]
    # args.manual_rules = [(0, "B4", "linear", "yours"), (0, "C4", "linear", "yours"), (0, "C4", "flipped", "flipped"), (1, "A0", "flipped", "flipped")]
    # args.manual_rules = [(0, "B4", "linear", "yours"), (0, "C4", "linear", "yours"), (0, "C4", "flipped", "flipped")]
    args.manual_rules = [(0, "A4", "placed", "placed"), (0, "B4", "flipped", "flipped"), (1, "B4", "placed", "placed"), (1, "B4", "linear", "yours"), (1, "C4", "flipped", "flipped")]
    trainer = FuzzyTrainer(model, args, train_base_resid=False)
    trainer.train()
else:
    args = FuzzyTrainingArgs(run_number, layer=layer, single_neuron=neuron)
    args.num_games_train = 10000 # Slighly higher for really good results ...
    # Nice: Geringe Batchsize funktioniert richtig gut ...
    args.batch_size = 5
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
    args.use_rule_weights = False
    args.learn_rules = False
    args.use_fuzzy_set_simple = True
    # args.neuron_membership_funcs = [(-0.8, -0.8, 0.5, 0.5), (1.6, 1.6, 3, 3)]
    trainer = FuzzyTrainer(model, args, train_base_resid=False)
    trainer.train()
rules_dict2 = get_rules_dict(trainer.weights, neuron=0)
print(rules_dict2)

# %%
print(trainer.rule_weights)

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


