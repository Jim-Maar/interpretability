# %% [markdown]
# # Prove the Flipping Circuit ...
# - Darauf achten
#     - Maximal Modular machen immer kleine Funktionen mit High Cohesion und Low Coupling
#     - für alle Tensoren Typing verwenden.
#     - Möglichst oft kleine Testfunktionenen schreiben...
# - Das ist ein bisschen ein Gamble. Ich böller das einfach heute fertig und wenn es klappt ist Insane! Ansonsten reevaluiere ich meine Situation!

# %% [markdown]
# ## Setup

# %%
from utils import *
import pickle
# focus_logits, focus_cache = get_focus_logits_and_cache()

# %%
# Load Datasets ...
board_seqs_int_train = t.load(
    os.path.join(
        section_dir,
        "data/board_seqs_int_train.pth",
    )
)
board_seqs_int_test = t.load(
    os.path.join(
        section_dir,
        "data/board_seqs_int_valid.pth",
    )
)
'''# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
board_seqs_string = t.load(
    os.path.join(
        section_dir, 
        "data/board_seqs_string_train.pth",
    )
)'''


# %% [markdown]
# ## Util Functions

# %%
DEBUG = False
if DEBUG:
    board_seqs_int_test = board_seqs_int

if not DEBUG:
    START_TRAIN = 0
    BATCH_SIZE = 500
    NUM_GAMES_TRAIN = 10000 # with 60 GB RAM I can do 20.000 (I could allocate more though haha)
    START_VALID = 0
    NUM_GAMES_VALID = 10000
    NUM_RULES = None
    MLP_BATCH_SIZE = 1000
else:
    START_TRAIN = 0
    BATCH_SIZE = 50
    NUM_GAMES_TRAIN = 500
    START_VALID = 0
    NUM_GAMES_VALID = 50
    NUM_RULES = 50
    MLP_BATCH_SIZE = 50

def get_activation(board_seqs_int, act_names, num_games=1000, start=0, games = []):
    # TODO: If this takes to long or something, Make a filter step!
    inference_size = 200
    if len(games) > 0:
        num_games = len(games)
    iterate = range(start, start+num_games, inference_size)
    if num_games > 1000:
        iterate = tqdm(iterate, total=num_games//inference_size)
    print("Getting activations ...")
    act_name_results = {act_name : [] for act_name in act_names}
    for batch in iterate:
        input_games = list(range(batch, min(batch + inference_size, batch + num_games)))
        if len(games) > 0:
            input_games = [games[i] for i in input_games]
        with t.inference_mode():
            _, cache = model.run_with_cache(
                board_seqs_int[input_games, :-1].to(device),
                return_type=None,
                names_filter=lambda name: name in act_names
                # names_filter=lambda name: name == f"blocks.{layer}.hook_resid_mid" or name == f"blocks.{layer}.mlp.hook_post"
                # names_filter=lambda name: name == f"blocks.{layer}.hook_resid_pre" or name == f"blocks.{layer}.mlp.hook_post"
            )
        for act_name in act_names:
            act_name_results[act_name] += [cache[act_name].detach().cpu()]
    for act_name in act_names:
        act_name_results[act_name] = t.cat(act_name_results[act_name], dim=0)
        act_name_results[act_name] = act_name_results[act_name][:num_games]
    return act_name_results

probes = {}
probe_names = ["linear", "flipped", "placed"]
for probe_name in probe_names:
    probe = []
    for layer in range(8):
        probe_in_layer = get_probe(layer, probe_name, "post")[0].detach()
        probe.append(probe_in_layer)
    probe : Float[Tensor, "layer d_model row col options"]= t.stack(probe, dim=0)
    probes[probe_name] = probe

# %% [markdown]
# ## Get the Rule for each Game/Position

# %% [markdown]
# ### Get the Rules ...

# %%
def get_string_from_rule(rule):
    return " OR ".join([f"({' AND '.join(conjunction)})" for conjunction in rule])

def get_features_in_line_rules_function(lines):
    def get_rules():
        rules = {}
        # line_length = len(lines[0])
        # assert all([len(line) == line_length for line in lines])
        # assert line_length > 1
        for row in range(8):
            for col in range(8):
                for row_delta in [-1, 0, 1]:
                    for col_delta in [-1, 0, 1]:
                        if row_delta == 0 and col_delta == 0:
                            continue
                        rule = []
                        for line, middle_point in lines:
                            line_length = len(line)
                            line_length_front = line_length - middle_point - 1
                            line_length_back = middle_point
                            row_start = row - line_length_back * row_delta
                            col_start = col - line_length_back * col_delta
                            row_end = row + line_length_front * row_delta
                            col_end = col + line_length_front * col_delta
                            if row_start < 0 or row_start >= 8 or col_start < 0 or col_start >= 8:
                                continue
                            if row_end < 0 or row_end >= 8 or col_end < 0 or col_end >= 8:
                                continue
                            conjunction = []
                            for i in range(line_length):
                                row_new = row + (i - middle_point) * row_delta
                                col_new = col + (i - middle_point) * col_delta
                                label = tuple_to_label((row_new, col_new))
                                features = line[i] # Den Shit zuende machen und testen ....
                                for feature in features:
                                    conjunction.append(f"{label} {feature}")
                            rule.append(conjunction)
                        if len(rule) == 0:
                            continue
                        rules[get_string_from_rule(rule)] = rule
        return rules
    return get_rules

def get_all_rules(rule_template_list):
    all_rules = {}
    for lines in rule_template_list:
        get_rules_function = get_features_in_line_rules_function(lines)
        rules = get_rules_function()
        all_rules.update(rules)
    return all_rules

line_rules_dict = {
    "flipping_test" : [([["flipped"], ["placed"]], 0)], # Die Reihenfolge ist sehr relevant
    "flipping" : [([["flipped"], ["placed"]], 0), # Die Reihenfolge ist sehr relevant
                  ([["flipped"], ["flipped"], ["placed"]], 0), # I like this because it's very simple!
                  ([["flipped"], ["flipped"], ["flipped"], ["placed"]], 0),
                  ([["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 0),
                  ([["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 0),
                  ([["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 0),
                  ([["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 0)],
}

# PROBLEM: Am ende not_empty hinzufügen macht alles kaputt im Moment, ich müsste dazu machen, dass die Line sich um das letzte Flipped drehen soll litereally

flipping_extra_list = [
    [([["yours"], ["mine"], ["placed"]], 1),
     ([["yours"], ["mine"], ["flipped"], ["placed"]], 1),
     ([["yours"], ["mine"], ["flipped"], ["flipped"], ["placed"]], 1),
     ([["yours"], ["mine"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 1),
     ([["yours"], ["mine"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 1),
     ([["yours"], ["mine"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 1)],
    [([["yours"], ["mine"], ["mine"], ["placed"]], 2),
     ([["yours"], ["mine"], ["mine"], ["flipped"], ["placed"]], 2),
     ([["yours"], ["mine"], ["mine"], ["flipped"], ["flipped"], ["placed"]], 2),
     ([["yours"], ["mine"], ["mine"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 2),
     ([["yours"], ["mine"], ["mine"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 2)],
    [([["yours"], ["mine"], ["mine"], ["mine"], ["placed"]], 3),
     ([["yours"], ["mine"], ["mine"], ["mine"], ["flipped"], ["placed"]], 3),
     ([["yours"], ["mine"], ["mine"], ["mine"], ["flipped"], ["flipped"], ["placed"]], 3),
     ([["yours"], ["mine"], ["mine"], ["mine"], ["flipped"], ["flipped"], ["flipped"], ["placed"]], 3)],
    [([["yours"], ["mine"], ["mine"], ["mine"], ["mine"], ["placed"]], 4),
     ([["yours"], ["mine"], ["mine"], ["mine"], ["mine"], ["flipped"], ["placed"]], 4),
     ([["yours"], ["mine"], ["mine"], ["mine"], ["mine"], ["flipped"], ["flipped"], ["placed"]], 4)],
    [([["yours"], ["mine"], ["mine"], ["mine"], ["mine"], ["mine"], ["placed"]], 5),
     ([["yours"], ["mine"], ["mine"], ["mine"], ["mine"], ["mine"], ["flipped"], ["placed"]], 5)],
    [([["yours"], ["mine"], ["mine"], ["mine"], ["mine"], ["mine"], ["mine"], ["placed"]], 6)],
]

# Also mit so vielen Regeln wird das scheiß lange daruern
# Komprimieren zu 4 Regeln? würde gehen ... wäre aber wahrscheinlich schlechter ...
# Ich könnte eine Funktion machen die Effizient alle Regeln findet ... Ich weiß garnicht wie einfach das ist, ich muss ja mit vektorisierten Funktionen competen (Das ist schlechte Idee)
# Das sind insgesamt so 7000 Regeln maybe 
# => Ich probiers einfach. Worst case dauerts halt


# Ich bin mir grade nciht sicher: davor schienen ja mit der alten Regel ganz okaye Ergebnisse zu bekommen und jede Regel benötigt mehr Zeit
# lostlostlostlsotssdltostosotstlotlso
# Eigneltich müsste ich halt gucken welche Regenl actually diese Neurons gut beschreiben ... Meine Fresse
# Okay oKay . da ich wirklich merke ich habe keinen Bock ahhhhhhhh 
# Oha nur 4000 Regeln

rules = get_all_rules(flipping_extra_list)
# print(list(rules.keys())[1000:1200])
print(len(rules))

# %% [markdown]
# ### Action

# %%
def get_probe_results(resid : Float[Tensor, "batch layer pos d_model"]) -> Dict[str, Float[Tensor, "batch layer pos row col"]]:
    probe_results = {}
    for probe_name in probe_names:
        probe = probes[probe_name]
        probe_result = einops.einsum(resid, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        probe_result = probe_result.argmax(dim=-1)
        probe_results[probe_name] = probe_result
    return probe_results

def get_probe_name_and_option_from_feature(feature):
    if feature == "flipped":
        return "flipped", FLIPPED
    elif feature == "placed":
        return "placed", PLACED
    elif feature == "yours":
        return "linear", YOURS
    elif feature == "mine":
        return "linear", MINE
    elif feature == "empty":
        return "linear", EMPTY
    else:
        raise ValueError(f"Unknown feature: {feature}")

def get_games_positions_layers_that_follow_rule(probe_results, rule):
    rule_bool = t.zeros(BATCH_SIZE, 8, 59, device=device, dtype=t.bool)
    for conjunction in rule:
        conjunction_bool = t.ones(BATCH_SIZE, 8, 59, device=device, dtype=t.bool)
        for literal in conjunction:
            label, feature = literal.split(" ") # Look if I can make Not_Feature possible. Otherwise change the rules ... Not_Empty doesen't work!
            not_feature = False
            if feature[:4] == "not_":
                feature = feature[4:]
                not_feature = True
            row, col = label_to_tuple(label)
            probe_name, option = get_probe_name_and_option_from_feature(feature)
            if not_feature:
                literal_bool = probe_results[probe_name][:, :, :, row, col] != option
            else:
                literal_bool = probe_results[probe_name][:, :, :, row, col] == option
            conjunction_bool &= literal_bool
        rule_bool |= conjunction_bool
    return rule_bool

# TODO: For each rule, layer and position I need the games where the rule is active, Think about what the format should be for the next steps ...
# TODO: Make this Modular so that I can also apply it to board_seqs_int_test
# TODO: Mhh this takes a shitload of time ...
# NO: for each game, layer and position I need one Rule that is active, What about no Rule ... I guess I could add just placed or something
# Take out games with no rule ...
def get_fake_cache(board_seqs_int, num_games, start):
    act_names = [f"blocks.{layer}.ln2.hook_normalized" for layer in range(8)]
    act_names += [utils.get_act_name("mlp_post", layer) for layer in range(8)]
    fake_cache = get_activation(board_seqs_int, act_names, num_games, start)
    return fake_cache

def get_games_for_rule_layer(rules, fake_cache, start, num_games):
    games_for_rule_layer = {}
    for rule_idx, rule_str in enumerate(rules):
        for layer in range(8):
            games_for_rule_layer[(rule_idx, layer)] = []

    for rule_idx, rule_str in tqdm(enumerate(rules), total=len(rules)):
        rule = rules[rule_str]
        for batch in range(start, start + num_games, BATCH_SIZE):
            act_name_results = {act_name : result[batch:batch+BATCH_SIZE].to(device) for act_name, result in fake_cache.items() if "ln2" in act_name}
            resid : Float[Tensor, "batch layer pos d_model"] = t.stack(list(act_name_results.values()), dim=1)
            probe_result = get_probe_results(resid)
            # print(rule_idx, batch)
            rule_bool : Float[Tensor, "batch layer pos"] = get_games_positions_layers_that_follow_rule(probe_result, rule)
            for layer in range(8): # This actually takes most of the time ...
                for pos in range(59):
                    games_for_rule_layer_pos = (rule_bool[:, layer, pos].nonzero().flatten() + start).tolist()
                    games_for_rule_layer[(rule_idx, layer)] += [(game, pos) for game in games_for_rule_layer_pos]
    return games_for_rule_layer

# Get avg Neuron Acts for each position and layer 
def get_avg_mlp_over_pos(num_games = 1000):
    act_names = [utils.get_act_name("mlp_post", layer) for layer in range(8)]
    fake_cache = get_activation(board_seqs_int_train, act_names, num_games=num_games, start=NUM_GAMES_TRAIN)
    avg_mlp_post = t.stack([fake_cache[act_name].to(device) for act_name in act_names])
    avg_mlp_post = avg_mlp_post.mean(dim=1)
    avg_mlp_post_over_pos = avg_mlp_post.mean(dim=1)
    return avg_mlp_post_over_pos, avg_mlp_post

# TODO: For each rule, layer and pos: compute the top neurons
# TODO: First I need to calculate all the mlp_post activations beforehand ...
# TODO: Then I need to check that the rules are actually working correctly ...
neuron_acts_diff_dict = {}
neuron_acts_dict = {}
# num_games_for_rule_layer= {}
def get_neuron_acts_diff_dict(rules, avg_mlp_post_over_pos, games_for_rule_layer, fake_cache):
    for rule_idx, rule_str in tqdm(enumerate(rules), total=len(rules)):
        rule = rules[rule_str]
        for layer in range(8):
            act_name = utils.get_act_name("mlp_post", layer)
            act_names = [act_name]
            games_and_positions = games_for_rule_layer[(rule_idx, layer)]
            if len(games_and_positions) == 0:
                continue
            index_tensor = t.Tensor(games_and_positions).to(dtype=t.int)
            # print(rule_idx, layer, pos, len(games))
            avg_mlp_post_layer = fake_cache[act_name].to(device)[index_tensor[:, 0], index_tensor[:, 1]].mean(dim=0)
            neuron_acts_diff = avg_mlp_post_layer - avg_mlp_post_over_pos[layer]
            neuron_acts_diff_dict[(rule_idx, layer)] = neuron_acts_diff
            # num_games_for_rule_layer[(rule_idx, layer)] = len(games_and_positions)
            neuron_acts_dict[(rule_idx, layer)] = avg_mlp_post_layer
            # if len(games_and_positions) > 5 and len(games_and_positions) < 500 and layer <= 4:
            #     print(rule_idx, layer, len(games_and_positions), neuron_acts_diff.abs().mean().item())
    return neuron_acts_diff_dict, neuron_acts_dict

# TODO: Turn this into a function that I can easily swap out ...
def get_top_neurons_and_activations_per_rule_layer(neuron_acts_diff_dict, neuron_acts_dict, num_neurons=30, neuron_mean_activation_difference_threshold=0.17):
    top_neurons_and_activations_per_rule_layer = {}
    for rule, layer in neuron_acts_diff_dict:
        neuron_acts_diff = neuron_acts_diff_dict[(rule, layer)]
        neuron_acts = neuron_acts_dict[(rule, layer)]
        # Get top neurons based on activation differences
        top_neurons = neuron_acts_diff.abs().topk(num_neurons, largest=True)
        neuron_indices = top_neurons.indices
        # Get corresponding activations
        selected_neuron_acts = neuron_acts[neuron_indices]
        # Filter based on threshold
        mask = selected_neuron_acts >= neuron_mean_activation_difference_threshold
        filtered_indices = neuron_indices[mask]
        filtered_acts = selected_neuron_acts[mask]    
        # Only store if there are neurons that meet the criteria
        if len(filtered_indices) > 0:
            top_neurons_and_activations_per_rule_layer[(rule, layer)] = (filtered_indices, filtered_acts)
    return top_neurons_and_activations_per_rule_layer

def get_rules_for_game_layer_pos(games_for_rule_layer):
    rules_for_game_layer_pos = {}
    for rule_idx, layer in games_for_rule_layer:
        # Qickfix
        # if debug and rule_idx > 10:
        #     continue
        games_and_positions = games_for_rule_layer[(rule_idx, layer)]
        for game, pos in games_and_positions:
            if (game, layer, pos) not in rules_for_game_layer_pos:
                rules_for_game_layer_pos[(game, layer, pos)] = []
            rules_for_game_layer_pos[(game, layer, pos)].append(rule_idx)
    return rules_for_game_layer_pos

def get_neurons_for_game_layer_pos(rules_for_game_layer_pos, top_neurons_and_activations_per_rule_layer):
    neuron_indices_for_game_layer_pos = {}
    neuron_activations_for_game_layer_pos = {}
    for game, layer, pos in rules_for_game_layer_pos:
        rules = rules_for_game_layer_pos[(game, layer, pos)]
        neuron_indices_all_rules = []
        neuron_activations_all_rules = []
        for rule in rules:
            if (rule, layer) not in top_neurons_and_activations_per_rule_layer:
                continue
            neuron_indices, neuron_activations = top_neurons_and_activations_per_rule_layer[(rule, layer)]
            neuron_indices_all_rules += neuron_indices.tolist()
            neuron_activations_all_rules += neuron_activations.tolist()
        neuron_to_activations_dict = {}
        for neuron_idx, neuron_act in zip(neuron_indices_all_rules, neuron_activations_all_rules):
            if neuron_idx not in neuron_to_activations_dict:
                neuron_to_activations_dict[neuron_idx] = []
            neuron_to_activations_dict[neuron_idx].append(neuron_act)
        for neuron_idx in neuron_to_activations_dict:
            neuron_to_activations_dict[neuron_idx] = max(neuron_to_activations_dict[neuron_idx])
        if not neuron_to_activations_dict:
            continue
        neuron_indices_for_game_layer_pos[(game, layer, pos)] = list(neuron_to_activations_dict.keys())
        neuron_activations_for_game_layer_pos[(game, layer, pos)] = list(neuron_to_activations_dict.values())
    return neuron_indices_for_game_layer_pos, neuron_activations_for_game_layer_pos

def bundle_fake_cache(fake_cache):
    bundled_fake_cache = {}
    key_names = list(set([".".join(key.split(".")[2:]) for key in fake_cache]))
    for key_name in key_names:
        act_name_results = {act_name : result for act_name, result in fake_cache.items() if key_name in act_name}
        stacked_result : Float[Tensor, "batch layer pos neurons"] = t.stack(list(act_name_results.values()), dim=1)
        bundled_fake_cache[key_name] = stacked_result
    return bundled_fake_cache

# %%
# Dinge die noch flasch sein könnten: Ich habe inputs vergessen und die Funktion nimmt die dann von den globalen ...
# TODO Encapsulate this in a function ...
# Die Funktion hat kack Cohesion
def run_all(rules, num_neurons=30, neruon_mean_activation_difference_threshold = 0.17):
    #  line_rules_name = "flipping_test"
    # line_rule_function = get_features_in_line_rules_function(line_rules_dict[line_rules_name])
    # print("Getting Line Rules ...")
    # rules2 = line_rule_function()

    print("Getting Fake Cache ...")
    fake_cache = get_fake_cache(board_seqs_int_train, NUM_GAMES_TRAIN, START_TRAIN)

    print("Getting Games for Rule Layer ...")
    games_for_rule_layer = get_games_for_rule_layer(rules, fake_cache, start=START_TRAIN, num_games=NUM_GAMES_TRAIN)

    print("Getting Avg MLP Post Over Pos ...")
    avg_mlp_post_over_pos, avg_mlp_post = get_avg_mlp_over_pos(MLP_BATCH_SIZE)

    print("Getting Neuron Acts Diff Dict ...")
    neuron_acts_diff_dict, neuron_acts_dict = get_neuron_acts_diff_dict(rules, avg_mlp_post_over_pos, games_for_rule_layer, fake_cache)

    print("Getting Top Neurons and Activations Per Rule Layer ...")
    top_neurons_and_activations_per_rule_layer = get_top_neurons_and_activations_per_rule_layer(neuron_acts_diff_dict, neuron_acts_dict, num_neurons, neruon_mean_activation_difference_threshold)
    return games_for_rule_layer, avg_mlp_post_over_pos, avg_mlp_post, neuron_acts_diff_dict, neuron_acts_dict, top_neurons_and_activations_per_rule_layer

def run_all_valid(rules, top_neurons_and_activations_per_rule_layer):
    print("Getting Rules for Game Layer Pos ...")
    # Eigentlich müsste ich das hier auf dem Test Set machen ...
    act_names = [utils.get_act_name("mlp_post", layer) for layer in range(8)]
    act_names += [utils.get_act_name("attn_out", layer) for layer in range(8)]
    act_names += [utils.get_act_name("resid_pre", layer) for layer in range(8)]
    act_names += [utils.get_act_name("resid_post", layer) for layer in range(8)]
    act_names += [f"blocks.{layer}.ln1.hook_normalized" for layer in range(8)]
    act_names += [f"blocks.{layer}.ln2.hook_normalized" for layer in range(8)]
    fake_cache_valid = get_activation(board_seqs_int_test, act_names, start=START_VALID, num_games=NUM_GAMES_VALID)
    # fake_cache_valid = get_fake_cache(board_seqs_int_test, NUM_GAMES_VALID, START_VALID)
    games_for_rule_layer_valid = get_games_for_rule_layer(rules, fake_cache_valid, start=START_VALID, num_games=NUM_GAMES_VALID)
    rules_for_game_layer_pos_valid = get_rules_for_game_layer_pos(games_for_rule_layer_valid)
    neuron_indices_for_game_layer_pos_valid, neuron_activations_for_game_layer_pos_valid = get_neurons_for_game_layer_pos(rules_for_game_layer_pos_valid, top_neurons_and_activations_per_rule_layer)
    return fake_cache_valid, neuron_indices_for_game_layer_pos_valid, neuron_activations_for_game_layer_pos_valid

# %%
W_out = model.W_out.detach()
b_out = model.b_out.detach()
b_out_repeat = einops.repeat(b_out, "layer d_model -> layer pos d_model", pos=59)

def get_masks(flipped_final_real, flipped_final_pred):
    mask = t.zeros_like(flipped_final_real).to(dtype=t.int)
    only_real_mask = t.zeros_like(flipped_final_real).to(dtype=t.int)
    only_pred_mask = t.zeros_like(flipped_final_real).to(dtype=t.int)
    mask[:, 0] = flipped_final_real[:, 0]
    for layer in range(1, 8):
        change_real = (flipped_final_real[:, layer - 1] != flipped_final_real[:, layer])
        change_pred = (flipped_final_real[:, layer - 1] != flipped_final_pred[:, layer]) # This is correct!
        mask[:, layer] = (change_real | change_pred).to(dtype=t.int)
        only_real_mask[:, layer] = change_real.to(dtype=t.int)
        only_pred_mask[:, layer] = change_pred.to(dtype=t.int)
    return mask, only_real_mask, only_pred_mask

def orthogonalize_vector_to_group(a, B, normalize=True):
    """Orthogonalizes vector a against a list of vectors B without in-place modification using PyTorch"""
    orthogonal_a = a.clone()  # Create a copy of a to avoid in-place modification
    B_prev = []
    for b in B:
        if not all([b @ b_prev < 1e-6 for b_prev in B_prev]):
            b = orthogonalize_vector_to_group(b, B_prev)
        # Project orthogonal_a onto b
        projection = einops.repeat(einops.einsum(a, b, "... d_model, d_model -> ...") / t.dot(b, b), "... -> ... d_model", d_model = b.shape[0]) * b
        # Update orthogonal_a by subtracting the projection
        orthogonal_a = orthogonal_a - projection
        B_prev += [b]
    
    # Normalize the resulting vector orthogonal_a
    if normalize:
        orthogonal_a = orthogonal_a / t.norm(orthogonal_a)
    
    return orthogonal_a

def orthogonalize_vectors(vectors, normalize=True):
    new_vectors = []
    for vector in vectors:
        vector = orthogonalize_vector_to_group(vector, new_vectors, normalize=normalize)
        new_vectors += [vector]
    return new_vectors

# %%
def run_beginning(num_neurons):
    rules = get_all_rules(flipping_extra_list)
    if DEBUG:
        keys = list(rules.keys())
        random.shuffle(keys)
        rules = {key: rules[key] for key in keys[:NUM_RULES]}
    # Neuron_Acts_Diff_Dict: Contains Mean Neuron Activation Difference for each Rule and Layer
    games_for_rule_layer, avg_mlp_post_over_pos, avg_mlp_post, neuron_acts_diff_dict, neuron_acts_dict, top_neurons_and_activations_per_rule_layer = run_all(rules, num_neurons)

    # top_neurons_and_activations_per_rule_layer = get_top_neurons_and_activations_per_rule_layer(neuron_acts_diff_dict, neuron_acts_dict, 2048)
    fake_cache_valid, neuron_indices_for_game_layer_pos_valid, neuron_activations_for_game_layer_pos_valid = run_all_valid(rules, top_neurons_and_activations_per_rule_layer)

    bundled_fake_cache_valid = bundle_fake_cache(fake_cache_valid)
    return bundled_fake_cache_valid, neuron_indices_for_game_layer_pos_valid, neuron_activations_for_game_layer_pos_valid, avg_mlp_post, neuron_acts_diff_dict

def evaluate_rules(bundled_fake_cache_valid, neuron_indices_for_game_layer_pos_valid, neuron_activations_for_game_layer_pos_valid, avg_mlp_post, use_real_attention, use_real_neuron_acts):
    results_dict = {
    "avg_neuron_count" : t.zeros(8, 59).to(device),
    "abs_mean_diff_flipped" : t.zeros(8, 59, 8, 8).to(device),
    "abs_mean_diff_not_flipped" : t.zeros(8, 59, 8, 8).to(device),
    "TP_diff" : t.zeros(8, 59, 8, 8).to(device),
    "FP_diff" : t.zeros(8, 59, 8, 8).to(device),
    "TN_diff" : t.zeros(8, 59, 8, 8).to(device),
    "FN_diff" : t.zeros(8, 59, 8, 8).to(device),
    "TP_final" : t.zeros(8, 59, 8, 8).to(device),
    # "FP_final" : t.zeros(8, 59, 8, 8).to(device),
    "TN_final" : t.zeros(8, 59, 8, 8).to(device),
    # "FN_final" : t.zeros(8, 59, 8, 8).to(device),
    "TP_normal" : t.zeros(8, 59, 8, 8).to(device),
    "FP_normal" : t.zeros(8, 59, 8, 8).to(device),
    "TN_normal" : t.zeros(8, 59, 8, 8).to(device),
    "FN_normal" : t.zeros(8, 59, 8, 8).to(device),
    }
    total_number_of_neurons = t.zeros(8, 59).to(device)
    total_number_of_predictions = t.zeros(8, 59).to(device)
    mask_sum = t.zeros(8, 59, 8, 8, device=device)
    mask_normal_sum = t.zeros(8, 59, 8, 8, device=device)

    probe = probes["flipped"]
    probe_lists = {}
    for layer in range(8):
        probe_lists[layer] = []
        for row in range(8):
            for col in range(8):
                probe_lists[layer].append(probe[layer, :, row, col, FLIPPED])
    linear_probe = probes["linear"]
        
    for batch in range(START_VALID, START_VALID + NUM_GAMES_VALID, BATCH_SIZE):
        mlp_post_real : Float[Tensor, "batch layer pos neurons"] = bundled_fake_cache_valid["mlp.hook_post"][batch:batch+BATCH_SIZE].to(device)
        resid_pre_real : Float[Tensor, "batch layer pos d_model"] = bundled_fake_cache_valid["hook_resid_pre"][batch:batch+BATCH_SIZE].to(device)
        if use_real_attention:
            attn_out_real : Float[Tensor, "batch layer pos d_model"] = bundled_fake_cache_valid["hook_attn_out"][batch:batch+BATCH_SIZE].to(device)
        else:
            resid_layernorm : Float[Tensor, "batch layer pos d_model"] = bundled_fake_cache_valid["ln1.hook_normalized"][batch:batch+BATCH_SIZE].to(device)
            for layer in range(8):
                resid_layernorm[:, layer] = orthogonalize_vector_to_group(resid_layernorm[:, layer], probe_lists[layer])
                # ...
            
        # resid_post_real : Float[Tensor, "batch layer pos d_model"] = bundled_fake_cache_valid["hook_resid_post"][batch:batch+BATCH_SIZE].to(device)
        num_games = mlp_post_real.shape[0]
        mlp_post_pred = einops.repeat(avg_mlp_post, "layer pos neurons -> batch layer pos neurons", batch=num_games).clone()
        for game in range(num_games):
            for layer in range(8):
                for pos in range(59):
                    if (game, layer, pos) not in neuron_indices_for_game_layer_pos_valid:
                        continue
                    neuron_indices = neuron_indices_for_game_layer_pos_valid[(game, layer, pos)]
                    neuron_acts = t.Tensor(neuron_activations_for_game_layer_pos_valid[(game, layer, pos)]).to(device)
                    if use_real_neuron_acts:
                        neuron_acts = mlp_post_real[game, layer, pos, neuron_indices]
                    mlp_post_pred[game, layer, pos, neuron_indices] = neuron_acts
                    total_number_of_neurons[layer, pos] += len(neuron_indices)
                    total_number_of_predictions[layer, pos] += 1


        mlp_out_real = einops.einsum(mlp_post_real, W_out, "batch layer pos neurons, layer neurons d_model -> batch layer pos d_model")
        # flipped_logits_real = einops.einsum(mlp_out_real + attn_out_real + b_out_repeat, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        flipped_logits_real = einops.einsum(mlp_out_real + attn_out_real + b_out_repeat, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        flipped_real = flipped_logits_real.argmax(dim=-1)

        mlp_out_pred = einops.einsum(mlp_post_pred, W_out, "batch layer pos neurons, layer neurons d_model -> batch layer pos d_model")
        # flipped_logits_pred = einops.einsum(mlp_out_pred + attn_out_real + b_out_repeat, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        flipped_logits_pred = einops.einsum(mlp_out_pred + attn_out_real + b_out_repeat, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        flipped_pred = flipped_logits_pred.argmax(dim=-1)

        resid_post_real = resid_pre_real + mlp_out_real + attn_out_real + b_out_repeat
        final_logits_real = einops.einsum(resid_post_real, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        flipped_final_real = (final_logits_real[:, :, :, :, :, 0] > final_logits_real[:, :, :, :, :, 1]).to(t.int)
        resid_post_pred = resid_pre_real + mlp_out_pred + attn_out_real + b_out_repeat # WTF! I forgot to add the attn_out_real
        final_logits_pred = einops.einsum(resid_post_pred, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        flipped_final_pred = (final_logits_pred[:, :, :, :, :, 0] > final_logits_pred[:, :, :, :, :, 1]).to(t.int)
        mask, only_real_mask, only_pred_mask = get_masks(flipped_final_real, flipped_final_pred)
        mask_sum += mask.sum(dim=0)

        abs_diff_flipped = (flipped_logits_real[:, :, :, :, :, FLIPPED] - flipped_logits_pred[:, :, :, :, :, FLIPPED]).abs().sum(dim=0)
        abs_diff_not_flipped = (flipped_logits_real[:, :, :, :, :, NOT_FLIPPED] - flipped_logits_pred[:, :, :, :, :, NOT_FLIPPED]).abs().sum(dim=0)
        TP = ((flipped_real == FLIPPED) & (flipped_pred == FLIPPED) & mask).sum(dim=0).float()
        FP = ((flipped_real == NOT_FLIPPED) & (flipped_pred == FLIPPED) & mask).sum(dim=0).float()
        TN = ((flipped_real == NOT_FLIPPED) & (flipped_pred == NOT_FLIPPED) & mask).sum(dim=0).float()
        FN = ((flipped_real == FLIPPED) & (flipped_pred == NOT_FLIPPED) & mask).sum(dim=0).float()
        results_dict["abs_mean_diff_flipped"] += abs_diff_flipped
        results_dict["abs_mean_diff_not_flipped"] += abs_diff_not_flipped
        results_dict["TP_diff"] += TP
        results_dict["FP_diff"] += FP
        results_dict["TN_diff"] += TN
        results_dict["FN_diff"] += FN

        flipped_change_real = (flipped_real == FLIPPED) & only_real_mask
        not_flipped_change_real = (flipped_real == NOT_FLIPPED) & only_real_mask
        flipped_change_pred = (flipped_pred == FLIPPED) & only_pred_mask
        not_flipped_change_pred = (flipped_pred == NOT_FLIPPED) & only_pred_mask
        TP_final = (flipped_change_real & flipped_change_pred).sum(dim=0).float()
        FP_final = (not_flipped_change_real & flipped_change_pred).sum(dim=0).float()
        TN_final = (not_flipped_change_real & not_flipped_change_pred).sum(dim=0).float()
        FN_final = (flipped_change_real & not_flipped_change_pred).sum(dim=0).float()
        # DICLAIMER: False Positive and False Negative where not done write, but I can get accuracy using mask_sum ...
        results_dict["TP_final"] += TP_final
        # results_dict["FP_final"] += FP_final
        results_dict["TN_final"] += TN_final
        # results_dict["FN_final"] += FN_final

        linear_logits = einops.einsum(resid_post_real, linear_probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        mask_normal = linear_logits.argmax(dim=-1) != EMPTY
        mask_normal_sum += mask_normal.sum(dim=0)
        TP_normal = ((flipped_real == FLIPPED) & (flipped_pred == FLIPPED) & mask_normal).sum(dim=0).float()
        FP_normal = ((flipped_real == NOT_FLIPPED) & (flipped_pred == FLIPPED) & mask_normal).sum(dim=0).float()
        TN_normal = ((flipped_real == NOT_FLIPPED) & (flipped_pred == NOT_FLIPPED) & mask_normal).sum(dim=0).float()
        FN_normal = ((flipped_real == FLIPPED) & (flipped_pred == NOT_FLIPPED) & mask_normal).sum(dim=0).float()
        results_dict["TP_normal"] += TP_normal
        results_dict["FP_normal"] += FP_normal
        results_dict["TN_normal"] += TN_normal
        results_dict["FN_normal"] += FN_normal

    results_dict["abs_mean_diff_flipped"] /= NUM_GAMES_VALID
    results_dict["abs_mean_diff_not_flipped"] /= NUM_GAMES_VALID
    results_dict["avg_neuron_count"] = total_number_of_neurons / total_number_of_predictions
    return results_dict, mask_sum, mask_normal_sum

def save_top_neurons_and_activations_per_rule_layer_etc(rules):
    # rules = get_all_rules(flipping_extra_list)
    games_for_rule_layer, avg_mlp_post_over_pos, avg_mlp_post, neuron_acts_diff_dict, neuron_acts_dict, top_neurons_and_activations_per_rule_layer = run_all(rules, 2048)
    # Save everything using pickle
    directory = "flipping_circuit_saves"
    with open(f"{directory}/games_for_rule_layer.pth", "wb") as f:
        pickle.dump(games_for_rule_layer, f)
    with open(f"{directory}/avg_mlp_post_over_pos.pth", "wb") as f:
        pickle.dump(avg_mlp_post_over_pos, f)
    with open(f"{directory}/avg_mlp_post.pth", "wb") as f:
        pickle.dump(avg_mlp_post, f)
    with open(f"{directory}/neuron_acts_diff_dict.pth", "wb") as f:
        pickle.dump(neuron_acts_diff_dict, f)
    with open(f"{directory}/neuron_acts_dict.pth", "wb") as f:
        pickle.dump(neuron_acts_dict, f)
    with open(f"{directory}/top_neurons_and_activations_per_rule_layer.pth", "wb") as f:
        pickle.dump(top_neurons_and_activations_per_rule_layer, f)
    print("Saved!")

def load_top_neurons_and_activations_per_rule_layer_etc():
    directory = "flipping_circuit_saves"
    with open(f"{directory}/games_for_rule_layer.pth", "rb") as f:
        games_for_rule_layer = pickle.load(f)
    with open(f"{directory}/avg_mlp_post_over_pos.pth", "rb") as f:
        avg_mlp_post_over_pos = pickle.load(f)
    with open(f"{directory}/avg_mlp_post.pth", "rb") as f:
        avg_mlp_post = pickle.load(f)
    with open(f"{directory}/neuron_acts_diff_dict.pth", "rb") as f:
        neuron_acts_diff_dict = pickle.load(f)
    with open(f"{directory}/neuron_acts_dict.pth", "rb") as f:
        neuron_acts_dict = pickle.load(f)
    with open(f"{directory}/top_neurons_and_activations_per_rule_layer.pth", "rb") as f:
        top_neurons_and_activations_per_rule_layer = pickle.load(f)
    return games_for_rule_layer, avg_mlp_post_over_pos, avg_mlp_post, neuron_acts_diff_dict, neuron_acts_dict, top_neurons_and_activations_per_rule_layer

import gc

if __name__ == "__main__":
    # check how many files are in the flipping_circuit_saves directory
    random.seed(42)
    rules = get_all_rules(flipping_extra_list)
    if DEBUG:
        keys = list(rules.keys())
        # random.shuffle(keys)
        # rules = {key: rules[key] for key in keys[:NUM_RULES]}
        key2 = '(C2 yours AND D3 mine AND E4 mine AND F5 placed) OR (C2 yours AND D3 mine AND E4 mine AND F5 flipped AND G6 placed) OR (C2 yours AND D3 mine AND E4 mine AND F5 flipped AND G6 flipped AND H7 placed)'
        key1 = '(C2 yours AND D3 mine AND E4 placed) OR (C2 yours AND D3 mine AND E4 flipped AND F5 placed) OR (C2 yours AND D3 mine AND E4 flipped AND F5 flipped AND G6 placed) OR (C2 yours AND D3 mine AND E4 flipped AND F5 flipped AND G6 flipped AND H7 placed)'
        rules = {
            key1: rules[key1],
            key2: rules[key2]
        }
    directory = "flipping_circuit_saves"
    num_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    if num_files == 0:
        save_top_neurons_and_activations_per_rule_layer_etc(rules)
    games_for_rule_layer, avg_mlp_post_over_pos, avg_mlp_post, neuron_acts_diff_dict, neuron_acts_dict, top_neurons_and_activations_per_rule_layer = load_top_neurons_and_activations_per_rule_layer_etc()
    print("Loaded!") # TODO: Noch del hinzufürgen bei fake_cache_valid ...
    fake_cache_valid, neuron_indices_for_game_layer_pos_valid, neuron_activations_for_game_layer_pos_valid = run_all_valid(rules, top_neurons_and_activations_per_rule_layer)
    bundled_fake_cache_valid = bundle_fake_cache(fake_cache_valid)
    del fake_cache_valid
    gc.collect()
    # TODO: Multiple Neuron sizes ...
    for use_real_attention in [True]:
        print(f"Running with real_attention: {use_real_attention}")
        for use_real_neuron_acts in [True, False]:
            print(f"Running with real_neuron_acts: {use_real_neuron_acts}")
            results_dict, mask_sum, mask_normal_sum = evaluate_rules(bundled_fake_cache_valid, neuron_indices_for_game_layer_pos_valid, neuron_activations_for_game_layer_pos_valid, avg_mlp_post, use_real_attention, use_real_neuron_acts)
            directory = "flipping_circuit_results"
            run_name = f"num_neurons_{2048}_real_attention_{use_real_attention}_real_neuron_acts_{use_real_neuron_acts}_actually_correct"
            save_path = f"{directory}/{run_name}"
            os.makedirs(save_path, exist_ok=True)
            t.save(results_dict, f"{save_path}/flipped_circuit_results_dict.pth")
            t.save(mask_sum, f"{save_path}/flipped_circuit_mask_sum.pth")
            t.save(mask_normal_sum, f"{save_path}/flipped_circuit_mask_normal_sum.pth")
            t.save(neuron_acts_diff_dict, f"{save_path}/flipped_circuit_neuron_acts_diff_dict.pth")
            print("Saved!")
            if DEBUG:
                break
        if DEBUG:
            break
    print("Done")