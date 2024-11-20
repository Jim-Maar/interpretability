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
focus_logits, focus_cache = get_focus_logits_and_cache()

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
START_TRAIN = 0
BATCH_SIZE = 1000
NUM_GAMES_TRAIN = 100000

def get_activation(board_seqs_int, act_names, num_games, start=0):
    # TODO: If this takes to long or something, Make a filter step!
    print("Getting activations ...")
    act_name_results = {act_name : [] for act_name in act_names}
    inference_size = 1000
    for batch in tqdm(range(start, start+num_games, inference_size), total=num_games//inference_size):
        with t.inference_mode():
            _, cache = model.run_with_cache(
                board_seqs_int[batch:batch+inference_size, :-1].to(device),
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
                        for line in lines:
                            line_length = len(line)
                            row_end = row + (line_length - 1) * row_delta
                            col_end = col + (line_length - 1) * col_delta
                            if row_end < 0 or row_end >= 8 or col_end < 0 or col_end >= 8:
                                continue
                            conjunction = []
                            for i in range(line_length):
                                row_new = row + i * row_delta
                                col_new = col + i * col_delta
                                label = tuple_to_label((row_new, col_new))
                                features = line[i]
                                for feature in features:
                                    conjunction.append(f"{label} {feature}")
                            rule.append(conjunction)
                        if len(rule) == 0:
                            continue
                        rules[get_string_from_rule(rule)] = rule
        return rules
    return get_rules

line_rules_dict = {
    "flipping_test" : [[["flipped"], ["placed"]]], # Die Reihenfolge ist sehr relevant
    "flipping" : [[["flipped"], ["placed"]], # Die Reihenfolge ist sehr relevant
                  [["flipped"], ["flipped"], ["placed"]],
                  [["flipped"], ["flipped"], ["flipped"], ["placed"]],
                  [["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]],
                  [["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]],
                  [["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]],
                  [["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["flipped"], ["placed"]]],
}
line_rules_name = "flipping_test"
line_rule_function = get_features_in_line_rules_function(line_rules_dict[line_rules_name])
rules = line_rule_function()
rules

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
            label, feature = literal.split(" ")
            row, col = label_to_tuple(label)
            probe_name, option = get_probe_name_and_option_from_feature(feature)
            literal_bool = probe_results[probe_name][:, :, :, row, col] == option
            conjunction_bool &= literal_bool
        rule_bool |= conjunction_bool
    return rule_bool

# TODO: For each rule, layer and position I need the games where the rule is active, Think about what the format should be for the next steps ...
# TODO: Make this Modular so that I can also apply it to board_seqs_int_test
# TODO: Mhh this takes a shitload of time ...
# NO: for each game, layer and position I need one Rule that is active, What about no Rule ... I guess I could add just placed or something
# Take out games with no rule ...
games_for_rule_layer_pos = {}
for rule_idx, rule_str in enumerate(rules):
    for layer in range(8):
        for pos in range(59):
            games_for_rule_layer_pos[(rule_idx, layer, pos)] = []
act_names = [f"blocks.{layer}.ln2.hook_normalized" for layer in range(8)]
act_name_results_all = get_activation(board_seqs_int_train, act_names, NUM_GAMES_TRAIN, START_TRAIN)

import pickle

# save act_name_results_all using pickle
with open('act_name_results_all.pickle', 'wb') as handle:
    pickle.dump(act_name_results_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
