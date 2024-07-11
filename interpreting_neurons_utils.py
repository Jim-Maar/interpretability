from utils import *

# %%
def get_w_in(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    '''
    Returns the input weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    '''
    # SOLUTION
    w_in = model.W_in[layer, :, neuron].detach().clone()
    if normalize: w_in /= w_in.norm(dim=0, keepdim=True)
    return w_in


def get_w_out(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    '''
    Returns the output weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    '''
    # SOLUTION
    w_out = model.W_out[layer, neuron, :].detach().clone()
    if normalize: w_out /= w_out.norm(dim=0, keepdim=True)
    return  w_out


def calculate_neuron_input_weights(
    model: HookedTransformer,
    probe: Float[Tensor, "modes d_model row col options"],
    layer: int,
    neuron: int,
    probe_option: int,
) -> Float[Tensor, "rows cols"]:
    '''
    Returns tensor of the input weights for the given neuron, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    '''
    # SOLUTION
    w_in = get_w_in(model, layer, neuron, normalize=True)

    return einops.einsum(
        w_in, probe,
        "d_model, modes d_model row col options -> modes row col options",
    )[0, :, :, probe_option]


def calculate_neuron_output_weights(
    model: HookedTransformer,
    probe: Float[Tensor, "modes d_model row col options"],
    layer: int,
    neuron: int,
    probe_option: int,
) -> Float[Tensor, "rows cols"]:
    '''
    Returns tensor of the output weights for the given neuron, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    '''
    # SOLUTION
    w_out = get_w_out(model, layer, neuron, normalize=True)

    return einops.einsum(
        w_out, probe,
        "d_model, modes d_model row col options -> modes row col options",
    )[0, :, :, probe_option]

# %%
from utils import probe_directions_list

# %%
def get_fraction_of_variance_from_neuron_explained_by_probe(
        neuron : Int,
        layer : Int,
        in_out : str,
        tiles : List[Tuple[str, str, str]], # = probe_directions_list,
    ) -> Float:
    # assert type(tiles) == dict
    # TODO: Update asserts and add option for all tiles
    # if not all_tiles:
    # assert all([type(v) == dict and all ([type(v2) == list for _, v2 in v.items()]) for _, v in probe_names_and_directions.items()])
    # else:
    #     assert all([type(v) == list for _, v in probe_names_and_directions.items()])
    if in_out == "in":
        neuron_w = get_w_in(model, layer, neuron, normalize=True)
    else:
        neuron_w = get_w_out(model, layer, neuron, normalize=True)
    probes = []
    # probes_out = []
    for tile_label, probe_name, direction_name in tiles:
        tile_tuple = label_to_tuple(tile_label)
        y, x = tile_tuple
        if in_out == "in":
            probe = get_probe(layer, probe_name, "mid")
        else:
            probe = get_probe(layer, probe_name, "post")
        direction_int = get_direction_int(direction_name)
        probe_direction = probe[0, :, y, x, direction_int]
        probes.append(probe_direction)

    probes = t.stack(probes, dim=-1)
    U, S, Vh = t.svd(
        probes
    )
    return ((neuron_w @ U)[neuron_w @ U > 0]).norm().item()**2

# %%
from utils import plot_boards_general

# %%
# TODO: I trained the linear probe wrong, I need to do it again ...

# %%
from utils import probes
from utils import probe_directions
from collections import defaultdict
from utils import get_short_cut

# %%
def kurtosis(tensor: Tensor, reduced_axes, fisher=True):
    """
    Computes the kurtosis of a tensor over specified dimensions.
    """
    return (((tensor - tensor.mean(dim=reduced_axes, keepdim=True)) / tensor.std(dim=reduced_axes, keepdim=True))**4).mean(dim=reduced_axes, keepdim=False) - fisher*3

def plot_neuron_weights(neurons : Float[Tensor, "d_mlp"], layer : Int, title : str, probe_names_and_directions : Dict[str, List[str]] = probe_directions_list, save=False):
    direction_dict = defaultdict(list)
    for neuron in neurons:
        neuron = neuron.item()
        for in_out in ["in", "out"]:
            for probe_name in probe_names_and_directions:
                for direction_str in probe_names_and_directions[probe_name]:
                    if in_out == "in":
                        probe_module = "mid"
                        probe = get_probe(layer = layer, probe_type = probe_name, probe_module = probe_module).clone()
                    else:
                        probe_module = "post"
                        probe = get_probe(layer = layer, probe_type = probe_name, probe_module = probe_module).clone()
                    if probe.isnan().sum().item() > 0:
                        print(f"Probe {probe_name} in is nan")
                        continue
                    probe_normalized = probe / probe.norm(dim=1, keepdim=True)
                    direction_int = probe_directions[probe_name][direction_str]
                    if in_out == "in":
                        neuron_weights = calculate_neuron_input_weights(model, probe_normalized, layer, neuron, direction_int)
                    else:
                        neuron_weights = calculate_neuron_output_weights(model, probe_normalized, layer, neuron, direction_int)
                    direction_dict[f"{get_short_cut(probe_name)}_{get_short_cut(direction_str)}_{in_out}"].append(neuron_weights)

    for direction, weights in direction_dict.items():
        direction_dict[direction] = t.stack(weights)

    boards = t.stack(list(direction_dict.values()))
    plot_boards_general(
        x_labels = list(direction_dict.keys()),
        y_labels = [f"N{i.item()}" for i in neurons],
        boards = boards,
        title_text = title,
        save=save,
    )

# get_fraction_of_variance_from_neuron_explained_by_probe(neuron = neuron, layer = 1)
def get_max_acitvations_of_neuron(
    cache: ActivationCache,
    layer: int,
    neuron: int,
    num_activations: int = 10,
) -> Tuple[Float[Tensor, "game move"], Float[Tensor, "game move"]]:
    '''
    Returns the top activations for a given neuron in a given layer.
    '''
    # SOLUTION
    post_activations = cache["post", layer][:, :, neuron]
    batch_size, seq_len = post_activations.shape
    post_activations = post_activations.reshape(-1)
    top_activations = post_activations.argsort(descending=True)
    activation_value = post_activations.sort(descending=True)
    top_activations = top_activations[:num_activations]
    top_games = top_activations // seq_len
    top_moves = top_activations % seq_len
    print(f"Top Game: {top_games}, Top Move: {top_moves}")
    print(f"Activation values: {activation_value}")
    return top_games, top_moves
# %%
def get_similiarity(neuron : Int, layer : Int, tiles : List[Tuple[str, str, str, str]], metric = "avg"):
    avg_similiarity = 0
    direction_all = t.zeros([512]).to(device)
    for label, probe_type, feature_str, in_or_out in tiles:
        tile_tuple = label_to_tuple(label)
        y, x = tile_tuple
        feature = get_direction_int(feature_str)
        if in_or_out == "in":
            probe_module = "mid"
            w = get_w_in(model, layer, neuron, normalize=True)
        else:
            probe_module = "post"
            w = get_w_out(model, layer, neuron, normalize=True)
        probe = get_probe(layer, probe_type=probe_type, probe_module=probe_module)
        direction = probe[0, :, y, x, feature]
        direction = direction / direction.norm()
        direction_all += direction
        similiarity = einops.einsum(direction, w, "d_model, d_model ->").item()
        if feature_str == "empty":
            similiarity = similiarity / 3
        avg_similiarity += similiarity
    direction_all = direction_all / direction_all.norm()
    similiarity_all = einops.einsum(direction_all, w, "d_model, d_model ->").item()
    if metric == "avg":
        return avg_similiarity / len(tiles)
    else:
        return similiarity_all