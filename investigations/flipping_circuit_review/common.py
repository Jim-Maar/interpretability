"""Model, probes, data and rule definitions shared by the review scripts.

Everything here mirrors what `flipping_circuit/prove_flipped_circuit_2.py` and `utils.py` do,
so that the review scripts test the same objects as the original experiment.
"""

from pathlib import Path

import numpy as np
import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig

from othello import generate_games, squares_to_tokens

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
PROBE_DIR = REPO_ROOT / "probes"
DATA_DIR = REVIEW_DIR / "data"

N_LAYERS = 8
D_MLP = 2048
N_POS = 59

# Option indices of Jim's probes (utils.py).
EMPTY, YOURS, MINE = 0, 1, 2
FLIPPED, NOT_FLIPPED = 0, 1
PLACED, NOT_PLACED = 0, 1

PROBE_NAMES = ["linear", "flipped", "placed"]
FEATURE_TO_PROBE_OPTION = {
    "flipped": ("flipped", FLIPPED),
    "placed": ("placed", PLACED),
    "yours": ("linear", YOURS),
    "mine": ("linear", MINE),
    "empty": ("linear", EMPTY),
}
ALPHA = "ABCDEFGH"

# The original run sizes from prove_flipped_circuit_2.py.
ORIGINAL_BATCH_SIZE = 500
NUM_GAMES_TRAIN = 10_000
NUM_GAMES_MEAN = 1_000
NUM_GAMES_VALID = 10_000
NEURON_THRESHOLD = 0.17


def load_model() -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=D_MLP,
        d_vocab=61,
        n_ctx=N_POS,
        act_fn="gelu",
        normalization_type="LNPre",
        device="cpu",
    )
    model = HookedTransformer(cfg)
    weights_path = hf_hub_download("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
    model.load_state_dict(t.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def load_probes(module: str) -> dict[str, t.Tensor]:
    """Returns {probe_name: tensor [layer, d_model, row, col, option]}, like the original script."""
    return {
        name: t.stack(
            [
                t.load(PROBE_DIR / module / name / f"resid_{layer}_{name}.pth", map_location="cpu")[0].detach()
                for layer in range(N_LAYERS)
            ]
        )
        for name in PROBE_NAMES
    }


def load_games(split: str) -> np.ndarray:
    """Random legal games as squares 0..63, cached on disk. Splits use different seeds."""
    split_sizes_and_seeds = {
        "train": (NUM_GAMES_TRAIN + NUM_GAMES_MEAN, 1),
        "valid": (NUM_GAMES_VALID, 2),
    }
    num_games, seed = split_sizes_and_seeds[split]
    path = DATA_DIR / f"games_{split}.npy"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        np.save(path, generate_games(num_games, seed))
    return np.load(path)


def load_tokens(split: str) -> t.Tensor:
    return t.from_numpy(squares_to_tokens(load_games(split)))


def tuple_to_label(row: int, col: int) -> str:
    return f"{ALPHA[row]}{col}"


def label_to_tuple(label: str) -> tuple[int, int]:
    return ALPHA.index(label[0]), int(label[1])


# The rule templates from prove_flipped_circuit_2.py, copied verbatim. Each template is a line
# of features plus the index of the line's "middle point". A rule is the OR of all templates in
# one group, for one middle tile and one direction.
FLIPPING_EXTRA_LIST = [
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


def rules_for_template_group(lines) -> dict[str, list[list[str]]]:
    """Port of `get_features_in_line_rules_function` from prove_flipped_circuit_2.py."""
    rules = {}
    for row in range(8):
        for col in range(8):
            for row_delta in [-1, 0, 1]:
                for col_delta in [-1, 0, 1]:
                    if row_delta == 0 and col_delta == 0:
                        continue
                    rule = []
                    for line, middle_point in lines:
                        line_length = len(line)
                        row_start = row - middle_point * row_delta
                        col_start = col - middle_point * col_delta
                        row_end = row + (line_length - middle_point - 1) * row_delta
                        col_end = col + (line_length - middle_point - 1) * col_delta
                        if not (0 <= row_start < 8 and 0 <= col_start < 8 and 0 <= row_end < 8 and 0 <= col_end < 8):
                            continue
                        conjunction = []
                        for i in range(line_length):
                            label = tuple_to_label(row + (i - middle_point) * row_delta, col + (i - middle_point) * col_delta)
                            conjunction += [f"{label} {feature}" for feature in line[i]]
                        rule.append(conjunction)
                    if rule:
                        rules[" OR ".join(f"({' AND '.join(c)})" for c in rule)] = rule
    return rules


def get_all_rules() -> dict[str, list[list[str]]]:
    all_rules = {}
    for lines in FLIPPING_EXTRA_LIST:
        all_rules.update(rules_for_template_group(lines))
    return all_rules


class RuleEvaluator:
    """Evaluates all rules on probe read-outs.

    This computes the same boolean as the original `get_games_positions_layers_that_follow_rule`,
    but compares each (tile, feature) literal only once and reuses it across rules.
    `sanity_checks.py` checks that both agree.
    """

    FEATURES = list(FEATURE_TO_PROBE_OPTION)

    def __init__(self, rules: dict[str, list[list[str]]]):
        self.rules = [[[self.literal_index(literal) for literal in conjunction] for conjunction in rule] for rule in rules.values()]
        self.num_rules = len(self.rules)

    @classmethod
    def literal_index(cls, literal: str) -> int:
        label, feature = literal.split(" ")
        row, col = label_to_tuple(label)
        return cls.FEATURES.index(feature) * 64 + row * 8 + col

    @classmethod
    def literal_table(cls, probe_argmax: dict[str, t.Tensor]) -> t.Tensor:
        """probe_argmax: {probe_name: [..., 8, 8]} -> bool [num_literals, ...]."""
        columns = []
        for feature in cls.FEATURES:
            probe_name, option = FEATURE_TO_PROBE_OPTION[feature]
            columns.append((probe_argmax[probe_name] == option).flatten(-2).movedim(-1, 0))
        return t.cat(columns).contiguous()

    def __call__(self, probe_argmax: dict[str, t.Tensor]) -> t.Tensor:
        """Returns bool [..., num_rules]: which rules are true at each position."""
        table = self.literal_table(probe_argmax)
        results = []
        for rule in self.rules:
            rule_true = t.zeros_like(table[0])
            for conjunction in rule:
                conjunction_true = table[conjunction[0]].clone()
                for literal in conjunction[1:]:
                    conjunction_true &= table[literal]
                rule_true |= conjunction_true
            results.append(rule_true)
        return t.stack(results, dim=-1)


def probe_argmax(resid: t.Tensor, probes: dict[str, t.Tensor]) -> dict[str, t.Tensor]:
    """resid: [batch, layer, pos, d_model] -> {probe_name: [batch, layer, pos, 8, 8]}."""
    return {
        name: t.einsum("blpd,ldrco->blprco", resid, probe).argmax(dim=-1)
        for name, probe in probes.items()
    }


def run_with_cache(model: HookedTransformer, tokens: t.Tensor, hook_suffixes: list[str]) -> dict[str, t.Tensor]:
    """Runs the model on [batch, 60] tokens (dropping the last move, like the original).

    Returns {hook_suffix: [batch, layer, pos, dim]} with the layers stacked.
    """
    names = {f"blocks.{layer}.{suffix}" for layer in range(N_LAYERS) for suffix in hook_suffixes}
    with t.inference_mode():
        _, cache = model.run_with_cache(tokens[:, :-1], return_type=None, names_filter=lambda name: name in names)
    return {
        suffix: t.stack([cache[f"blocks.{layer}.{suffix}"] for layer in range(N_LAYERS)], dim=1)
        for suffix in hook_suffixes
    }
