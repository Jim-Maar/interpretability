"""Looks up the hand-verified example from the post (neuron L1N1411 and its C2-D3 rule) in each classification.

Run after `experiment.py`: `uv run check_example_neuron.py`
"""

import torch as t

from common import NEURON_THRESHOLD, get_all_rules
from experiment import CONFIGS, RESULTS_DIR

EXAMPLE_LAYER = 1
EXAMPLE_NEURON = 1411
EXAMPLE_RULE = (
    "(C2 yours AND D3 mine AND E4 placed) OR (C2 yours AND D3 mine AND E4 flipped AND F5 placed) OR "
    "(C2 yours AND D3 mine AND E4 flipped AND F5 flipped AND G6 placed) OR "
    "(C2 yours AND D3 mine AND E4 flipped AND F5 flipped AND G6 flipped AND H7 placed)"
)


def main():
    rule_index = list(get_all_rules()).index(EXAMPLE_RULE)
    for config in CONFIGS:
        saved = t.load(RESULTS_DIR / f"classification_{config.name}.pt")
        kept = saved["kept"][EXAMPLE_LAYER, rule_index]
        mean_on_rule = saved["approx"][EXAMPLE_LAYER, rule_index, EXAMPLE_NEURON].item()
        difference = saved["difference"][EXAMPLE_LAYER, rule_index, EXAMPLE_NEURON].item()
        samples = saved["counts_per_pos"][EXAMPLE_LAYER, rule_index].sum().item()
        rank = (saved["difference"][EXAMPLE_LAYER, rule_index] > difference).sum().item() + 1
        print(
            f"{config.name:36s} samples {samples:6.0f}  L1N1411 kept {bool(kept[EXAMPLE_NEURON])!s:5s}  "
            f"mean act on rule {mean_on_rule:6.3f}  difference {difference:6.3f} (rank {rank})  "
            f"neurons kept for this rule {kept.sum().item()}"
        )
    print(f"(threshold {NEURON_THRESHOLD})")


if __name__ == "__main__":
    main()
