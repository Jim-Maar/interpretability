"""Recomputes L1N1411's classification for its rule at the DEBUG scale of the original script.

In DEBUG mode the original used 500 training games in batches of 50, so the index bug mixed
10 batches instead of 20. Run after `experiment.py`: `uv run check_debug_scale.py`
"""

import torch as t

from check_example_neuron import EXAMPLE_LAYER, EXAMPLE_NEURON, EXAMPLE_RULE
from common import NEURON_THRESHOLD, get_all_rules, load_model, load_tokens, run_with_cache
import __main__
from experiment import RESULTS_DIR, RuleStatistics

# experiment.py ran as a script, so its cache pickles RuleStatistics as __main__.RuleStatistics.
__main__.RuleStatistics = RuleStatistics

DEBUG_NUM_GAMES_TRAIN = 500
DEBUG_BATCH_SIZE = 50


def main():
    rule_index = list(get_all_rules()).index(EXAMPLE_RULE)
    train_rows, _ = t.load(RESULTS_DIR / "cache_training_scan.pt", weights_only=False)
    rows = train_rows["post_probe_on_ln2"].long()
    selected = (rows[:, 0] < DEBUG_NUM_GAMES_TRAIN) & (rows[:, 1] == EXAMPLE_LAYER) & (rows[:, 3] == rule_index)
    game, pos = rows[selected, 0], rows[selected, 2]
    mlp_post = run_with_cache(load_model(), load_tokens("train")[:DEBUG_NUM_GAMES_TRAIN], ["mlp.hook_post"])["mlp.hook_post"]
    activation = mlp_post[:, EXAMPLE_LAYER, :, EXAMPLE_NEURON]
    correct_mean = activation[game, pos].mean().item()
    buggy_mean = activation[game % DEBUG_BATCH_SIZE, pos].mean().item()
    print(f"{len(game)} rule hits in the first {DEBUG_NUM_GAMES_TRAIN} games")
    print(f"L1N1411 mean activation on the rule, correct index: {correct_mean:.3f}")
    print(f"L1N1411 mean activation on the rule, DEBUG-scale index bug: {buggy_mean:.3f} "
          f"(kept: {buggy_mean >= NEURON_THRESHOLD})")


if __name__ == "__main__":
    main()
