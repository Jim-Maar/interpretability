"""Reruns the flipping-circuit ablation experiment, once as originally written and once per fix.

The original script (`flipping_circuit/prove_flipped_circuit_2.py`) keeps all activations of
10,000 games in memory, which needs about 60 GB of RAM. This script computes the same
quantities in a streaming way, so it runs on a CPU with a few GB of RAM.

Run from this folder: `uv run experiment.py`
Results are written to `results/results.json`.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch as t

from common import (
    D_MLP,
    FLIPPED,
    N_LAYERS,
    N_POS,
    NEURON_THRESHOLD,
    NUM_GAMES_MEAN,
    NUM_GAMES_TRAIN,
    NUM_GAMES_VALID,
    ORIGINAL_BATCH_SIZE,
    REVIEW_DIR,
    RuleEvaluator,
    get_all_rules,
    load_model,
    load_probes,
    load_tokens,
    probe_argmax,
    run_with_cache,
)

RESULTS_DIR = REVIEW_DIR / "results"
FORWARD_BATCH_SIZE = 100
# Evaluation holds several full-size copies of mlp.hook_post at once, so it uses smaller batches.
EVAL_BATCH_SIZE = 50
# The original evaluation used all 10,000 validation games. We decide which rules are true on
# all 10,000 games, because the original index bug mixes rule sets across them, but we score
# only the first NUM_GAMES_EVAL games to save CPU time.
NUM_GAMES_EVAL = 2_000
RANDOM_SEED = 0
CENTRE = slice(1, 7)

# Where the rule detector reads the board from.
#   "post_probe_on_ln2": the original choice. It applies the probe trained on resid_post of
#     layer L to blocks.L.ln2.hook_normalized, which is the normalized resid_mid of layer L.
#   "mid_probe_on_resid_mid": the probe trained on resid_mid of layer L, applied to resid_mid.
RULE_INPUTS = ["post_probe_on_ln2", "mid_probe_on_resid_mid"]


@dataclass(frozen=True)
class Config:
    name: str
    index_bug: bool
    # "activation" (original code), "difference" (as described in the thesis), or
    # "absolute_difference" (also keeps neurons that switch off when the rule is true)
    select_on: str
    difference_reference: str  # "all_positions" (original) or "same_position"
    rule_input: str


CONFIGS = [
    Config("original", True, "activation", "all_positions", "post_probe_on_ln2"),
    Config("fix_index", False, "activation", "all_positions", "post_probe_on_ln2"),
    Config("fix_index_select_on_difference", False, "difference", "all_positions", "post_probe_on_ln2"),
    Config("fix_index_difference_same_position", False, "difference", "same_position", "post_probe_on_ln2"),
    Config("all_fixes_mid_probes", False, "difference", "same_position", "mid_probe_on_resid_mid"),
    Config("fix_index_absolute_difference_same_position", False, "absolute_difference", "same_position", "post_probe_on_ln2"),
]


def batches(start: int, stop: int, batch_size: int = FORWARD_BATCH_SIZE):
    for batch_start in range(start, stop, batch_size):
        yield batch_start, min(batch_start + batch_size, stop)


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------------------------
# Phase 1: mean activations and rule detection
# ---------------------------------------------------------------------------------------------


def compute_mean_mlp_post(model, tokens) -> t.Tensor:
    """Mean of mlp.hook_post per layer and position, over the games after the training games.

    This matches `get_avg_mlp_over_pos` in the original, which uses 1000 games starting at 10,000.
    Returns [layer, pos, neuron].
    """
    total = t.zeros(N_LAYERS, N_POS, D_MLP)
    for start, stop in batches(NUM_GAMES_TRAIN, NUM_GAMES_TRAIN + NUM_GAMES_MEAN):
        cache = run_with_cache(model, tokens[start:stop], ["mlp.hook_post"])
        total += cache["mlp.hook_post"].sum(dim=0)
    return total / NUM_GAMES_MEAN


def rule_inputs_from_cache(cache, probes_post, probes_mid) -> dict[str, dict[str, t.Tensor]]:
    return {
        "post_probe_on_ln2": probe_argmax(cache["ln2.hook_normalized"], probes_post),
        "mid_probe_on_resid_mid": probe_argmax(cache["hook_resid_mid"], probes_mid),
    }


def rule_rows(evaluator, readout, first_game: int) -> t.Tensor:
    """int32 rows (game, layer, pos, rule) where the rule is true."""
    rows = evaluator(readout).nonzero().to(t.int32)
    rows[:, 0] += first_game
    return rows


def collect_rule_rows(model, tokens, num_games, evaluator, probes_post, probes_mid) -> dict[str, t.Tensor]:
    """For every rule input, the rows (game, layer, pos, rule) where the rule is true."""
    rows = {rule_input: [] for rule_input in RULE_INPUTS}
    for start, stop in batches(0, num_games):
        cache = run_with_cache(model, tokens[start:stop], ["ln2.hook_normalized", "hook_resid_mid"])
        for rule_input, readout in rule_inputs_from_cache(cache, probes_post, probes_mid).items():
            rows[rule_input].append(rule_rows(evaluator, readout, start))
        if stop % 2000 == 0:
            log(f"  rules collected for games up to {stop}")
    return {rule_input: t.cat(parts) for rule_input, parts in rows.items()}


# ---------------------------------------------------------------------------------------------
# Phase 2: classify neurons per (rule, layer)
# ---------------------------------------------------------------------------------------------


def original_recorded_game(game: t.Tensor) -> t.Tensor:
    """The game index the original bookkeeping records.

    `get_games_for_rule_layer` adds `start` (always 0) instead of `batch` to the index within
    each batch of 500 games. So a rule hit in game g is recorded as game g mod 500.
    """
    return game % ORIGINAL_BATCH_SIZE


class RuleStatistics:
    """Sums of mlp.hook_post over the (game, pos) samples assigned to each (layer, rule)."""

    def __init__(self, num_rules: int):
        self.num_rules = num_rules
        self.sums = t.zeros(N_LAYERS, num_rules, D_MLP)
        self.counts_per_pos = t.zeros(N_LAYERS, num_rules, N_POS)

    def add(self, rows: t.Tensor, mlp_post: t.Tensor, first_game: int):
        """rows must only contain games first_game .. first_game + len(mlp_post)."""
        game, layer, pos, rule = rows.long().unbind(dim=1)
        self.counts_per_pos.index_put_((layer, rule, pos), t.ones(len(rows)), accumulate=True)
        values = mlp_post[game - first_game, layer, pos]
        self.sums.view(-1, D_MLP).index_add_(0, layer * self.num_rules + rule, values)


def scan_training_games(model, tokens, evaluator, probes_post, probes_mid):
    """One pass over the training games. Detects rules and accumulates correctly indexed statistics.

    Returns (rows per rule input, RuleStatistics per rule input).
    """
    rows = {rule_input: [] for rule_input in RULE_INPUTS}
    statistics = {rule_input: RuleStatistics(evaluator.num_rules) for rule_input in RULE_INPUTS}
    hooks = ["ln2.hook_normalized", "hook_resid_mid", "mlp.hook_post"]
    for start, stop in batches(0, NUM_GAMES_TRAIN):
        cache = run_with_cache(model, tokens[start:stop], hooks)
        for rule_input, readout in rule_inputs_from_cache(cache, probes_post, probes_mid).items():
            batch_rows = rule_rows(evaluator, readout, start)
            rows[rule_input].append(batch_rows)
            statistics[rule_input].add(batch_rows, cache["mlp.hook_post"], start)
        if stop % 2000 == 0:
            log(f"  training games scanned up to {stop}")
    return {rule_input: t.cat(parts) for rule_input, parts in rows.items()}, statistics


def original_statistics(model, tokens, rows: t.Tensor, num_rules: int) -> RuleStatistics:
    """Statistics as the original computes them: each rule hit in game g reads game g mod 500."""
    recorded = rows.clone()
    recorded[:, 0] = original_recorded_game(recorded[:, 0])
    statistics = RuleStatistics(num_rules)
    for start, stop in batches(0, ORIGINAL_BATCH_SIZE):
        mlp_post = run_with_cache(model, tokens[start:stop], ["mlp.hook_post"])["mlp.hook_post"]
        in_batch = (recorded[:, 0] >= start) & (recorded[:, 0] < stop)
        statistics.add(recorded[in_batch], mlp_post, start)
    return statistics


def select_neurons(statistics: RuleStatistics, mean_mlp_post, config: Config):
    """Returns (kept, mean_on_rule, difference), each [layer, rule, neuron].

    `mean_on_rule` is the mean activation on the rule's positive samples, which the original uses to
    replace the kept neurons' activations in the "approximated neuron activations" variant.
    """
    counts_per_pos = statistics.counts_per_pos
    counts = counts_per_pos.sum(dim=-1)
    has_samples = counts > 0
    mean_on_rule = statistics.sums / counts.clamp(min=1)[..., None]
    if config.difference_reference == "all_positions":
        reference = mean_mlp_post.mean(dim=1)[:, None, :]
    else:
        expected_sum = t.einsum("lrp,lpn->lrn", counts_per_pos, mean_mlp_post)
        reference = expected_sum / counts.clamp(min=1)[..., None]
    difference = mean_on_rule - reference
    score = {
        "activation": mean_on_rule,
        "difference": difference,
        "absolute_difference": difference.abs(),
    }[config.select_on]
    kept = (score >= NEURON_THRESHOLD) & has_samples[..., None]
    return kept, mean_on_rule, difference


# ---------------------------------------------------------------------------------------------
# Phase 3: evaluate on validation games
# ---------------------------------------------------------------------------------------------


def eval_rule_rows(rows: t.Tensor, index_bug: bool) -> t.Tensor:
    """Validation rule rows (game, layer, pos, rule) with the game replaced by its lookup key.

    Without the bug, the key is the game itself. With the bug, the rule hits of all validation
    games are recorded under g mod 500, and evaluation game e later looks up key e mod 500. So e
    receives the union of the rules of all 20 games that share its index mod 500.
    """
    if not index_bug:
        return rows[rows[:, 0] < NUM_GAMES_EVAL]
    recorded = rows.clone()
    recorded[:, 0] = original_recorded_game(recorded[:, 0])
    return recorded


def dense_rules(rows: t.Tensor, start: int, stop: int, num_rules: int, index_bug: bool) -> t.Tensor:
    """bool [game, layer, pos, rule] for evaluation games start..stop.

    Batches never cross a multiple of 500, so with the bug the keys are one contiguous range.
    """
    key_start = original_recorded_game(t.tensor(start)).item() if index_bug else start
    key_stop = key_start + (stop - start)
    in_batch = rows[(rows[:, 0] >= key_start) & (rows[:, 0] < key_stop)].long()
    dense = t.zeros(stop - start, N_LAYERS, N_POS, num_rules, dtype=t.bool)
    dense[in_batch[:, 0] - key_start, in_batch[:, 1], in_batch[:, 2], in_batch[:, 3]] = True
    return dense


def kept_neurons_per_position(rules_true: t.Tensor, kept: t.Tensor) -> t.Tensor:
    """Union of the kept neurons of all true rules. Returns bool [game, layer, pos, neuron]."""
    return t.einsum("blpr,lrn->blpn", rules_true.float(), kept.float()) > 0


def approximated_activations(rules_true: t.Tensor, kept: t.Tensor, approx: t.Tensor) -> t.Tensor:
    """For each kept neuron, the max of its rule-mean activation over the true rules (as in the original)."""
    values = t.full((*rules_true.shape[:3], D_MLP), -float("inf"))
    masked_approx = t.where(kept, approx, t.tensor(-float("inf")))
    game, layer, pos, rule = rules_true.nonzero().unbind(dim=1)
    chunk = 20_000
    flat_values = values.view(-1, D_MLP)
    flat_index = (game * N_LAYERS + layer) * N_POS + pos
    for start in range(0, len(game), chunk):
        part = slice(start, start + chunk)
        flat_values.index_reduce_(0, flat_index[part], masked_approx[layer[part], rule[part]], "amax")
    return values


def random_matched_mask(kept_mask: t.Tensor, generator: t.Generator) -> t.Tensor:
    """For each (game, layer, pos), as many random neurons as the circuit keeps there."""
    random_mask = t.zeros_like(kept_mask)
    for layer in range(N_LAYERS):
        num_kept = kept_mask[:, layer].sum(dim=-1, keepdim=True)
        permutation = t.rand(kept_mask[:, layer].shape, generator=generator).argsort(dim=-1)
        random_mask[:, layer].scatter_(-1, permutation, t.arange(D_MLP).expand_as(permutation) < num_kept)
    return random_mask


class Scores:
    """Accumulates the original metric and two cleaner metrics, per layer, for centre and rim tiles."""

    REGIONS = ["centre", "rim"]

    def __init__(self):
        zeros = lambda: t.zeros(N_LAYERS, 2)
        self.original_correct, self.original_mask = zeros(), zeros()
        self.mlp_flip_recovered, self.mlp_flip_count = zeros(), zeros()
        self.no_flip_kept, self.no_flip_count = zeros(), zeros()
        self.abs_error, self.abs_error_baseline = zeros(), zeros()
        self.abs_error_on_flips, self.abs_error_baseline_on_flips = zeros(), zeros()
        self.neurons_kept, self.positions_with_neurons = t.zeros(N_LAYERS, N_POS), t.zeros(N_LAYERS, N_POS)

    @staticmethod
    def by_region(values: t.Tensor) -> t.Tensor:
        """values: [game, layer, pos, 8, 8] -> [layer, region] sums."""
        per_tile = values.float().sum(dim=(0, 2))
        centre = per_tile[:, CENTRE, CENTRE].sum(dim=(-2, -1))
        return t.stack([centre, per_tile.sum(dim=(-2, -1)) - centre], dim=-1)

    def as_dict(self) -> dict:
        def ratio(numerator, denominator):
            return (numerator / denominator).tolist()

        per_position_count = self.neurons_kept / self.positions_with_neurons
        return {
            "original_final_accuracy": ratio(self.original_correct, self.original_mask),
            "original_final_accuracy_overall": (self.original_correct.sum() / self.original_mask.sum()).item(),
            "mlp_flip_recall": ratio(self.mlp_flip_recovered, self.mlp_flip_count),
            "no_flip_specificity": ratio(self.no_flip_kept, self.no_flip_count),
            "mlp_flip_count": self.mlp_flip_count.tolist(),
            "effect_recovered": (1 - self.abs_error / self.abs_error_baseline).tolist(),
            "effect_recovered_on_mlp_flips": (1 - self.abs_error_on_flips / self.abs_error_baseline_on_flips).tolist(),
            "avg_neurons_kept": t.nanmean(per_position_count, dim=1).tolist(),
        }


def original_change_masks(final_real: t.Tensor, final_pred: t.Tensor):
    """`get_masks` from the original, verbatim logic. Inputs are int [game, layer, pos, 8, 8]."""
    mask = t.zeros_like(final_real)
    only_real = t.zeros_like(final_real)
    only_pred = t.zeros_like(final_real)
    mask[:, 0] = final_real[:, 0]
    for layer in range(1, N_LAYERS):
        change_real = final_real[:, layer - 1] != final_real[:, layer]
        change_pred = final_real[:, layer - 1] != final_pred[:, layer]
        mask[:, layer] = (change_real | change_pred).int()
        only_real[:, layer] = change_real.int()
        only_pred[:, layer] = change_pred.int()
    return mask, only_real, only_pred


class LayerReadout:
    """The flipped-probe read-outs that the scores need, for one batch of validation games."""

    def __init__(self, cache, model, flipped_probe, mean_mlp_post):
        self.cache = cache
        self.W_out = model.W_out.detach()
        self.b_out = model.b_out.detach()
        self.probe = flipped_probe  # [layer, d_model, 8, 8, 2]
        self.resid_mid = cache["hook_resid_mid"]
        self.real_mlp_out = self.mlp_out(cache["mlp.hook_post"])
        self.baseline_mlp_out = self.mlp_out(mean_mlp_post.expand_as(cache["mlp.hook_post"]))
        self.real_logits = self.logits(cache["hook_resid_post"])
        self.mid_logits = self.logits(self.resid_mid)
        self.real_mlp_logit_diff = self.logit_diff(self.logits(self.real_mlp_out))
        self.baseline_mlp_logit_diff = self.logit_diff(self.logits(self.baseline_mlp_out))

    def mlp_out(self, mlp_post: t.Tensor) -> t.Tensor:
        return t.einsum("blpn,lnd->blpd", mlp_post, self.W_out) + self.b_out[None, :, None, :]

    def logits(self, resid: t.Tensor) -> t.Tensor:
        return t.einsum("blpd,ldrco->blprco", resid, self.probe)

    @staticmethod
    def logit_diff(logits: t.Tensor) -> t.Tensor:
        return logits[..., FLIPPED] - logits[..., 1 - FLIPPED]


def score_prediction(scores: Scores, readout: LayerReadout, pred_mlp_post: t.Tensor):
    pred_mlp_out = readout.mlp_out(pred_mlp_post)
    pred_logits = readout.logits(readout.resid_mid + pred_mlp_out)
    real_logits = readout.real_logits

    # The original "final" metric, computed exactly like evaluate_rules + get_accuracy_pos.
    attn_out = readout.cache["hook_attn_out"]
    delta_real = readout.logits(attn_out + readout.real_mlp_out).argmax(dim=-1)
    delta_pred = readout.logits(attn_out + pred_mlp_out).argmax(dim=-1)
    final_real = (real_logits[..., 0] > real_logits[..., 1]).int()
    final_pred = (pred_logits[..., 0] > pred_logits[..., 1]).int()
    mask, only_real, only_pred = original_change_masks(final_real, final_pred)
    true_positive = (delta_real == FLIPPED) & only_real.bool() & (delta_pred == FLIPPED) & only_pred.bool()
    true_negative = (delta_real != FLIPPED) & only_real.bool() & (delta_pred != FLIPPED) & only_pred.bool()
    scores.original_correct += Scores.by_region(true_positive | true_negative)
    scores.original_mask += Scores.by_region(mask)

    # Cleaner metric 1: tiles where this layer's MLP changes the flipped read-out of the same probe.
    real_decision = real_logits.argmax(dim=-1)
    mid_decision = readout.mid_logits.argmax(dim=-1)
    pred_decision = pred_logits.argmax(dim=-1)
    mlp_flips = real_decision != mid_decision
    scores.mlp_flip_recovered += Scores.by_region(mlp_flips & (pred_decision == real_decision))
    scores.mlp_flip_count += Scores.by_region(mlp_flips)
    scores.no_flip_kept += Scores.by_region(~mlp_flips & (pred_decision == real_decision))
    scores.no_flip_count += Scores.by_region(~mlp_flips)

    # Cleaner metric 2: how much of the MLP's effect on the flipped logit difference is recovered,
    # relative to mean-ablating the whole layer. 1 means perfect, 0 means as bad as full ablation.
    # We report it over all tiles and over only the tiles where the MLP changes the decision.
    pred_diff = LayerReadout.logit_diff(readout.logits(pred_mlp_out))
    abs_error = (pred_diff - readout.real_mlp_logit_diff).abs()
    abs_error_baseline = (readout.baseline_mlp_logit_diff - readout.real_mlp_logit_diff).abs()
    scores.abs_error += Scores.by_region(abs_error)
    scores.abs_error_baseline += Scores.by_region(abs_error_baseline)
    scores.abs_error_on_flips += Scores.by_region(abs_error * mlp_flips)
    scores.abs_error_baseline_on_flips += Scores.by_region(abs_error_baseline * mlp_flips)


def count_kept_neurons(scores: Scores, kept_mask: t.Tensor):
    num_kept = kept_mask.sum(dim=-1).float()  # [game, layer, pos]
    scores.neurons_kept += num_kept.sum(dim=0)
    scores.positions_with_neurons += (num_kept > 0).float().sum(dim=0)


def evaluate(model, tokens, mean_mlp_post, flipped_probe, circuits: dict, num_rules: int) -> dict:
    """circuits: {config name: (config, eval rule rows, kept, approx)}. Returns scores per variant."""
    generator = t.Generator().manual_seed(RANDOM_SEED)
    scores = {"baseline_mean_ablate_all": Scores()}
    for name in circuits:
        for variant in ["real_acts", "approx_acts", "random_matched"]:
            scores[f"{name}/{variant}"] = Scores()
    hooks = ["mlp.hook_post", "hook_resid_mid", "hook_resid_post", "hook_attn_out"]
    for start, stop in batches(0, NUM_GAMES_EVAL, EVAL_BATCH_SIZE):
        cache = run_with_cache(model, tokens[start:stop], hooks)
        readout = LayerReadout(cache, model, flipped_probe, mean_mlp_post)
        real_mlp_post = cache["mlp.hook_post"]
        mean_expanded = mean_mlp_post.expand_as(real_mlp_post)
        score_prediction(scores["baseline_mean_ablate_all"], readout, mean_expanded)
        for name, (config, rows, kept, approx) in circuits.items():
            rules_true = dense_rules(rows, start, stop, num_rules, config.index_bug)
            kept_mask = kept_neurons_per_position(rules_true, kept)
            predictions = {
                "real_acts": lambda: t.where(kept_mask, real_mlp_post, mean_expanded),
                "approx_acts": lambda: t.where(kept_mask, approximated_activations(rules_true, kept, approx), mean_expanded),
                "random_matched": lambda: t.where(random_matched_mask(kept_mask, generator), real_mlp_post, mean_expanded),
            }
            for variant, make_prediction in predictions.items():
                score_prediction(scores[f"{name}/{variant}"], readout, make_prediction())
                count_kept_neurons(scores[f"{name}/{variant}"], kept_mask)
        if stop % 500 == 0:
            log(f"  evaluated games up to {stop}")
    return {name: score.as_dict() for name, score in scores.items()}


# ---------------------------------------------------------------------------------------------


def classification_summary(kept: t.Tensor, counts_per_pos: t.Tensor) -> dict:
    """How many neurons each (rule, layer) keeps, over the (rule, layer) pairs that have samples."""
    has_samples = counts_per_pos.sum(dim=-1) > 0
    per_rule_layer = kept.sum(dim=-1).float()
    return {
        "mean_neurons_per_rule": [per_rule_layer[l][has_samples[l]].mean().item() for l in range(N_LAYERS)],
        "fraction_rules_with_no_neurons": [
            (per_rule_layer[l][has_samples[l]] == 0).float().mean().item() for l in range(N_LAYERS)
        ],
        "rules_with_samples": has_samples.sum(dim=-1).tolist(),
    }


def cached(name: str, compute):
    """Loads a phase's output from results/ if an earlier run saved it, otherwise computes and saves it."""
    path = RESULTS_DIR / f"cache_{name}.pt"
    if path.exists():
        log(f"  loaded {path.name}")
        return t.load(path, weights_only=False)
    value = compute()
    t.save(value, path)
    return value


def main():
    t.set_grad_enabled(False)
    RESULTS_DIR.mkdir(exist_ok=True)
    model = load_model()
    probes_post, probes_mid = load_probes("post"), load_probes("mid")
    rules = get_all_rules()
    evaluator = RuleEvaluator(rules)
    train_tokens, valid_tokens = load_tokens("train"), load_tokens("valid")

    log("Phase 1: mean activations, rule detection and neuron statistics")
    mean_mlp_post = cached("mean_mlp_post", lambda: compute_mean_mlp_post(model, train_tokens))
    train_rows, statistics = cached(
        "training_scan", lambda: scan_training_games(model, train_tokens, evaluator, probes_post, probes_mid)
    )
    buggy_statistics = cached(
        "original_statistics",
        lambda: original_statistics(model, train_tokens, train_rows["post_probe_on_ln2"], len(rules)),
    )
    valid_rows = cached(
        "valid_rules",
        lambda: collect_rule_rows(model, valid_tokens, NUM_GAMES_VALID, evaluator, probes_post, probes_mid),
    )

    log("Phase 2: classify neurons")
    circuits, classification = {}, {}
    for config in CONFIGS:
        config_statistics = buggy_statistics if config.index_bug else statistics[config.rule_input]
        kept, mean_on_rule, difference = select_neurons(config_statistics, mean_mlp_post, config)
        rows = eval_rule_rows(valid_rows[config.rule_input], config.index_bug)
        circuits[config.name] = (config, rows, kept, mean_on_rule)
        classification[config.name] = classification_summary(kept, config_statistics.counts_per_pos)
        t.save(
            {"kept": kept, "mean_on_rule": mean_on_rule, "difference": difference,
             "counts_per_pos": config_statistics.counts_per_pos},
            RESULTS_DIR / f"classification_{config.name}.pt",
        )

    log("Phase 3: evaluate")
    scores = evaluate(model, valid_tokens, mean_mlp_post, probes_post["flipped"], circuits, len(rules))

    output = {
        "configs": [asdict(config) for config in CONFIGS],
        "num_games_eval": NUM_GAMES_EVAL,
        "classification": classification,
        "scores": scores,
    }
    Path(RESULTS_DIR / "results.json").write_text(json.dumps(output, indent=1))
    log("Done")


if __name__ == "__main__":
    main()
