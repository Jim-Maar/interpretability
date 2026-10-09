"""Checks that the review setup reproduces the original objects, plus a minimal check of the index bug.

Run from this folder: `uv run sanity_checks.py`
"""

import numpy as np
import torch as t

from common import (
    FEATURE_TO_PROBE_OPTION,
    FLIPPED,
    MINE,
    PLACED,
    YOURS,
    EMPTY,
    RuleEvaluator,
    get_all_rules,
    label_to_tuple,
    load_games,
    load_model,
    load_probes,
    load_tokens,
    probe_argmax,
    run_with_cache,
)
from othello import BLACK, game_labels, initial_board, legal_moves, tiles_flipped_by

NUM_CHECK_GAMES = 200


def legal_move_sets(moves: list[int]) -> list[set[int]]:
    """The set of legal squares for the move after each position 0..58."""
    board = initial_board()
    player = BLACK
    sets = []
    for square in moves[:-1]:
        if not tiles_flipped_by(board, square // 8, square % 8, player):
            player = -player
        row, col = square // 8, square % 8
        for r, c in tiles_flipped_by(board, row, col, player):
            board[r, c] = player
        board[row, col] = player
        player = -player
        options = legal_moves(board, player) or legal_moves(board, -player)
        sets.append(set(options))
    return sets


def check_model_plays_legal_moves(model, tokens, games):
    with t.inference_mode():
        logits = model(tokens[:NUM_CHECK_GAMES, :-1])
    top_tokens = logits.argmax(dim=-1)
    token_to_square = [None] + [s for s in range(64) if s not in [27, 28, 35, 36]]
    legal = 0
    total = 0
    for game_index in range(NUM_CHECK_GAMES):
        sets = legal_move_sets(list(games[game_index]))
        for pos, legal_set in enumerate(sets):
            total += 1
            legal += token_to_square[top_tokens[game_index, pos].item()] in legal_set
    print(f"Model top-1 move is legal: {legal / total:.4f} (expect about 0.999)")


def ground_truth(games) -> dict[str, t.Tensor]:
    labels = [game_labels(list(game)) for game in games[:NUM_CHECK_GAMES]]
    board = t.from_numpy(np.stack([l["board"] for l in labels]))[:, :59]
    linear = t.full_like(board, EMPTY)
    linear[board == 1] = YOURS
    linear[board == -1] = MINE
    flipped = t.from_numpy(np.stack([l["flipped"] for l in labels]))[:, :59]
    placed = t.from_numpy(np.stack([l["placed"] for l in labels]))[:, :59]
    return {
        "linear": linear,
        "flipped": t.where(flipped == 1, FLIPPED, 1 - FLIPPED),
        "placed": t.where(placed == 1, PLACED, 1 - PLACED),
    }


def check_probe_accuracy(cache, probes_post, probes_mid, truth):
    """Accuracy per layer against the board after the current move.

    `post on resid_post` is how the probes were trained. `post on ln2` is what the original script
    used to decide which rules are true. `mid on resid_mid` is the matching probe for that input.
    """
    readouts = {
        "post probe on resid_post": probe_argmax(cache["hook_resid_post"], probes_post),
        "post probe on ln2.hook_normalized (original rule input)": probe_argmax(cache["ln2.hook_normalized"], probes_post),
        "mid probe on resid_mid": probe_argmax(cache["hook_resid_mid"], probes_mid),
    }
    for readout_name, readout in readouts.items():
        print(f"\n{readout_name}")
        for probe_name in ["linear", "flipped", "placed"]:
            target = truth[probe_name][:, None].expand_as(readout[probe_name])
            if probe_name == "linear":
                accuracy = (readout[probe_name] == target).float().mean(dim=(0, 2, 3, 4))
                detail = ""
            else:
                option = FLIPPED if probe_name == "flipped" else PLACED
                positive = target == option
                recall = ((readout[probe_name] == option) & positive).sum(dim=(0, 2, 3, 4)) / positive.sum(dim=(0, 2, 3, 4))
                predicted = readout[probe_name] == option
                precision = (predicted & positive).sum(dim=(0, 2, 3, 4)) / predicted.sum(dim=(0, 2, 3, 4)).clamp(min=1)
                accuracy = recall
                detail = "  precision " + " ".join(f"{x:.2f}" for x in precision.tolist())
            print(f"  {probe_name:8s} {'recall' if detail else 'acc   '} " + " ".join(f"{x:.2f}" for x in accuracy.tolist()) + detail)


def original_rule_loop(probe_results, rule, batch_size):
    """`get_games_positions_layers_that_follow_rule` from prove_flipped_circuit_2.py, verbatim logic."""
    rule_bool = t.zeros(batch_size, 8, 59, dtype=t.bool)
    for conjunction in rule:
        conjunction_bool = t.ones(batch_size, 8, 59, dtype=t.bool)
        for literal in conjunction:
            label, feature = literal.split(" ")
            row, col = label_to_tuple(label)
            probe_name, option = FEATURE_TO_PROBE_OPTION[feature]
            conjunction_bool &= probe_results[probe_name][:, :, :, row, col] == option
        rule_bool |= conjunction_bool
    return rule_bool


def check_rule_evaluator(cache, probes_post):
    rules = get_all_rules()
    readout = probe_argmax(cache["ln2.hook_normalized"][:50], probes_post)
    fast = RuleEvaluator(rules)(readout)
    slow = t.stack([original_rule_loop(readout, rule, 50) for rule in rules.values()], dim=-1)
    print(f"\nRules: {len(rules)}. Fast evaluator agrees with the original loop: {bool((fast == slow).all())}")
    print("Rules true per (game, layer, pos), mean over layers 0..7: "
          + " ".join(f"{x:.2f}" for x in fast.sum(-1).float().mean(dim=(0, 2)).tolist()))


def check_index_bug():
    """The original bookkeeping, verbatim, with a rule that is true only in game 700 at position 5."""
    start, batch_size, num_games = 0, 500, 1000
    true_game, true_pos = 700, 5
    all_rule_bool = t.zeros(num_games, 8, 59, dtype=t.bool)
    all_rule_bool[true_game, 1, true_pos] = True
    recorded = []
    for batch in range(start, start + num_games, batch_size):
        rule_bool = all_rule_bool[batch : batch + batch_size]
        for layer in range(8):
            for pos in range(59):
                games_for_rule_layer_pos = (rule_bool[:, layer, pos].nonzero().flatten() + start).tolist()
                recorded += [(game, pos) for game in games_for_rule_layer_pos]
    print(f"\nIndex bug check: rule true in game {true_game}, recorded as (game, pos) = {recorded}")


def main():
    model = load_model()
    tokens = load_tokens("valid")
    games = load_games("valid")
    check_model_plays_legal_moves(model, tokens, games)
    cache = run_with_cache(
        model, tokens[:NUM_CHECK_GAMES], ["hook_resid_post", "hook_resid_mid", "ln2.hook_normalized"]
    )
    check_probe_accuracy(cache, load_probes("post"), load_probes("mid"), ground_truth(games))
    check_rule_evaluator(cache, load_probes("post"))
    check_index_bug()


if __name__ == "__main__":
    main()
