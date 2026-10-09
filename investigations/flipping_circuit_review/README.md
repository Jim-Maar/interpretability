# Review of the flipping circuit experiment

This folder reviews the experiment behind the "Flipping Circuit Hypothesis" (thesis chapter 7.3 and 7.4, and the LessWrong post "Exploring how OthelloGPT computes its world model"). The original code lives in `flipping_circuit/prove_flipped_circuit_2.py` and `flipping_circuit/analyze_flipped_circuit_results.ipynb`. It is left untouched. Everything here is a separate reimplementation that can run the original pipeline and fixed versions of it side by side.

## Short answer

Yes, there was a bug, and it explains the negative result. The script that assigns neurons to rules records the wrong game for every rule hit outside the first batch of 500 games. So 95% of the activations it averages come from unrelated games. In the original run, the neuron set it produced was no better than a random set of neurons of the same size. The neuron L1N1411, which motivated the whole experiment, was not even assigned to its own rule.

With the bug fixed, the rule neurons clearly beat both the baseline and a random control. But they only explain part of what the MLP does. So the hypothesis comes out partly supported, not refuted.

## Terms used below

- **Flipped tile.** In Othello, placing a tile flips every opponent tile that lies in a straight line between the new tile and another tile of the mover's colour. The far tile of the mover's colour is called the **anchor**.
- **Mine and yours.** These are the labels of Jim's linear probe. "Yours" means the colour of the player who just moved at this position, and "mine" means the other colour. Before the model has computed the flips, the tiles that are about to flip are still "mine".
- **Rule.** This is a name from the codebase. A rule fixes an anchor tile, a direction and a number *n* between 1 and 6. It is true at a layer when, along that direction, the probes read "yours" on the anchor, then *n* "mine" tiles, then zero or more tiles already read as "flipped", then the "placed" tile. There are 1036 rules. The probes are read at the input of the layer's MLP.
- **Rule neuron.** A neuron that the classification step assigns to a rule (for one layer).
- **Circuit.** At each game, layer and position, the circuit is the union of the rule neurons of all rules that are true there. Every other neuron of that layer's MLP is mean-ablated, using its mean at that position.
- **Rim.** The outer ring of the board. The **centre** is the inner 6×6 squares.
- **The original metric.** This is the "final" accuracy in the analysis notebook, which produced thesis Table 7.1 and the plot in the post. It looks at tiles where the flipped-probe decision on resid_post changes from layer L−1 to layer L, either in the real run or in the ablated run. It counts as correct the cases where both runs change.

## What I checked

1. I read the post, thesis chapter 7, the three versions of `prove_flipped_circuit*.py`, the baseline script, the analysis notebook, `utils.py` and the probe training code.
2. I rebuilt the setup on a CPU. This used the same synthetic OthelloGPT weights, Jim's saved probes from `probes/`, and freshly generated random legal games, because the original `data/` folder is not in the repo. `sanity_checks.py` confirms the setup:
   - The model's top-1 move is legal 99.95% of the time.
   - The probes read the board well from resid_post. Across layers 0 to 7, the linear probe reaches 0.91 to 0.99 accuracy and the flipped probe reaches 0.71 to 0.97 recall.
   - My fast rule evaluator gives exactly the same booleans as Jim's rule loop.
3. `experiment.py` reruns the whole pipeline. It uses the same sizes as the original: 10,000 training games for classification, 1,000 further games for the mean activations, and 10,000 validation games for rule detection. Scoring uses the first 2,000 validation games, to save CPU time.
4. With the original logic, the rerun reproduces the thesis almost exactly. This is the strongest evidence that the rerun really is Jim's experiment:

   | | circuit | circuit (approx. acts) | baseline |
   |---|---|---|---|
   | Thesis Table 7.1 | 15.4 | 13.2 | 14.2 |
   | Rerun, original logic | 15.3 | 13.2 | 14.2 |

   | Neurons kept per layer | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
   |---|---|---|---|---|---|---|---|---|
   | Thesis Table 7.2 | 148 | 21 | 48 | 86 | 199 | 291 | 448 | 97 |
   | Rerun, original logic | 148 | 20 | 48 | 86 | 200 | 287 | 450 | 97 |

## Findings

### 1. Rule hits are recorded under the wrong game (real bug, certain)

In `get_games_for_rule_layer` (`prove_flipped_circuit_2.py:290`), the game index of a rule hit is computed as

```python
(rule_bool[:, layer, pos].nonzero().flatten() + start).tolist()
```

Here `start` is always 0. It should be `batch`, the first game of the current batch of 500. So a rule hit in game 700 is recorded as game 200. The function `check_index_bug` in `sanity_checks.py` runs this code on a rule that is true only in game 700, and it prints `(200, 5)`. The same line appears in all three versions of the script.

This has two consequences:

- **Classification.** The mean activation of a neuron "on a rule" is averaged over the right positions but mostly over the wrong games. Only the first batch of 500 out of 10,000 games is correct, so the rule signal is diluted about 20 times and replaced by the neuron's average activation at those positions. `check_example_neuron.py` shows this for the neuron from the post. On its own rule (C2 yours, D3 mine, then a tile placed at E4, F5, G6 or H7), L1N1411's mean activation is 2.09 when indexed correctly, the highest of all 2048 layer-1 neurons. The original code computes 0.13 instead, which is below the 0.17 cutoff. So the original circuit drops L1N1411 for this rule and keeps 12 other neurons.
- **Evaluation.** Validation rule hits are stored the same way. So when the evaluation loop asks for the rules of game *e*, it gets the union of the rules of all 20 games that share *e*'s index modulo 500.

This probably explains why the hand check of L1N1411 passed. I am inferring that it ran with the script's `DEBUG` settings, which use 500 training games in batches of 50. There the bug mixes only 10 batches, and the diluted value is 0.29, which still passes 0.17 (`check_debug_scale.py`). At the full scale of 20 batches it drops to 0.13 and fails.

The effect on the result is large. With the original logic, the circuit is no better than the random control:

| Recall of MLP-driven flips | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Baseline (mean-ablate the whole layer) | 0.15 | 0.19 | 0.29 | 0.36 |
| Original circuit | 0.18 | 0.24 | 0.37 | 0.47 |
| Random neurons, same count as the original circuit | 0.16 | 0.21 | 0.31 | 0.41 |
| Fixed circuit | 0.38 | 0.38 | 0.44 | 0.52 |
| Random neurons, same count as the fixed circuit | 0.16 | 0.20 | 0.29 | 0.37 |

"Fixed circuit" here is the `fix_index_difference_same_position` setup described below. The metric is explained under "What the corrected experiment looks like".

### 2. The code selects neurons by activation, not by activation difference (real mismatch, certain; small effect on the scores)

The thesis and the post select a neuron when its mean activation difference is above 0.17. The code (`prove_flipped_circuit_2.py:336-341`) first takes the top `num_neurons` neurons by absolute difference. The main run uses `num_neurons = 2048`, which is all neurons, so this step does nothing. It then filters with `selected_neuron_acts >= 0.17`, where `selected_neuron_acts` is the raw mean activation on rule-true samples, not the difference. So any neuron that is generally active at those positions passes.

The histograms in the thesis (Figure 7.3) and the post are computed from the difference dictionary (`neuron_acts_diff_dict`). So they do not show the neurons that were actually kept.

After fixing the index, switching to the difference changes the scores very little. Jim's metric is 21.8 in both cases. But it roughly halves the circuit in layers 3 to 6. In layer 6, for example, it goes from 349 to 159 neurons.

### 3. The difference is taken against the mean over all positions (deviation from the thesis, certain; matters mainly in layers 6 and 7)

The thesis defines the difference as "mean when the rule is true" minus "mean when the rule is false". The code subtracts the neuron's mean over all 59 positions (`avg_mlp_post_over_pos`) instead. Rules are strongly tied to particular positions, and many neurons have position-dependent baselines, so this confounds rule and position. Comparing against the mean at the same positions removes the confound. It shrinks the layer-6 circuit from 159 to 83 neurons and the layer-7 circuit from 61 to 26. It leaves layers 1 to 5 almost unchanged.

### 4. The rule detector reads an off-distribution input (measured inaccuracy, but not the cause)

Rules are evaluated by applying the probes trained on resid_post of layer L to `blocks.L.ln2.hook_normalized`. That is the layer-normalized resid_mid, a different input than the probe was trained on. On that input, the "placed" probe's precision drops to 0.51 at layer 0 and 0.36 at layer 6. In every other combination it is 1.00. The repo also contains probes trained on resid_mid (`probes/mid/`), and they read resid_mid accurately: "placed" has precision 1.00 everywhere.

However, using those mid probes made the fixed circuit somewhat *worse*: 18.0 instead of 21.7 on Jim's metric. That is mostly because they detect fewer rule hits (the L1N1411 rule drops from 1160 to 325 hits). So this is a real imprecision, but it does not explain the negative result. I don't know why the off-distribution read-out works better here.

### 5. The original metric hides differences between setups (by design, certain)

- Layer 0 always scores 0. `get_masks` sets the "changed" masks only from layer 1 on, but it counts layer 0 in the denominator.
- A "change" compares the layer L−1 probe on resid_post at L−1 with the layer L probe on resid_post at L. These are two different probes, so some changes are probe disagreements rather than changes in the model.
- Most changes come from the attention layer, which the experiment keeps intact. Those count as correct for every setup, including the baseline. That is why all numbers sit near 14%.

None of this is a bug that flips the conclusion. But it makes a real improvement look small. With the index fixed, the original metric still separates the setups: 21.7 for the circuit, 14.4 for the random control and 14.2 for the baseline.

### Things I checked that are fine

- Hook names exist and mean what the code assumes in TransformerLens 2.x: `blocks.L.mlp.hook_post`, `blocks.L.ln2.hook_normalized`, `hook_resid_pre`, `hook_attn_out` and `hook_resid_post`.
- `resid_pre + attn_out + mlp_post @ W_out + b_out` reconstructs resid_post exactly. `bundle_fake_cache` stacks the layers in the right order.
- The mine and yours encoding is consistent. The probes were trained with "yours" meaning the colour of the player who just moved, and the rules use "yours" for the anchor and "mine" for the tiles still to be flipped. That is correct for the input of the MLP, before the flips are computed.
- Classification uses the training set and evaluation uses the validation set. The mean activations come from 1000 further training games. There is no leak between them.
- The evaluation is per layer: one MLP is ablated at a time, with the real input from earlier layers. That is lenient, because errors cannot compound across layers, but it is not wrong.
- Two minor issues. The probe labels take "who just moved" from the move's parity, which is wrong after a pass, but passes are rare. The notebook histogram loops over `range(1035)` while there are 1036 rules.

## What the corrected experiment looks like

The setup I would call "corrected" fixes the index bug and selects neurons by the thesis definition: mean activation on rule-true samples minus the expected mean at the same positions, at least 0.17. It keeps the original rule detector and the original data sizes. I score it with Jim's metric and with two metrics that isolate the MLP:

- **Recall of MLP-driven flips.** The flipped probe of layer L is applied to resid_mid and to resid_post of layer L. These are the same probe before and after this MLP. The metric counts the tiles where the decision changes between them, and asks how often the ablated MLP produces the same final decision.
- **Effect recovered.** This uses the MLP's contribution to the flipped logit difference (flipped minus not flipped) of each tile. It is 1 minus the mean absolute error of the ablated MLP's contribution, divided by the same error for full mean ablation. So 1 means perfect, 0 means no better than ablating the whole layer, and negative means worse. I report it over all tiles and over only the tiles where the MLP flips the decision.

The controls are full mean ablation of the layer (the baseline), and a random set of neurons of the same size at every game, layer and position. The random neurons keep their real activations.

Results for the corrected setup (`fix_index_difference_same_position`), rule neurons with real activations:

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| Neurons kept | 27 | 30 | 37 | 41 | 55 | 65 | 83 | 26 |
| Original metric, centre: circuit | 0.00 | 0.33 | 0.37 | 0.40 | 0.42 | 0.33 | 0.20 | 0.53 |
| Original metric, centre: baseline | 0.00 | 0.20 | 0.22 | 0.25 | 0.22 | 0.17 | 0.16 | 0.53 |
| Recall of MLP flips: circuit | 0.54 | 0.38 | 0.38 | 0.44 | 0.52 | 0.75 | 0.96 | 0.87 |
| Recall of MLP flips: random control | 0.46 | 0.16 | 0.20 | 0.29 | 0.37 | 0.70 | 0.95 | 0.90 |
| Recall of MLP flips: baseline | 0.46 | 0.15 | 0.19 | 0.29 | 0.36 | 0.70 | 0.95 | 0.90 |
| Effect recovered on MLP-flip tiles, centre: circuit | 0.24 | 0.26 | 0.25 | 0.23 | 0.28 | 0.17 | 0.16 | 0.08 |
| Effect recovered on MLP-flip tiles, centre: random control | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 | 0.00 |
| Effect recovered on all tiles, centre: circuit | −0.00 | −0.06 | −0.03 | −0.00 | 0.01 | 0.01 | 0.02 | −0.11 |
| Effect recovered on all tiles, rim: circuit | −0.16 | −0.24 | −0.13 | 0.02 | 0.02 | 0.00 | −0.00 | 0.16 |

When the kept neurons are set to their mean activation on the rule instead of their real activation (the thesis's "approximated neuron activations" variant), recall in layers 1 to 4 is 0.32, 0.30, 0.35 and 0.43. Effect recovered on MLP-flip tiles drops to about 0.07 to 0.11.

The circuit keeps only neurons whose activation goes up when the rule is true. Neurons that switch off for a rule are mean-ablated, which pushes their output in the wrong direction. To test whether this matters, I added a variant that keeps every neuron whose difference is at least 0.17 in either direction (`fix_index_absolute_difference_same_position`). It keeps more neurons in later layers, 145 instead of 83 in layer 6, but it scores the same: 21.3 on Jim's metric and the same recall. So neurons that switch off do not account for the missing part.

All setups and all tables are in `results/summary.md`.

## What this means for the Flipping Circuit Hypothesis

- The original negative result was an artifact of the index bug. Most of the rule-neuron assignment was noise, and the circuit it produced behaved like a random set of neurons.
- After the fix, the rule neurons carry real, rule-specific flipping information. In layers 1 to 4, about 30 to 55 neurons per position recover 1.4 to 2.4 times as many MLP-driven flips as the same number of random neurons. The gap is largest in layers 1 and 2. On the tiles the MLP flips, they also recover about a quarter of the MLP's effect on the flipped direction, where random neurons recover almost nothing.
- They are still far from a complete explanation. About half of the MLP-driven flips in layers 1 to 4 are not reproduced. On tiles that should not change, keeping only these neurons is often worse than ablating everything, especially on the rim. This suggests that the rule neurons' output is normally partly cancelled by other neurons that the circuit mean-ablates. Those other neurons are not rule-specific by this measure, because keeping neurons that switch off for a rule did not help. The variant with constant activations does clearly worse, so these neurons are not simple on/off rule detectors either.
- Together, this fits a picture where flipping is spread over many neurons. Some of them line up with these rules, but the rules capture only part of what the MLP computes. That is closer to the "bag of heuristics" view than evidence against it. I would not read the corrected result as strong support or strong refutation of the hypothesis. It is a partial positive.

## How to run

Run these from this folder. They need about 4 GB of RAM and no GPU. The first run downloads the 101 MB model weights and generates the games.

```
uv run sanity_checks.py          # setup checks and the index-bug demo (1 min)
uv run experiment.py             # full rerun (about 40 min on 4 CPU cores), writes results/results.json
uv run summarize.py              # prints the tables (results/summary.md)
uv run check_example_neuron.py   # L1N1411 in every classification
uv run check_debug_scale.py      # L1N1411 at the DEBUG scale of the original script
```

## Caveats

- The games are freshly generated random legal games, not the original `data/*.pth` files. That the thesis numbers are reproduced to within 0.1 percentage points suggests this does not matter.
- Scoring uses 2,000 validation games instead of 10,000. Rule detection still uses all 10,000, because the original bug mixes rule sets across them.
- I did not try other thresholds, resample ablation, or a full forward pass with all layers ablated at once. Those would be the natural next steps, as would looking at the failure cases of a single rule, as the post suggests.
