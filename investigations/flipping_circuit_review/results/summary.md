### Original metric, overall (thesis Table 7.1: circuit 15.4, approx 13.2, baseline 14.2)

| setup | accuracy |
|---|---|
| baseline_mean_ablate_all | 14.2 |
| original/real_acts | 15.3 |
| original/approx_acts | 13.2 |
| original/random_matched | 15.0 |
| fix_index/real_acts | 21.8 |
| fix_index/approx_acts | 18.2 |
| fix_index/random_matched | 14.7 |
| fix_index_select_on_difference/real_acts | 21.8 |
| fix_index_select_on_difference/approx_acts | 18.1 |
| fix_index_select_on_difference/random_matched | 14.5 |
| fix_index_difference_same_position/real_acts | 21.7 |
| fix_index_difference_same_position/approx_acts | 18.2 |
| fix_index_difference_same_position/random_matched | 14.4 |
| all_fixes_mid_probes/real_acts | 18.0 |
| all_fixes_mid_probes/approx_acts | 16.7 |
| all_fixes_mid_probes/random_matched | 14.3 |
| fix_index_absolute_difference_same_position/real_acts | 21.3 |
| fix_index_absolute_difference_same_position/approx_acts | 17.7 |
| fix_index_absolute_difference_same_position/random_matched | 14.5 |

### Original metric per layer, centre tiles

| setup | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| mean-ablate all (baseline) | 0.00 | 0.20 | 0.22 | 0.25 | 0.22 | 0.17 | 0.16 | 0.53 |
| original / real_acts | 0.00 | 0.20 | 0.21 | 0.23 | 0.24 | 0.27 | 0.36 | 0.80 |
| original / approx_acts | 0.00 | 0.20 | 0.22 | 0.25 | 0.22 | 0.18 | 0.10 | 0.25 |
| original / random_matched | 0.00 | 0.20 | 0.23 | 0.26 | 0.25 | 0.23 | 0.22 | 0.54 |
| fix_index / real_acts | 0.00 | 0.33 | 0.36 | 0.39 | 0.41 | 0.34 | 0.24 | 0.73 |
| fix_index / approx_acts | 0.00 | 0.30 | 0.29 | 0.32 | 0.30 | 0.20 | 0.15 | 0.49 |
| fix_index / random_matched | 0.00 | 0.20 | 0.23 | 0.26 | 0.24 | 0.20 | 0.19 | 0.54 |
| fix_index_select_on_difference / real_acts | 0.00 | 0.33 | 0.37 | 0.40 | 0.42 | 0.33 | 0.21 | 0.73 |
| fix_index_select_on_difference / approx_acts | 0.00 | 0.29 | 0.29 | 0.32 | 0.31 | 0.20 | 0.15 | 0.49 |
| fix_index_select_on_difference / random_matched | 0.00 | 0.20 | 0.23 | 0.26 | 0.23 | 0.19 | 0.17 | 0.53 |
| fix_index_difference_same_position / real_acts | 0.00 | 0.33 | 0.37 | 0.40 | 0.42 | 0.33 | 0.20 | 0.53 |
| fix_index_difference_same_position / approx_acts | 0.00 | 0.29 | 0.29 | 0.32 | 0.31 | 0.20 | 0.15 | 0.51 |
| fix_index_difference_same_position / random_matched | 0.00 | 0.20 | 0.23 | 0.26 | 0.23 | 0.18 | 0.17 | 0.53 |
| all_fixes_mid_probes / real_acts | 0.00 | 0.26 | 0.29 | 0.33 | 0.32 | 0.27 | 0.18 | 0.54 |
| all_fixes_mid_probes / approx_acts | 0.00 | 0.25 | 0.27 | 0.30 | 0.29 | 0.22 | 0.16 | 0.48 |
| all_fixes_mid_probes / random_matched | 0.00 | 0.20 | 0.22 | 0.25 | 0.22 | 0.18 | 0.16 | 0.53 |
| fix_index_absolute_difference_same_position / real_acts | 0.00 | 0.33 | 0.35 | 0.39 | 0.41 | 0.33 | 0.21 | 0.54 |
| fix_index_absolute_difference_same_position / approx_acts | 0.00 | 0.29 | 0.27 | 0.30 | 0.30 | 0.20 | 0.17 | 0.50 |
| fix_index_absolute_difference_same_position / random_matched | 0.00 | 0.20 | 0.23 | 0.26 | 0.23 | 0.19 | 0.17 | 0.53 |

### Recall of MLP-driven flips (centre and rim)

| setup | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| mean-ablate all (baseline) | 0.46 | 0.15 | 0.19 | 0.29 | 0.36 | 0.70 | 0.95 | 0.90 |
| original / real_acts | 0.46 | 0.18 | 0.24 | 0.37 | 0.47 | 0.80 | 0.98 | 0.95 |
| original / approx_acts | 0.46 | 0.16 | 0.20 | 0.25 | 0.31 | 0.74 | 0.99 | 0.81 |
| original / random_matched | 0.46 | 0.16 | 0.21 | 0.31 | 0.41 | 0.73 | 0.96 | 0.90 |
| fix_index / real_acts | 0.54 | 0.37 | 0.38 | 0.45 | 0.53 | 0.76 | 0.97 | 0.93 |
| fix_index / approx_acts | 0.51 | 0.32 | 0.30 | 0.35 | 0.42 | 0.70 | 0.97 | 0.89 |
| fix_index / random_matched | 0.46 | 0.16 | 0.21 | 0.30 | 0.39 | 0.71 | 0.96 | 0.90 |
| fix_index_select_on_difference / real_acts | 0.55 | 0.38 | 0.38 | 0.44 | 0.52 | 0.75 | 0.96 | 0.93 |
| fix_index_select_on_difference / approx_acts | 0.51 | 0.32 | 0.30 | 0.36 | 0.42 | 0.70 | 0.96 | 0.89 |
| fix_index_select_on_difference / random_matched | 0.46 | 0.16 | 0.20 | 0.29 | 0.37 | 0.70 | 0.95 | 0.90 |
| fix_index_difference_same_position / real_acts | 0.54 | 0.38 | 0.38 | 0.44 | 0.52 | 0.75 | 0.96 | 0.87 |
| fix_index_difference_same_position / approx_acts | 0.51 | 0.32 | 0.30 | 0.35 | 0.43 | 0.71 | 0.96 | 0.89 |
| fix_index_difference_same_position / random_matched | 0.46 | 0.16 | 0.20 | 0.29 | 0.37 | 0.70 | 0.95 | 0.90 |
| all_fixes_mid_probes / real_acts | 0.52 | 0.26 | 0.28 | 0.37 | 0.45 | 0.72 | 0.95 | 0.90 |
| all_fixes_mid_probes / approx_acts | 0.51 | 0.23 | 0.25 | 0.34 | 0.42 | 0.71 | 0.95 | 0.89 |
| all_fixes_mid_probes / random_matched | 0.46 | 0.15 | 0.19 | 0.29 | 0.37 | 0.70 | 0.95 | 0.90 |
| fix_index_absolute_difference_same_position / real_acts | 0.55 | 0.37 | 0.37 | 0.44 | 0.52 | 0.75 | 0.96 | 0.87 |
| fix_index_absolute_difference_same_position / approx_acts | 0.52 | 0.31 | 0.29 | 0.35 | 0.42 | 0.71 | 0.97 | 0.89 |
| fix_index_absolute_difference_same_position / random_matched | 0.46 | 0.16 | 0.20 | 0.30 | 0.37 | 0.70 | 0.95 | 0.90 |

### Specificity: tiles the MLP does not change stay unchanged, centre

| setup | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| mean-ablate all (baseline) | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| original / real_acts | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| original / approx_acts | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |
| original / random_matched | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index / real_acts | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index / approx_acts | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index / random_matched | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index_select_on_difference / real_acts | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index_select_on_difference / approx_acts | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index_select_on_difference / random_matched | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index_difference_same_position / real_acts | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index_difference_same_position / approx_acts | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index_difference_same_position / random_matched | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| all_fixes_mid_probes / real_acts | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| all_fixes_mid_probes / approx_acts | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| all_fixes_mid_probes / random_matched | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index_absolute_difference_same_position / real_acts | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index_absolute_difference_same_position / approx_acts | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fix_index_absolute_difference_same_position / random_matched | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Effect recovered on the flipped logit difference, centre

| setup | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| mean-ablate all (baseline) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| original / real_acts | 0.04 | 0.02 | 0.09 | 0.12 | 0.11 | 0.10 | 0.43 | 0.84 |
| original / approx_acts | -9.17 | -0.01 | -0.02 | -0.02 | -0.05 | -0.05 | -1.18 | -0.38 |
| original / random_matched | 0.03 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.09 | 0.00 |
| fix_index / real_acts | 0.01 | -0.04 | -0.00 | 0.02 | 0.02 | 0.02 | 0.12 | 0.83 |
| fix_index / approx_acts | -0.93 | -0.13 | -0.06 | -0.02 | -0.02 | -0.01 | -0.10 | -0.02 |
| fix_index / random_matched | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| fix_index_select_on_difference / real_acts | 0.01 | -0.06 | -0.04 | -0.00 | 0.01 | 0.01 | 0.05 | 0.83 |
| fix_index_select_on_difference / approx_acts | -0.95 | -0.14 | -0.08 | -0.03 | -0.02 | -0.01 | -0.08 | -0.02 |
| fix_index_select_on_difference / random_matched | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| fix_index_difference_same_position / real_acts | -0.00 | -0.06 | -0.03 | -0.00 | 0.01 | 0.01 | 0.02 | -0.11 |
| fix_index_difference_same_position / approx_acts | -0.13 | -0.14 | -0.07 | -0.03 | -0.02 | -0.01 | -0.05 | -0.01 |
| fix_index_difference_same_position / random_matched | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| all_fixes_mid_probes / real_acts | -0.01 | -0.01 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| all_fixes_mid_probes / approx_acts | -0.06 | -0.02 | -0.01 | -0.00 | 0.00 | -0.00 | -0.00 | -0.01 |
| all_fixes_mid_probes / random_matched | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fix_index_absolute_difference_same_position / real_acts | 0.01 | -0.06 | -0.04 | -0.01 | 0.02 | 0.01 | 0.07 | -0.11 |
| fix_index_absolute_difference_same_position / approx_acts | -0.15 | -0.14 | -0.08 | -0.03 | -0.01 | -0.00 | -0.04 | -0.01 |
| fix_index_absolute_difference_same_position / random_matched | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |

### Effect recovered on the flipped logit difference, rim

| setup | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| mean-ablate all (baseline) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| original / real_acts | 0.07 | 0.01 | 0.00 | 0.06 | 0.04 | 0.09 | 0.46 | 0.85 |
| original / approx_acts | -11.07 | -0.03 | -0.04 | -0.01 | -0.09 | -0.15 | -2.57 | -0.35 |
| original / random_matched | 0.02 | 0.00 | 0.01 | 0.01 | 0.02 | 0.03 | 0.11 | 0.02 |
| fix_index / real_acts | -0.13 | -0.23 | -0.12 | 0.03 | 0.03 | 0.02 | 0.10 | 0.83 |
| fix_index / approx_acts | -1.25 | -0.35 | -0.17 | -0.01 | -0.01 | -0.01 | -0.26 | -0.02 |
| fix_index / random_matched | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.03 | 0.02 |
| fix_index_select_on_difference / real_acts | -0.15 | -0.25 | -0.14 | 0.02 | 0.02 | 0.00 | 0.04 | 0.83 |
| fix_index_select_on_difference / approx_acts | -1.29 | -0.37 | -0.19 | -0.01 | 0.00 | -0.01 | -0.21 | -0.02 |
| fix_index_select_on_difference / random_matched | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| fix_index_difference_same_position / real_acts | -0.16 | -0.24 | -0.13 | 0.02 | 0.02 | 0.00 | -0.00 | 0.16 |
| fix_index_difference_same_position / approx_acts | -0.30 | -0.36 | -0.18 | -0.01 | 0.00 | -0.01 | -0.12 | -0.01 |
| fix_index_difference_same_position / random_matched | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| all_fixes_mid_probes / real_acts | -0.12 | -0.05 | -0.02 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 |
| all_fixes_mid_probes / approx_acts | -0.19 | -0.07 | -0.03 | 0.00 | 0.00 | -0.00 | -0.00 | -0.01 |
| all_fixes_mid_probes / random_matched | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fix_index_absolute_difference_same_position / real_acts | -0.14 | -0.24 | -0.13 | 0.02 | 0.03 | 0.01 | 0.08 | 0.16 |
| fix_index_absolute_difference_same_position / approx_acts | -0.31 | -0.36 | -0.18 | -0.01 | 0.01 | 0.00 | -0.10 | -0.01 |
| fix_index_absolute_difference_same_position / random_matched | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |

### Effect recovered on the flipped logit difference, only tiles where the MLP flips the decision, centre

| setup | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| mean-ablate all (baseline) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| original / real_acts | 0.14 | -0.02 | -0.04 | -0.02 | 0.06 | 0.27 | 0.55 | 1.00 |
| original / approx_acts | -2.49 | -0.02 | -0.01 | -0.00 | 0.00 | 0.04 | -0.07 | -0.00 |
| original / random_matched | 0.07 | 0.01 | 0.02 | 0.03 | 0.08 | 0.10 | 0.16 | 0.02 |
| fix_index / real_acts | 0.31 | 0.26 | 0.23 | 0.22 | 0.29 | 0.20 | 0.32 | 0.99 |
| fix_index / approx_acts | -0.09 | 0.12 | 0.07 | 0.07 | 0.09 | 0.03 | 0.07 | -0.00 |
| fix_index / random_matched | 0.02 | 0.01 | 0.02 | 0.02 | 0.04 | 0.03 | 0.08 | 0.02 |
| fix_index_select_on_difference / real_acts | 0.32 | 0.26 | 0.24 | 0.22 | 0.28 | 0.17 | 0.20 | 0.99 |
| fix_index_select_on_difference / approx_acts | -0.10 | 0.11 | 0.07 | 0.07 | 0.09 | 0.02 | 0.05 | -0.00 |
| fix_index_select_on_difference / random_matched | 0.03 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.04 | 0.00 |
| fix_index_difference_same_position / real_acts | 0.24 | 0.26 | 0.25 | 0.23 | 0.28 | 0.17 | 0.16 | 0.08 |
| fix_index_difference_same_position / approx_acts | 0.10 | 0.11 | 0.07 | 0.07 | 0.10 | 0.03 | 0.08 | -0.00 |
| fix_index_difference_same_position / random_matched | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 | 0.00 |
| all_fixes_mid_probes / real_acts | 0.15 | 0.13 | 0.12 | 0.12 | 0.16 | 0.08 | 0.01 | 0.00 |
| all_fixes_mid_probes / approx_acts | 0.10 | 0.08 | 0.06 | 0.06 | 0.09 | 0.03 | 0.00 | -0.00 |
| all_fixes_mid_probes / random_matched | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 |
| fix_index_absolute_difference_same_position / real_acts | 0.31 | 0.25 | 0.23 | 0.21 | 0.28 | 0.18 | 0.22 | 0.08 |
| fix_index_absolute_difference_same_position / approx_acts | 0.10 | 0.11 | 0.05 | 0.05 | 0.09 | 0.04 | 0.06 | -0.00 |
| fix_index_absolute_difference_same_position / random_matched | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.04 | 0.00 |

### Average neurons kept per position (thesis Table 7.2: 148 21 48 86 199 291 448 97)

| setup | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| original | 147.99 | 20.13 | 48.32 | 85.82 | 199.52 | 286.83 | 449.51 | 96.82 |
| fix_index | 53.28 | 37.47 | 67.79 | 115.14 | 194.79 | 262.03 | 349.35 | 101.53 |
| fix_index_select_on_difference | 58.33 | 32.33 | 39.45 | 43.63 | 62.33 | 97.94 | 159.46 | 61.37 |
| fix_index_difference_same_position | 27.23 | 30.33 | 37.42 | 41.42 | 55.42 | 65.12 | 82.98 | 25.62 |
| all_fixes_mid_probes | 26.32 | 25.79 | 33.98 | 40.34 | 62.44 | 88.00 | 101.68 | 21.21 |
| fix_index_absolute_difference_same_position | 28.80 | 32.47 | 42.47 | 49.07 | 77.54 | 101.51 | 144.50 | 70.43 |

### Classification: mean neurons per (rule, layer), and share of rules with no neuron

| setup | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| original mean | 22.86 | 7.97 | 33.79 | 80.40 | 164.34 | 261.76 | 288.11 | 89.12 |
| original share empty | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fix_index mean | 38.40 | 34.94 | 68.38 | 115.64 | 182.10 | 254.41 | 296.14 | 97.36 |
| fix_index share empty | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fix_index_select_on_difference mean | 41.47 | 33.71 | 48.04 | 56.61 | 79.81 | 122.47 | 132.41 | 66.43 |
| fix_index_select_on_difference share empty | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fix_index_difference_same_position mean | 24.10 | 31.56 | 44.08 | 53.11 | 71.69 | 100.67 | 53.32 | 32.28 |
| fix_index_difference_same_position share empty | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| all_fixes_mid_probes mean | 27.62 | 35.56 | 50.83 | 61.44 | 89.14 | 115.27 | 109.80 | 28.83 |
| all_fixes_mid_probes share empty | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fix_index_absolute_difference_same_position mean | 25.39 | 33.64 | 52.64 | 66.72 | 109.92 | 177.47 | 100.86 | 80.55 |
| fix_index_absolute_difference_same_position share empty | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

MLP-driven flip events per layer (centre + rim): 297396 100944 53471 32244 22083 35784 210109 34884
