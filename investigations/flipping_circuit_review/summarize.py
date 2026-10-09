"""Prints the tables of the report from results/results.json. Run: `uv run summarize.py`"""

import json

from experiment import RESULTS_DIR

LAYERS = range(8)
VARIANTS = ["real_acts", "approx_acts", "random_matched"]


def row(name: str, values) -> str:
    return f"| {name} | " + " | ".join("—" if v != v else f"{v:.2f}" for v in values) + " |"


def header(first: str) -> str:
    names = " | ".join(f"L{layer}" for layer in LAYERS)
    separator = "|---" * (len(LAYERS) + 1) + "|"
    return f"| {first} | {names} |\n{separator}"


def combined_recall(scores: dict) -> list[float]:
    recall, count = scores["mlp_flip_recall"], scores["mlp_flip_count"]
    return [
        (recall[l][0] * count[l][0] + recall[l][1] * count[l][1]) / (count[l][0] + count[l][1])
        for l in LAYERS
    ]


def print_table(title: str, results: dict, metric):
    print(f"\n### {title}\n")
    print(header("setup"))
    scores = results["scores"]
    print(row("mean-ablate all (baseline)", metric(scores["baseline_mean_ablate_all"])))
    for config in results["configs"]:
        for variant in VARIANTS:
            print(row(f"{config['name']} / {variant}", metric(scores[f"{config['name']}/{variant}"])))


def main():
    results = json.loads((RESULTS_DIR / "results.json").read_text())
    scores = results["scores"]

    print("### Original metric, overall (thesis Table 7.1: circuit 15.4, approx 13.2, baseline 14.2)\n")
    print("| setup | accuracy |\n|---|---|")
    for name, score in scores.items():
        print(f"| {name} | {100 * score['original_final_accuracy_overall']:.1f} |")

    print_table("Original metric per layer, centre tiles", results,
                lambda s: [s["original_final_accuracy"][l][0] for l in LAYERS])
    print_table("Recall of MLP-driven flips (centre and rim)", results, combined_recall)
    print_table("Specificity: tiles the MLP does not change stay unchanged, centre", results,
                lambda s: [s["no_flip_specificity"][l][0] for l in LAYERS])
    print_table("Effect recovered on the flipped logit difference, centre", results,
                lambda s: [s["effect_recovered"][l][0] for l in LAYERS])
    print_table("Effect recovered on the flipped logit difference, rim", results,
                lambda s: [s["effect_recovered"][l][1] for l in LAYERS])
    print_table("Effect recovered on the flipped logit difference, only tiles where the MLP flips the decision, centre",
                results, lambda s: [s["effect_recovered_on_mlp_flips"][l][0] for l in LAYERS])

    print("\n### Average neurons kept per position (thesis Table 7.2: 148 21 48 86 199 291 448 97)\n")
    print(header("setup"))
    for config in results["configs"]:
        print(row(config["name"], scores[f"{config['name']}/real_acts"]["avg_neurons_kept"]))

    print("\n### Classification: mean neurons per (rule, layer), and share of rules with no neuron\n")
    print(header("setup"))
    for name, summary in results["classification"].items():
        print(row(f"{name} mean", summary["mean_neurons_per_rule"]))
        print(row(f"{name} share empty", summary["fraction_rules_with_no_neurons"]))

    counts = scores["baseline_mean_ablate_all"]["mlp_flip_count"]
    print("\nMLP-driven flip events per layer (centre + rim): " + " ".join(f"{int(c[0] + c[1])}" for c in counts))


if __name__ == "__main__":
    main()
