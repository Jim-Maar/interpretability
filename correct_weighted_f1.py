# %% [markdown]
# # Evaluate Results

# %% [markdown]
# ## Correct Weighted f1

# %%
from decision_trees_run import ClassifierDecisionTree
from decision_trees_run import NeuronPredictorArgs
from decision_trees_run import NeuronPredictor
from decision_trees_run import NeuronPredictorEvaluator

from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys

if __name__ == "__main__":
    # script_name = sys.argv[0]
    # layer = int(sys.argv[1])
    layer = 2
    bis_dataset_train = pd.read_csv(f"data/neuron_datasets/big_argmax_train_L{layer}.csv")
    big_dataset_test = pd.read_csv(f"data/neuron_datasets/big_argmax_test_L{layer}.csv")
    df = pd.read_csv(f"results_decision_tree_classifier_L{layer}.csv")
    min_impurity_decreases = [0.0001, 0.0005]
    df = pd.read_csv(f"results_decision_tree_classifier_L{layer}.csv")
    # remove column Unnamed: 0
    df = df.drop(columns=["Unnamed: 0"])
    for neuron in range(2048):
        print(f"L{layer}_N{neuron}")
        args = NeuronPredictorArgs(
            dataset_kind = "big_argmax",
            neuron_predictor_type = "decision_tree_classifier",
            preprocess_train_type = "round",
            preprocess_eval_type = "normal",
            layer = layer,
            neuron = neuron,
            len_data = 100000,
            dataset_train = bis_dataset_train,
            dataset_test = big_dataset_test,
        )
        evaluator_round = NeuronPredictorEvaluator(args)
        # weighted_f1, f1, variable_count, rules, y_pred_test = evaluator_round.train_and_evaluate(min_impurity_decrease=0.00001)
        for j, min_impurity_decrease in enumerate(min_impurity_decreases):
            dt = ClassifierDecisionTree(layer, neuron)
            dt.load(layer, neuron, name = f"neuron_classifier_L{layer}_N{neuron}_{min_impurity_decrease}")
            evaluator_round.neuron_predictor = dt
            weighted_f1, weighted_f1_train, f1, f1_train, variable_count, rules = evaluator_round.evaluate()
            correct_row = (df["neuron"] == neuron) & (df["min_impurity_decrease"] == min_impurity_decrease)
            df.loc[correct_row, "weighted_f1"] = weighted_f1
            df.loc[correct_row, "weighted_f1_train"] = weighted_f1_train
            # df.loc[correct_row, "rules"] = rules
    # save df with new path
    df.to_csv(f"results_decision_tree_classifier_L{layer}_correct_weighted_f1.csv")
    print(f"Saved results_decision_tree_classifier_L{layer}_correct_weighted_f1.csv")