# %%
from neuron_predictor import NeuronPredictor

# import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  
from sklearn.tree import DecisionTreeClassifier

import math

import numpy as np
from utils import probe_directions_list, tuple_to_label
import os

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from create_dataset import get_filtered_dataset
from create_dataset import get_variable_names

import pandas as pd

import joblib
from typing import List, Tuple

def tree_to_dnf(tree, feature_names=None):
    recurse_count = 0
    def recurse(node, path):
        nonlocal recurse_count
        recurse_count += 1
        if tree.feature[node] != -2:  # not a leaf
            feature = feature_names[tree.feature[node]] if feature_names is not None else f"feature_{tree.feature[node]}"
            threshold = tree.threshold[node]

            if feature[3:].startswith("not"): # The First Case probably never happens
                feature = feature[:3] + feature[7:]
                threshold = 1-threshold
            left_feature = (feature, "<=", threshold)
            right_feature = (feature, ">", threshold)
            
            left_path = path + [left_feature]
            right_path = path + [right_feature]
            
            yield from recurse(tree.children_left[node], left_path)
            yield from recurse(tree.children_right[node], right_path)
        else:  # leaf
            predicted_value = tree.value[node][0, -1]
            yield (path, predicted_value)

    rules = list(recurse(0, []))
    rules.sort(key=lambda x: x[-1], reverse=True) # sort by the predicted value
    return rules, recurse_count

def process_disjunction(disjunction: List[Tuple[str, str, float]]) -> str:
    # Step 1: Remove redundancies
    cleaned = remove_redundancies(disjunction)
    
    # Step 2: Remove certain "placed" variables
    result = remove_placed_variables(cleaned)
    
    # Step 3: Group literals by variable
    # grouped = group_literals(cleaned)
    
    # Step 4: Generate the final string
    # result = generate_string(grouped)
    
    return result

def remove_redundancies(disjunction: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
    result = []
    for var, op, threshold in disjunction:
        if op == "<=":
            existing = next((x for x in result if x[0] == var and x[1] == "<="), None)
            if existing is None or threshold < existing[2]:
                result = [x for x in result if not (x[0] == var and x[1] == "<=")]
                result.append((var, op, threshold))
        elif op == ">":
            existing = next((x for x in result if x[0] == var and x[1] == ">"), None)
            if existing is None or threshold > existing[2]:
                result = [x for x in result if not (x[0] == var and x[1] == ">")]
                result.append((var, op, threshold))
    return result

def remove_placed_variables(disjunction: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
    placed_vars = [var for var, op, _ in disjunction if "placed" in var and op == ">"]
    if len(placed_vars) > 0:
        return [x for x in disjunction if not ("placed" in x[0] and x[1] == "<=")]
    return disjunction

def group_literals(disjunction: List[Tuple[str, str, float]]) -> dict:
    grouped = {}
    for var, op, threshold in disjunction:
        if var not in grouped:
            grouped[var] = {">" : None, "<=" : None}
        grouped[var][op] = f"{threshold:.4f}"
    return grouped

def generate_string_from_rule(rule: List[Tuple[str, str, float]]) -> str:
    out_str_list = []
    for var, op, threshold in rule:
        if op == "<=":
            var = var[:3] + "not " + var[3:]
        out_str_list += [f"{var}"]
    return " AND ".join(out_str_list)

def generate_string(grouped: dict) -> str:
    parts = []
    for var, ops in grouped.items():
        if ops[">"] is not None and ops["<="] is not None:
            parts.append(f"{ops['>']} < {var} <= {ops['<=']}")
        elif ops[">"] is not None:
            parts.append(f"{var}")
        elif ops["<="] is not None:
            parts.append(f"not {var}")
    return " AND ".join(parts)

def get_accuracy(y_pred, y_test):
    correct = 0
    for i in range(len(y_test)):
        if (y_pred[i] > 0 and y_test[i] > 0) or (y_pred[i] <= 0 and y_test[i] <= 0):
            correct += 1
    return correct / len(y_test)

EPSILON = 0.000001

def scorefunction(depth, accuracy):
    return math.log(accuracy + EPSILON, 2) * 2**depth

class DecisionTree(NeuronPredictor):
    def __init__(self, layer, neuron):
        # This is bad coupling
        self.column_names_input = get_variable_names(True)
        self.neuron = neuron
        self.layer = layer

    def predict(self, X):
        return self.regressor.predict(X)

    def fit(self, X, y, **kwargs):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.regressor = DecisionTreeRegressor(random_state = 0, **kwargs)  
        self.regressor.fit(X, y)

    def get_top_rules(self, thresh = 0.9, column_names=None):
        if column_names is None:
            column_names = self.column_names_input
        cnf_rules, _ = tree_to_dnf(self.regressor.tree_, feature_names=column_names)
        rules = []
        preds = []
        cnf_rules = [(disjunction, pred_value) for (disjunction, pred_value) in cnf_rules if pred_value > thresh]
        for disjunction, pred_value in cnf_rules:
            disjunction = process_disjunction(disjunction)
            rules += [disjunction]
            preds += [pred_value]
        return rules, preds

    def get_clean_format(self, column_names=None):
        if column_names is None:
            column_names = self.column_names_input
        # cnf_rules = tree_to_cnf(self.regressor.tree_, feature_names=column_names)
        cnf_rules, variable_count = tree_to_dnf(self.regressor.tree_, feature_names=column_names)
        rules_nice = []
        for disjunction, pred_value in cnf_rules:
            disjunction = process_disjunction(disjunction)
            disjunction_str = generate_string_from_rule(disjunction)
            disjunction_str += f" => {pred_value:.4f}"
            rules_nice.append(disjunction_str)
        return rules_nice, variable_count

    def get_variable_count(self):
        _, variable_count = self.get_clean_format(self.column_names_input)
        return variable_count

    def load(self, layer, neuron):
        # Save the decision tree to a file
        file_path = f"neuron_predictors/decision_trees/decision_tree_L{layer}_N{neuron}.joblib"
        if os.path.exists(file_path):
            self.regressor = joblib.load(file_path)
            self.max_depth = self.regressor.get_depth()
            self.layer = layer
            self.neuron = neuron
            return True
        return False

    def save(self):
        file_path = f"neuron_predictors/decision_trees/decision_tree_L{self.layer}_N{self.neuron}.joblib"#
        # Save the decision tree to a file
        joblib.dump(self.regressor, file_path)


class ClassifierDecisionTree(NeuronPredictor):
    def __init__(self, layer, neuron):
        # This is bad coupling
        self.column_names_input = get_variable_names(True)
        self.neuron = neuron
        self.layer = layer

    def predict(self, X):
        return self.regressor.predict(X)

    def fit(self, X, y, **kwargs):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.regressor = DecisionTreeClassifier(**kwargs)  
        self.regressor.fit(X, y)

    def get_top_rules(self, thresh = 0.9, column_names=None):
        if column_names is None:
            column_names = self.column_names_input
        cnf_rules, _ = tree_to_dnf(self.regressor.tree_, feature_names=column_names)
        rules = []
        preds = []
        cnf_rules = [(disjunction, pred_value) for (disjunction, pred_value) in cnf_rules if pred_value > thresh]
        for disjunction, pred_value in cnf_rules:
            disjunction = process_disjunction(disjunction)
            rules += [disjunction]
            preds += [pred_value]
        return rules, preds

    def get_clean_format(self, column_names=None):
        if column_names is None:
            column_names = self.column_names_input
        # cnf_rules = tree_to_cnf(self.regressor.tree_, feature_names=column_names)
        cnf_rules, variable_count = tree_to_dnf(self.regressor.tree_, feature_names=column_names)
        rules_nice = []
        for disjunction, pred_value in cnf_rules:
            disjunction = process_disjunction(disjunction)
            disjunction_str = generate_string_from_rule(disjunction)
            disjunction_str += f" => {pred_value:.4f}"
            rules_nice.append(disjunction_str)
        return rules_nice, variable_count

    def get_variable_count(self):
        _, variable_count = self.get_clean_format(self.column_names_input)
        return variable_count

    def load(self, layer, neuron, name=None):
        # Save the decision tree to a file
        if name is not None:
            file_path = f"neuron_predictors/decision_tree_classifiers/{name}.joblib"
        else:
            file_path = f"neuron_predictors/decision_tree_classifiers/decision_tree_classifier_L{self.layer}_N{self.neuron}.joblib"
        if os.path.exists(file_path):
            self.regressor = joblib.load(file_path)
            # self.max_depth = self.regressor.get_depth()
            self.layer = layer
            self.neuron = neuron
            return True
        return False

    def save(self, name=None):
        if name is not None:
            file_path = f"neuron_predictors/decision_tree_classifiers/{name}.joblib"
        else:
            file_path = f"neuron_predictors/decision_tree_classifiers/decision_tree_classifier_L{self.layer}_N{self.neuron}.joblib"
        # Save the decision tree to a file
        joblib.dump(self.regressor, file_path)

def very_new_weighted_f1_score(y, y_pred):
    y = np.asarray(y)
    y = y[:, 0]
    y_pred = np.asarray(y_pred)

    sum_y = np.sum(y[y>0])
    len_y = len(y[y>0])
    
    # Score for each element
    score = y / sum_y * len_y
    
    # True positives (tp): y and y_pred are both positive
    tp = np.sum(score[(y_pred > 0) & (y > 0)])
    
    # True negatives (tn): both y and y_pred are non-positive
    # tn = np.sum((y_pred <= 0) & (y <= 0))
    
    # False positives (fp): y_pred is positive but y is non-positive
    fp = np.sum((y_pred > 0) & (y <= 0))
    
    # False negatives (fn): y_pred is non-positive but y is positive
    fn = np.sum(score[(y_pred <= 0) & (y > 0)])
    
    # Precision: tp / (tp + fp)
    precision = tp / (tp + fp + EPSILON)
    
    # Recall: tp / (tp + fn)
    recall = tp / (tp + fn + EPSILON)
    
    # F1 Score: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall + EPSILON)
    
    return f1

def get_data(layer, neuron, dataset : pd.DataFrame | str = None, **kwargs):
    if type(dataset) == str:
        small_dataset_softmax = pd.read_csv(dataset)
    elif type(dataset) == pd.DataFrame:
        small_dataset_softmax = dataset
    else:
        small_dataset_softmax = pd.read_csv(f"data/neuron_datasets/logic_small_L{layer}.csv")
    dataset = get_filtered_dataset(small_dataset_softmax, layer, neuron, overfitting_strength=None, **kwargs)
    dataset_columns_new = dataset.columns.tolist()

    dataset_columns_new = [col_name for col_name in dataset_columns_new if col_name[3:6] != "not"]
    dataset = dataset[dataset_columns_new]

    column_names = dataset.columns
    column_names_input = column_names[:-1]
    column_names_output = [column_names[-1]]

    X = dataset[column_names_input].astype(float)
    y = dataset[column_names_output].astype(float)
    # turn into numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]
    return X_train, X_test, y_train, y_test, column_names_input

def train_decision_tree(X_train, y_train, layer, neuron, **kwargs) -> DecisionTree:
    descision_tree = DecisionTree(layer, neuron)
    descision_tree.fit(X_train, y_train, **kwargs)
    return descision_tree

def evaluate_decision_tree(descision_tree : DecisionTree, X_test, y_test):
    column_names_input = descision_tree.column_names_input
    y_pred_test = descision_tree.predict(X_test)
    weighted_f1 = very_new_weighted_f1_score(y_test, y_pred_test)
    f1 = f1_score(y_test > 0, y_pred_test > 0)
    rules, variable_count = descision_tree.get_clean_format(column_names_input)
    return weighted_f1, f1, variable_count, rules, y_pred_test

def train_and_evaluate_decision_tree(layer : int, neuron : int, dataset : pd.DataFrame | str = None, **kwargs):
    X_train, X_test, y_train, y_test, column_names_input = get_data(layer, neuron, dataset)
    descision_tree = train_decision_tree(X_train, y_train, layer, neuron, **kwargs)
    descision_tree.column_names_input = column_names_input
    weighted_f1, f1, variable_count, rules, _ = evaluate_decision_tree(descision_tree, X_test, y_test)
    return descision_tree, weighted_f1, f1, variable_count, rules

# %%
# from utils import *
from create_dataset import get_filtered_dataset
# from descision_trees import DecisionTree
from create_dataset import save_filtered_dataset_for_neurons
# import abstractmethod
from abc import ABC, abstractmethod
from dataclasses import dataclass

# %%
@dataclass
class NeuronPredictorArgs:
    dataset_kind : str
    neuron_predictor_type : str
    preprocess_train_type : str
    preprocess_eval_type : str
    layer : int
    neuron : int
    len_data : int
    dataset_train : pd.DataFrame = None
    dataset_test : pd.DataFrame = None

class PreprocessAbc(ABC):
    @abstractmethod
    def preprocess(self, dataset : pd.DataFrame, args : NeuronPredictorArgs) -> pd.DataFrame:
        pass

def my_round(x):
    if x < 0:
        return -0.1
    if x < 0.3:
        return 0.1
    if x < 1:
        return 0.7
    return 1.5

class PreproccessorTrain(PreprocessAbc):
    def __init__(self):
        pass

    def preprocess(self, dataset : pd.DataFrame, args : NeuronPredictorArgs) -> pd.DataFrame:
        input_columns = dataset.columns[:-1]
        input_columns = [col for col in input_columns if col[3:] == "flipped" or col[3:] == "placed"]
        dataset[input_columns] = dataset[input_columns].apply(lambda x: x.apply(lambda y : 0 if y < 0.5 else 1))
        dataset["neuron activation"] = dataset["neuron activation"].apply(lambda x: my_round(x))
        return dataset
    
class PreproccessorTrainRound(PreprocessAbc):
    def __init__(self):
        pass

    def preprocess(self, dataset : pd.DataFrame, args : NeuronPredictorArgs) -> pd.DataFrame:
        dataset["neuron activation"] = dataset["neuron activation"].apply(lambda x: -1 if x <= 0 else 1)
        return dataset

class PreproccessorEval(PreprocessAbc):
    def __init__(self):
        pass

    def preprocess(self, dataset : pd.DataFrame, args : NeuronPredictorArgs) -> pd.DataFrame:
        return dataset

def get_neuron_predictor(args : NeuronPredictorArgs, **kwargs) -> NeuronPredictor:
    if args.neuron_predictor_type == "decision_tree":
        return DecisionTree(args.layer, args.neuron, **kwargs)
    elif args.neuron_predictor_type == "decision_tree_classifier":
        return ClassifierDecisionTree(args.layer, args.neuron, **kwargs)
    else:
        raise ValueError(f"neuron_predictor_type {args.neuron_predictor_type} not recognized")
    
def get_preprocess_train(args : NeuronPredictorArgs, **kwargs) -> PreprocessAbc:
    if args.preprocess_train_type == "normal":
        return PreproccessorTrain(**kwargs)
    elif args.preprocess_train_type == "round":
        return PreproccessorTrainRound(**kwargs)
    else:
        raise ValueError(f"preprocess_train_type {args.preprocess_train_type} not recognized")

def get_preprocess_eval(args : NeuronPredictorArgs, **kwargs) -> PreprocessAbc:
    if args.preprocess_eval_type == "normal":
        return PreproccessorEval(**kwargs)
    else:
        raise ValueError(f"preprocess_eval_type {args.preprocess_eval_type} not recognized")

def get_dataset(dataset_kind : str, train_or_eval : str, layer : int):
    assert train_or_eval in ["train", "eval"]
    dataset = pd.read_csv(f"data/neuron_datasets/{dataset_kind}_{train_or_eval}_L{layer}.csv")
    return dataset

def dataset_to_X_y(dataset : pd.DataFrame):
    column_names = dataset.columns
    column_names_input = column_names[:-1]
    column_names_output = [column_names[-1]]

    X = dataset[column_names_input].astype(float)
    y = dataset[column_names_output].astype(float)
    # turn into numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y

def evaluate_neuron_predictor(neuron_predictor : NeuronPredictor, X_test, y_test):
    y_pred_test = neuron_predictor.predict(X_test)
    weighted_f1 = very_new_weighted_f1_score(y_test, y_pred_test)
    f1 = f1_score(y_test > 0, y_pred_test > 0)
    rules, variable_count = neuron_predictor.get_clean_format()
    return weighted_f1, f1, variable_count, rules, y_pred_test

# This class get's a train / eval dataset AND/OR a layer and neuron
# AND a neuron predictor and 
class NeuronPredictorEvaluator:
    def get_dataset_train_filtered(self):
        # dataset_train = get_dataset(self.args.dataset_kind, "train", self.args.layer)
        dataset_train_filtered = get_filtered_dataset(self.dataset_train, self.args.layer, self.args.neuron, size=self.args.len_data, remove_negative_features=True)
        dataset_train_filtered = self.preprocessor_train.preprocess(dataset_train_filtered, self.args)
        return dataset_train_filtered

    def get_dataset_test_filtered(self):
        # dataset_test = get_dataset(self.args.dataset_kind, "eval", self.args.layer)
        dataset_test_filtered = get_filtered_dataset(self.dataset_test, self.args.layer, self.args.neuron, size=self.args.len_data)
        dataset_test_filtered = self.preprocessor_eval.preprocess(dataset_test_filtered, self.args)
        return dataset_test_filtered

    def __init__(
        self,
        args : NeuronPredictorArgs,
        **kwargs
    ):
        # initialize the neuron predictor
        # TODO: when initialized: big dataset is loaded, you can also pass big dataset as an argument
        if args.dataset_train is not None:
            self.dataset_train = args.dataset_train
        else:
            self.dataset_train = get_dataset(self.args.dataset_kind, "train", self.args.layer)
        if args.dataset_test is not None:
            self.dataset_test = args.dataset_test
        else:
            self.dataset_test = get_dataset(self.args.dataset_kind, "test", self.args.layer)

        self.neuron_predictor : NeuronPredictor = get_neuron_predictor(args, **kwargs)
        self.preprocessor_train : PreprocessAbc = get_preprocess_train(args)
        self.preprocessor_eval : PreprocessAbc = get_preprocess_eval(args)
        self.args = args

    def evaluate(self):
        dataset_train = self.get_dataset_train_filtered()
        X_train, y_train = dataset_to_X_y(dataset_train)
        weighted_f1_train, f1_train, _, _, _ = evaluate_neuron_predictor(self.neuron_predictor, X_train, y_train)	
        dataset_test = self.get_dataset_test_filtered()
        X_test, y_test = dataset_to_X_y(dataset_test)
        weighted_f1, f1, variable_count, rules, _ = evaluate_neuron_predictor(self.neuron_predictor, X_test, y_test)
        return weighted_f1, weighted_f1_train, f1, f1_train, variable_count, rules

    def train_and_evaluate(self, **kwargs):
        dataset_train = self.get_dataset_train_filtered()
        X_train, y_train = dataset_to_X_y(dataset_train)
        self.neuron_predictor.fit(X_train, y_train, **kwargs)
        weighted_f1_train, f1_train, _, _, _ = evaluate_neuron_predictor(self.neuron_predictor, X_train, y_train)	
        # TODO: save the neuron predictor
        # evaluate the neuron predictor
        # TODO: load the neuron predictor if it exists
        dataset_test = self.get_dataset_test_filtered()
        X_test, y_test = dataset_to_X_y(dataset_test)
        weighted_f1, f1, variable_count, rules, _ = evaluate_neuron_predictor(self.neuron_predictor, X_test, y_test)
        return weighted_f1, weighted_f1_train, f1, f1_train, variable_count, rules

# %%
from utils import label_to_tuple
import torch as t
from utils import plot_boards_general

# %%
def visualize_rules(rules, preds, how_many = 10):
    # generate 10 random indices
    num_rules = min(how_many, len(rules))
    indices = np.random.choice(len(rules), num_rules, replace=False)
    # sort indices
    indices.sort()
    # num_options = 11 + 3
    num_options = 7
    boards = t.zeros((num_options, num_rules, 8, 8))
    # boards = t.zeros((how_many, 7, 8, 8))
    '''feature_indices = {
        "empty" : 0,
        "not empty" : 1,
        "yours" : 2,
        "not yours" : 3,
        "mine" : 4,
        "not mine" : 5,
        "flipped" : 3 + 3,
        "not flipped" : 4 + 3,
        "placed" : 5 + 3,
        "not placed" : 6 + 3,
        "legal" : 7 + 3,
        "not legal" : 8 + 3,
        "accessible" : 9 + 3,
        "not accessible" : 10 + 3
    }'''
    feature_indices = {
        "empty" : 0,
        "yours" : 1,
        "mine" : 2,
        "flipped" : 3,
        "placed" : 4,
        "legal" : 5,
        "accessible" : 6,
    }
    for i in indices:
        rule = rules[i]
        pred = preds[i]
        if pred < 0.5:
            continue
        for j in range(len(rule)):
            literal = rule[j]
            feature, op, threshold = literal
            # TODO: I need to change not A4 Placed to A4 not_placed
            label = feature[:2]
            tile_tuple = label_to_tuple(label)
            option_str = feature[3:]
            option_str = option_str.lower()
            if op == "<=":
                # option_str = "not " + option_str
                val = -1
            else:
                val = 1
            option = feature_indices[option_str]
            boards[option, i, tile_tuple[0], tile_tuple[1]] = val
    plot_boards_general(x_labels=list(feature_indices.keys()), y_labels=[f"Rule {i}" for i in indices], boards=boards)

# %%
import pandas as pd
from tqdm import tqdm
import torch as t
import plotly.express as px
import plotly.graph_objects as go
import sys

if __name__ == "__main__":
    script_name = sys.argv[0]
    if len(sys.argv) < 2:
        print(f"Usage: {script_name} <dataset>")
        sys.exit(1)
    layer = int(sys.argv[1])
    # %%
    # load datasets
    df = pd.read_csv(f"data/neuron_datasets/big_argmax_train_L{layer}.csv")

    # %%
    # load datasets
    df_test = pd.read_csv(f"data/neuron_datasets/big_argmax_test_L{layer}.csv")

    # y_pred_save = t.zeros((8, 2048, 2, 50000))
    results_dict = {
        "layer" : [],
        "neuron" : [],
        "min_impurity_decrease" : [],
        "f1" : [],
        "f1_train" : [],
        "weighted_f1" : [],
        "weighted_f1_train" : [],
        "variable_count" : [],
        "rules" : [],
    }
    min_impurity_decreases = [0.0001, 0.0005]
    neurons = list(range(2048))
    for i, neuron in enumerate(neurons):
        args = NeuronPredictorArgs(
            dataset_kind = "big_argmax",
            neuron_predictor_type = "decision_tree_classifier",
            preprocess_train_type = "round",
            preprocess_eval_type = "normal",
            layer = layer,
            neuron = neuron,
            len_data = 100000,
            dataset_train = df,
            dataset_test = df_test,
        )
        evaluator_round = NeuronPredictorEvaluator(args)
        # weighted_f1, f1, variable_count, rules, y_pred_test = evaluator_round.train_and_evaluate(min_impurity_decrease=0.00001)
        for j, min_impurity_decrease in enumerate(min_impurity_decreases):
            weighted_f1, weighted_f1_train, f1, f1_train, variable_count, rules = evaluator_round.train_and_evaluate(min_impurity_decrease=min_impurity_decrease)
            results_dict["layer"].append(layer)
            results_dict["neuron"].append(neuron)
            results_dict["min_impurity_decrease"].append(min_impurity_decrease)
            results_dict["f1"].append(f1)
            results_dict["f1_train"].append(f1_train)
            results_dict["weighted_f1"].append(weighted_f1)
            results_dict["weighted_f1_train"].append(weighted_f1_train)
            results_dict["variable_count"].append(variable_count)
            results_dict["rules"].append(rules)
            # y_pred_save[layer, neuron, j] = t.tensor(y_pred_test)
            # TODO: Save decision tree
            try:
                evaluator_round.neuron_predictor.save(name=f"neuron_classifier_L{layer}_N{neuron}_{min_impurity_decrease}")
            except:
                print(f"Failed to save neuron_classifier_L{layer}_N{neuron}_{min_impurity_decrease}")
            print(f"L{layer}_N{neuron}_impurity_{min_impurity_decrease:.4f}: f1: {f1:.4f}, f1_train: {f1_train:.4f}, weighted_f1: {weighted_f1:.4f}, weighted_f1_train: {weighted_f1_train:.4f}, variable_count: {variable_count}")

    # Save results dict as csv
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f"results_decision_tree_classifier_L{layer}.csv")
    # save y_pred_save
    # t.save(y_pred_save, f"y_pred_decision_tree_classifier_L{layer}.pt")