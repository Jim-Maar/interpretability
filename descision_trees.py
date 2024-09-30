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

'''def tree_to_cnf(tree, feature_names=None):
    recurse_count = 0
    def recurse(node, path):
        nonlocal recurse_count
        recurse_count += 1
        if tree.feature[node] != -2:  # not a leaf
            feature = feature_names[tree.feature[node]] if feature_names is not None else f"feature_{tree.feature[node]}"
            threshold = tree.threshold[node]

            if feature[3:].startswith("not"):
                left_feature = feature[:3] + feature[7:]
                right_feature = feature
            else:
                left_feature = feature[:3] + "not " + feature[3:]
                right_feature = feature
            
            left_path = path + [f"{left_feature}"]
            right_path = path + [f"{right_feature}"]
            
            yield from recurse(tree.children_left[node], left_path)
            yield from recurse(tree.children_right[node], right_path)
        else:  # leaf
            predicted_value = tree.value[node][0, 0]
            yield f"({' AND '.join(path)} => {predicted_value:.4f})"

    return list(recurse(0, [])), recurse_count'''

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
    out_str = ""
    for var, op, threshold in rule:
        out_str += f"{var} {op} {threshold:.4f} "
    return out_str

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
            disjunction_str += f"=> {pred_value:.4f}"
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
            disjunction_str += f"=> {pred_value:.4f}"
            rules_nice.append(disjunction_str)
        return rules_nice, variable_count

    def get_variable_count(self):
        _, variable_count = self.get_clean_format(self.column_names_input)
        return variable_count

    def load(self, layer, neuron):
        # Save the decision tree to a file
        file_path = f"neuron_predictors/decision_tree_classifiers/decision_tree_classifier_L{layer}_N{neuron}.joblib"
        if os.path.exists(file_path):
            self.regressor = joblib.load(file_path)
            self.max_depth = self.regressor.get_depth()
            self.layer = layer
            self.neuron = neuron
            return True
        return False

    def save(self):
        file_path = f"neuron_predictors/decision_tree_classifiers/decision_tree_classifier_L{self.layer}_N{self.neuron}.joblib"#
        # Save the decision tree to a file
        joblib.dump(self.regressor, file_path)

def new_weighted_f1_score(y, y_pred, only_y=True):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y)):
        if only_y:
            score = y[i]
        else:
            score = max(y[i], y_pred[i])
        if y_pred[i] > 0 and y[i] > 0:
            tp += score
        elif y_pred[i] <= 0 and y[i] <= 0:
            tn += 1
        elif y_pred[i] > 0 and y[i] <= 0:
            fp += score
        elif y_pred[i] <= 0 and y[i] > 0:
            fn += score
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1 = 2 * (precision * recall) / (precision + recall + EPSILON)
    return f1.item()

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
    return X_train, X_test, y_train, y_test, column_names_input

def train_decision_tree(X_train, y_train, layer, neuron, **kwargs) -> DecisionTree:
    descision_tree = DecisionTree(layer, neuron)
    descision_tree.fit(X_train, y_train, **kwargs)
    return descision_tree

def evaluate_decision_tree(descision_tree : DecisionTree, X_test, y_test):
    column_names_input = descision_tree.column_names_input
    y_pred_test = descision_tree.predict(X_test)
    weighted_f1 = new_weighted_f1_score(y_test, y_pred_test)
    f1 = f1_score(y_test > 0, y_pred_test > 0)
    rules, variable_count = descision_tree.get_clean_format(column_names_input)
    return weighted_f1, f1, variable_count, rules, y_pred_test

def train_and_evaluate_decision_tree(layer : int, neuron : int, dataset : pd.DataFrame | str = None, **kwargs):
    X_train, X_test, y_train, y_test, column_names_input = get_data(layer, neuron, dataset)
    descision_tree = train_decision_tree(X_train, y_train, layer, neuron, **kwargs)
    descision_tree.column_names_input = column_names_input
    weighted_f1, f1, variable_count, rules, _ = evaluate_decision_tree(descision_tree, X_test, y_test)
    return descision_tree, weighted_f1, f1, variable_count, rules