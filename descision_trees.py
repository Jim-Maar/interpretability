from neuron_predictor import NeuronPredictor

# import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  
import math

import numpy as np
from utils import probe_directions_list, tuple_to_label
import os

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from create_dataset import get_filtered_dataset

import pandas as pd

import joblib

def get_variable_names():
    variable_names = []
    for probe_name in ["linear", "flipped", "placed"]:
        for row in range(8):
            for col in range(8):
                for option in probe_directions_list[probe_name]:
                    label = tuple_to_label((row, col))
                    variable_names.append(f"{label} {option}")
    return variable_names

def tree_to_cnf(tree, feature_names=None):
    recurse_count = 0
    def recurse(node, path):
        nonlocal recurse_count
        recurse_count += 1
        if tree.feature[node] != -2:  # not a leaf
            feature = feature_names[tree.feature[node]] if feature_names is not None else f"feature_{tree.feature[node]}"

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

    return list(recurse(0, [])), recurse_count

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
        #TODO: This is a bad coupling
        self.column_names_input = get_variable_names()
        self.neuron = neuron
        self.layer = layer

    def predict(self, X):
        return self.regressor.predict(X)

    def fit(self, X, y, **kwargs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.regressor = DecisionTreeRegressor(random_state = 0, **kwargs)  
        self.regressor.fit(X_train, y_train)

    def get_clean_format(self, column_names):
        if column_names is None:
            column_names = self.column_names_input
        cnf_rules = tree_to_cnf(self.regressor.tree_, feature_names=column_names)
        return cnf_rules

    def get_sparcity(self):
        return 1 / (self.max_depth * 2**self.max_depth)

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


def new_weighted_f1_score(y, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y)):
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