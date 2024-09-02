from neuron_predictor import NeuronPredictor

# import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  
import math

import numpy as np
from utils import probe_directions_list, tuple_to_label
import os

from sklearn.metrics import f1_score

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
    def recurse(node, path):
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

    return list(recurse(0, []))

def get_accuracy(y_pred, y_test):
    correct = 0
    for i in range(len(y_test)):
        if (y_pred[i] > 0 and y_test[i] > 0) or (y_pred[i] <= 0 and y_test[i] <= 0):
            correct += 1
    return correct / len(y_test)

def get_f1(y_pred, y_test):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_test)):
        if y_pred[i] > 0 and y_test[i] > 0:
            tp += 1
        elif y_pred[i] <= 0 and y_test[i] <= 0:
            tn += 1
        elif y_pred[i] > 0 and y_test[i] <= 0:
            fp += 1
        elif y_pred[i] <= 0 and y_test[i] > 0:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

EPSILON = 0.000001

def scorefunction(depth, accuracy):
    return math.log(accuracy + EPSILON, 2) * 2**depth

class DecisionTree(NeuronPredictor):
    def __init__(self, layer, neuron):
        self.column_names_input = get_variable_names()
        self.neuron = neuron
        self.layer = layer

    def predict(self, X):
        return self.regressor.predict(X)

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        top_score = -math.inf
        self.regressor = DecisionTreeRegressor(random_state = 0, max_depth=3)  
        self.regressor.fit(X_train, y_train)
        self.max_depth = 3
        # I think high max_depth will be the main limmiting time factor
        """for max_depth in range(1, 7):
            regressor = DecisionTreeRegressor(random_state = 0, max_depth=max_depth)  
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            # Get the f1 score using a library function
            f1 = f1_score(y_test > 0, y_pred > 0)
            # f1 = get_f1(y_pred, y_test)
            score = scorefunction(max_depth, f1)
            if score > top_score:
                top_score = score
                self.regressor = regressor
                self.max_depth = max_depth
                if f1 >= 0.95:
                    break"""

    def get_clean_format(self):
        cnf_rules = tree_to_cnf(self.regressor.tree_, feature_names=self.column_names_input)
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