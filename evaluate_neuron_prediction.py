from create_dataset import get_filtered_dataset
from create_dataset import create_big_dataset
import pandas as pd
import os

from neuron_predictor import NeuronPredictor
from descision_trees import DecisionTree#
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

train_size = 100000
eval_size = 10000
dataset_name_train = "logic_train"
dataset_name_eval = "logic_eval"

'''train_size = 1000
eval_size = 1000
dataset_name_train = "logic_train_test"
dataset_name_eval = "logic_eval_test"'''

def train_neuron_predictor(X, y, neuron_predictor : NeuronPredictor):
    """
    Train the neuron_predictor on the dataset
    """
    neuron_predictor.fit(X, y)

def evaluate_neuron_predictor(X, y, neuron_predictor : NeuronPredictor):
    """
    Return Sparcity, Accuracy, Precision, Recall, F1 on the dataset
    """
    sparcity = neuron_predictor.get_sparcity()
    y_pred = neuron_predictor.predict(X)
    # y_pred = y_pred[:, None]
    y_binary = y > 0
    y_pred_binary = y_pred > 0
    """accuracy = (y_binary == y_pred_binary).mean()
    precision = (y_binary & y_pred_binary).sum() / y_pred_binary.sum()
    recall = (y_binary & y_pred_binary).sum() / y_binary.sum()
    f1 = 2 * (precision * recall) / (precision + recall)"""
    accuracy = accuracy_score(y_binary, y_pred_binary)
    precision = precision_score(y_binary, y_pred_binary)
    recall = recall_score(y_binary, y_pred_binary)
    f1 = f1_score(y_binary, y_pred_binary)
    return sparcity, accuracy, precision, recall, f1

def get_in_and_output_from_dataset(big_dataset : pd.DataFrame, train_or_eval : str):
    """
    Return the input and output columns from the dataset
    """
    if train_or_eval == "train":
        dataset = get_filtered_dataset(big_dataset, layer, neuron, size = train_size, overfitting_strength=None)
    else:
        dataset = get_filtered_dataset(big_dataset, layer, neuron, size = eval_size, overfitting_strength=None)
    column_names = dataset.columns
    column_names_input = column_names[:-1]
    column_names_output = [column_names[-1]]
    X = dataset[column_names_input].astype(float)
    y = dataset[column_names_output].astype(float)
    # turn into numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y


if __name__ == "__main__":
    # Create a Dataframe containing all the evaluation results
    # The columns should be:  Sparcity, Accuracy, Precision, Recall, F1
    # And the rows should be L{layer}_N{neuron}
    # Save the Dataframe to a csv file

    # Create the results Dictonary, this will be used to create the Dataframe
    results = {
        "Layer": [],
        "Neuron": [],
        "Sparcity": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": []
    }

    datasets_path = "data/neuron_datasets/"
    eval_only = False
    for layer in range(1, 3):
        if not os.path.exists(datasets_path + f"{dataset_name_train}_L{layer}.csv"):
            create_big_dataset(layer, num_samples=train_size, dataset_name=dataset_name_train)
        if not os.path.exists(datasets_path + f"{dataset_name_eval}_L{layer}.csv"):
            create_big_dataset(layer, num_samples=eval_size, start=train_size, dataset_name=dataset_name_eval)
        if not eval_only:
            big_dataset_train = pd.read_csv(f"data/neuron_datasets/{dataset_name_train}_L{layer}.csv")
        big_dataset_eval = pd.read_csv(f"data/neuron_datasets/{dataset_name_eval}_L{layer}.csv")

        for neuron in tqdm(range(2048)):
            decision_tree = DecisionTree(layer, neuron)
            succes = decision_tree.load(layer, neuron)
            assert succes or not eval_only
            if not succes:
                X_train, y_train = get_in_and_output_from_dataset(big_dataset_train, "train")
                train_neuron_predictor(X_train, y_train, decision_tree)
                decision_tree.save()
            X_test, y_test = get_in_and_output_from_dataset(big_dataset_eval, "test")
            # y_test = y_test[:, 0]
            sparcity, accuracy, precision, recall, f1 = evaluate_neuron_predictor(X_test, y_test, decision_tree)
            results["Layer"].append(layer)
            results["Neuron"].append(neuron)
            results["Sparcity"].append(sparcity)
            results["Accuracy"].append(accuracy)
            results["Precision"].append(precision)
            results["Recall"].append(recall)
            results["F1"].append(f1)
            print(f"Layer {layer}, Neuron {neuron}, Sparcity {sparcity}, Accuracy {accuracy}, Precision {precision}, Recall {recall}, F1 {f1}")
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/evaluation_results.csv", index=False)
    print(results_df)
    print("Evaluation done")