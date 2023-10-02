"""
Python module that contains tensorflow helper functions
"""
import matplotlib.pyplot as plt


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    plot training data, test data and predictions of a model that predict 1 output based on 1 input
    """
    plt.figure(figsize=(10, 7))
    # plot training data
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # plot test data
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # plot prediction
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # show legend
    plt.legend()
