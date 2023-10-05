"""
Python module that contains tensorflow helper functions
"""
import datetime
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf


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


def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    This function has been adapted from two phenomenal resources:
     1. CS231n - https://cs231n.github.io/neural-networks-case-study/
     2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

    # Create X values (we're going to predict on all of these)
    x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

    # Make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi-class
    if model.output_shape[-1] > 1: # checks the final dimension of the model's output shape, if this is > (greater than) 1, it's multi-class
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification...")
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

      If classes is passed, confusion matrix will be labelled, if not, integer class values
      will be used.

      Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).

      Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.

      Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                              y_pred=y_preds, # predicted labels
                              classes=class_names, # array of class label names
                              figsize=(15, 15),
                              text_size=10)
      """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.matshow.html
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.xaxis.label.set_size(text_size)
    ax.yaxis.label.set_size(text_size)
    ax.title.set_size(text_size + 5)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)


def check_images_folder(folder_path):
    """
    print out the number of subdirectories and filenames of a folder
    :param folder_path: path of the folder in string format
    :return: nothing
    """
    for dirpath, dirnames, filenames in os.walk(folder_path):
        if len(filenames) > 0:
            ext_set = {os.path.splitext(file)[-1] for file in filenames}
            ext_string = f" {ext_set}"
        else:
            ext_string = ""

        print(f"There are {len(dirnames)} directories and {len(filenames)} files{ext_string} in '{dirpath}'.")


def plot_loss_and_metrics_curves(history_dict, metrics=None):
    """
    Plot loss curves and metrics curves in separate graph
    :param history_dict: dictionary containing loss and metrics keys & data
    :param metrics: a string of a single metrics or list of multiple metrics to plot
    :return: nothing
    """
    epochs = range(len(history_dict.get("loss")))

    loss = history_dict.get("loss")
    val_loss = history_dict.get("val_loss")

    # plot loss
    plt.figure()
    plt.plot(epochs, loss, label="training_loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="validation_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    # plot metrics
    if isinstance(metrics, str):
        metrics = [metrics]

    if isinstance(metrics, list):
        for metric in metrics:
            metric_values = history_dict.get(metric)
            metric_validation_values = history_dict.get(f"val_{metric}")

            if metric_values or metric_validation_values:
                plt.figure()
                if metric_values:
                    plt.plot(epochs, metric_values, label=metric)
                if metric_validation_values:
                    plt.plot(epochs, metric_validation_values, label=f"val_{metric}")
                plt.title(metric)
                plt.xlabel("epochs")
                plt.legend()


def create_tensorboard_callback(dir_name, experiment_name):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(dir_name, experiment_name, ts)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving Tensorboard log files to {log_dir}")
    return tensorboard_callback
