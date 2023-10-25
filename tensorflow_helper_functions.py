"""
Python module that contains tensorflow helper functions
"""
import datetime
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
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


def evaluate_classes_prediction(y_true, y_pred):
    """
    Calculate accuracy, precision, f1 score and recall between ground truth and predicted label and return values as a dictionary
    :param y_true: 1D Array of truth labels
    :param y_pred: 1D Array of predicted labels
    :return: a dictionary of accuracy, precision, recall, f1-score.
    """
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return {"accuracy": model_accuracy,
            "precision": model_precision * 100,
            "recall": model_recall * 100,
            "f1": model_f1 * 100}


def check_content_folder(folder_path, ignore_dot_folder=True):
    """
    print out the number of subdirectories and filenames of a folder
    :param folder_path: path of the folder in string format
    :param ignore_dot_folder: ignore folder that starts with a dot (like .git)
    :return: nothing
    """
    for dir_path, dir_names, filenames in os.walk(folder_path):
        if ignore_dot_folder and any([folder_name.startswith(".") for folder_name in str(dir_path).split(os.sep)]):
            continue

        if len(filenames) > 0:
            ext_set = {os.path.splitext(file)[-1] for file in filenames}
            ext_string = f" {ext_set}"
        else:
            ext_string = ""

        print(f"There are {len(dir_names)} directories and {len(filenames)} files{ext_string} in '{dir_path}'.")


def plot_loss_and_metrics_curves(histories, metrics=None):
    """
    Plot loss curves and metrics curves of histories in separate graph
    :param histories: a single history object or list of histories of a model fit with keras
    :param metrics: a string of a single metrics or list of multiple metrics to plot
    :return: nothing
    """
    # args checks
    if not isinstance(histories, list):
        histories = [histories]
    if isinstance(metrics, str):
        metrics = [metrics]

    # plot loss
    plt.figure()
    for history in histories:
        model_name = history.model.name
        epochs = range(history.epoch[0], history.epoch[-1] + 1)
        loss = history.history.get("loss")
        val_loss = history.history.get("val_loss")

        plt.plot(epochs, loss, label=f"{model_name}_training_loss")
        if val_loss:
            plt.plot(epochs, val_loss, label=f"{model_name}_validation_loss")

    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    # plot metrics
    if isinstance(metrics, list):
        for metric in metrics:
            plt.figure()
            for history in histories:
                model_name = history.model.name
                epochs = range(history.epoch[0], history.epoch[-1] + 1)
                metric_values = history.history.get(metric)
                metric_validation_values = history.history.get(f"val_{metric}")

                if metric_values or metric_validation_values:
                    if metric_values:
                        plt.plot(epochs, metric_values, label=f"{model_name}_{metric}")
                    if metric_validation_values:
                        plt.plot(epochs, metric_validation_values, label=f"{model_name}_val_{metric}")
            plt.title(metric)
            plt.xlabel("epochs")
            plt.legend()


def create_tensorboard_callback(dir_name, experiment_name):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(dir_name, experiment_name, ts)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving Tensorboard log files to {log_dir}")
    return tensorboard_callback


def load_and_prep_images(filenames, img_shape, channels=3, scale=True, return_batch=True):
    """
    Read one or more images from filenames, turns them into a tensor, reshape to specified format and rescale if needed
    :param filenames: str or list of path to target images
    :param img_shape: int or tuple height/width dimension of target image size
    :param channels: int number of color channel of the original image
    :param scale: if True, scale pixel from 0-255 to 0-1
    :param return_batch: if True, return the processed images as a batch list of tensors
    :return: if single path provided, return the image tensor of shape (img_shape, img_shape), otherwise return a list of
    image tensors of shape (img_shape, img_shape)
    """
    if isinstance(img_shape, int):
        img_shape = [img_shape, img_shape]
    if isinstance(filenames, str):
        filenames = [filenames]

    images = []
    for filename in filenames:
        img = tf.io.read_file(filename)
        img = tf.io.decode_image(img, channels=channels)
        img = tf.image.resize(img, img_shape)
        if scale:
            img = img/255.0
        images.append(img)

    if return_batch:
        return tf.stack([*images])

    if len(images) == 1:
        return images[0]
    return images


def pred_and_plot(model, image_tensors, class_names, true_labels=None, limit_bar=-1):
    """
    Make prediction with model on image_tensors and plot results and prediction probabilities
    :param model: a tensorflow model with predict method
    :param image_tensors: a single unbatched image tensor or a batch of image tensors
    :param class_names: class names corresponding to numbers predicted by the model
    :param true_labels: either list of numbers corresponding to true class number or list of string corresponding to class names
    :param limit_bar: (int) top k number of prediction to plot. plot all preds if negative or null
    :return: Nothing
    """
    if image_tensors.ndim == 3:
        # missing batch dimension
        image_tensors = tf.expand_dims(image_tensors, axis=0)

    if true_labels is not None and len(image_tensors) != len(true_labels):
        true_labels = None

    # convert true_labels to class_names labels if need
    if true_labels is not None and isinstance(true_labels[0], int):
        true_labels = [class_names[true_label] for true_label in true_labels]

    y_preds = model.predict(image_tensors, verbose=0)
    pred_classes = tf.argmax(y_preds, axis=1).numpy()
    pred_probs = tf.math.reduce_max(y_preds, axis=1).numpy()

    for i, image in enumerate(image_tensors):
        # plot image
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.axis(False)
        plt.imshow(image/255.0)
        # check for true labels
        color = "black"
        label = f"Predicted class: {class_names[pred_classes[i]]} ({round(pred_probs[i]*100,2)}%)"
        if true_labels:
            if true_labels[i] == class_names[pred_classes[i]]:
                # correct prediction
                color = "green"
            else:
                color = "red"
                label += f" Correct class: {true_labels[i]}"

        plt.title(label=label, color=color)

        # plot bar graph of probabilities
        plt.subplot(1, 2, 2)
        if limit_bar > 0:
            label = f"Top {limit_bar} prediction probabilites"
            result = tf.math.top_k(y_preds[i], k=limit_bar)
            values = result.values.numpy()
            indices = result.indices.numpy()
            class_names_plot = [class_names[indice] for indice in indices]

        else:
            label = "Prediction probabilites"
            values = y_preds[i]
            class_names_plot = class_names

        plt.bar(height=values, x=class_names_plot)
        plt.title(label=label)
        plt.xticks(rotation=70)


def plot_model_feature_maps(model, img, layer_class_to_plot=("Conv2D", "MaxPooling2D"), cmap="viridis", scale_factor=20):
    """
    Plot the results of an image passed througt each layer of a model
    :param model: a keras model
    :param img: an img processed with load_and_prep_images
    :param layer_class_to_plot: layer class to plot. Support Conv2D and MaxPooling2D for now
    :param cmap: color map from matplotlib
    :param scale_factor: factor to adjut the plot size
    :return: Nothing

    improve : - add heatmaps of activations : https://glassboxmedicine.com/2019/06/11/cnn-heat-maps-class-activation-mapping-cam/
              - support other layer class
    """

    successive_outputs = [layer.output for layer in model.layers]
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    successive_feature_maps = visualization_model.predict(img, verbose=0)

    layer_names = [layer.name for layer in model.layers]
    layer_classes = [layer.__class__.__name__ for layer in model.layers]
    layer_shapes = [f"IN: {layer.input_shape[1:]} -> OUT:{layer.output_shape[1:]}" for layer in model.layers]

    for layer_name, layer_class, layer_shape, feature_map in zip(layer_names, layer_classes, layer_shapes,
                                                                 successive_feature_maps):
        if layer_class in layer_class_to_plot:
            n_features = feature_map.shape[-1]  # number of features in feature map

            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]

            # Tile the images in this matrix
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                if x.std() != 0:
                    x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')

                # Tile each filter into this big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x

            # Display the grid
            scale = scale_factor / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(f"{layer_name} | {layer_shape}")
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap=cmap)


def plot_time_series(values, timesteps=None, series_label=None, y_label=None, start=0, end=-1, plot_styles="-", figsize=(10, 7)):
    """
    Plot values against timesteps.
    :param values: numpy array of values across time
    :param timesteps: numpy array of timesteps. If None, plot values end to end. If not enough timesteps for series, reuse the first one and truncated values or timesteps as needed
    :param series_label: labels corresponding to values for legend
    :param y_label: label for y-axis
    :param start: where to start the plot (setting a value will index from start of total x-axis)
    :param end: where to end the plot (setting a value will index from end of total x-axis)
    :param plot_styles: style of plot, default "-" (line)
    :param figsize: figsize for the plot
    :return: Nothing
    """
    plt.figure(figsize=figsize)

    if not isinstance(values, list):
        values = [values]

    if timesteps is None:
        timesteps = []
        last_index = 0
        for value in values:
            time = list(range(last_index, len(value) + last_index))
            timesteps.append(time)
            last_index = time[-1]
    if not isinstance(timesteps, list):
        # single timesteps provided
        timesteps = [timesteps]

    if len(timesteps) != len(values):
        # use first timesteps for every values
        timesteps = [timesteps[0]] * len(values)

    if series_label is None:
        # no label provided
        series_label = [None] * len(values)
    if not isinstance(series_label, list):
        # one label provided
        series_label = [series_label]
    if len(values) > len(series_label):
        # not enough label provided
        print(f"WARNING: Not enough labels for {len(values)} series. Only {len(series_label)} labels provided.")
        new_label = [f"label_{i}" for i in range(0, len(values)-len(series_label))]
        series_label.extend(new_label)

    for time, series, label in zip(timesteps, values, series_label):
        if len(time) > len(series):
            print(f"WARNING: number of timesteps ({len(time)}) > number of values ({len(series)}). Timesteps will be shortened to match values.")
            time = time[:len(series)]
        elif len(time) < len(series):
            print(f"WARNING: number of timesteps ({len(time)}) < number of values ({len(series)}). Values will be shortened to match timesteps.")
            series = series[:len(time)]

        plt.plot(time, series, plot_styles, label=label)

    plt.xlabel("Time")
    if y_label:
        plt.ylabel(y_label)
    plt.grid(True)
    if series_label:
        plt.legend(fontsize=14)
    if start != 0 or end != -1:
        common_axis = set()
        for timestep in timesteps:
            common_axis.update(timestep)
        common_axis = list(sorted(common_axis))

        plt.xlim(common_axis[start], common_axis[end])


def make_windows(x, window_size, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size and return it with corresponding labels of size horizon
    """
    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T
    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]
    # 4. Get the labelled windows
    windows, labels = windowed_array[:, :-horizon], windowed_array[:, -horizon:]
    return windows, labels


def train_test_time_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(len(windows) * (1-test_split))
    train_windows = windows[:split_size, :]
    train_labels = labels[:split_size, :]
    test_windows = windows[split_size:, :]
    test_labels = labels[split_size:, :]
    return train_windows, test_windows, train_labels, test_labels
