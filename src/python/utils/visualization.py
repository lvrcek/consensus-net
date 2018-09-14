import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# This order of labels is used also when creating dataset (X and y) from
# pileups. Those two orders must match!
# TODO(ajuric): Define this order in one place!
labels = ['A', 'C', 'G', 'T']
colors = ['green', 'blue', 'orange', 'red']


def _generate_x_axis_labels(num_labels):
    """
    Generates a list of labels for x axis when visualizing sample prediction.

    :param num_labels: number of ticks for x axis
    :type num_labels: int
    :return: list of labels for x axis
    :rtype list of str
    """
    numbers = (num_labels - 1) // 2
    labels = list()
    for position in range(-numbers, numbers + 1):
        if position < 0:
            labels.append('t - {}'.format(-position))
        elif position == 0:
            labels.append('t')
        else:
            labels.append('t + {}'.format(position))
    return labels


def visualize_sample(xi, yi, probabilities, predictions):
    """
    DEPRECATED!!!
    Displays visualization of example: shows the number of letters (A, C, G,
    T) for each position in example.

    Number of positions in example depends on the size of neighbourhood and
    is always odd. The middle position in the position for which the
    prediction is being calculated.

    :param xi: example
    :type xi: np.ndarray
    :param yi: ground truth
    :type yi: np.ndarray
    :param probabilities: predicted probabilites
    :type probabilities: np.ndarray
    :param predictions: predicted classes, one-hot encoded
    :type predictions: np.ndarray
    """

    # Create bar chart.
    num_positions = xi.shape[0]
    indices = np.arange(num_positions)
    width = 0.2

    fig, ax = plt.subplots(figsize=(15, 5))
    for i, (label, color) in enumerate(zip(labels, colors)):
        rects = ax.bar(indices + i * width, xi[:, 0, i], width, label=label,
                       color=color)
        for rect in rects: # fills numbers at the top of bar charts
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h,
                    '%s:%d' % (label, int(h)), ha='center', va='bottom')

    plt.ylabel('Count')
    plt.title('Sample visualization')
    x_axis_labels = _generate_x_axis_labels(num_positions)
    plt.xticks(indices + ((len(labels) - 1) * width) / 2, x_axis_labels)
    plt.legend(loc='best')
    plt.show()

    # Print classes, probabilities and predictions below bar chart.
    plt.figure(figsize=(15, 5))
    yi_class = np.argmax(yi)
    pi_class = np.argmax(predictions)
    text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif',
                   'fontweight': 'bold'}
    plt.text(0.5, 1, 'True label: $y_i={}$'.format(labels[yi_class]),
             color=colors[yi_class], size=20, **text_params)
    plt.text(0.5, 0.9, 'Predicted label: $p_i={}$'.format(labels[pi_class]),
             color=colors[pi_class], size=20, **text_params)
    plt.text(0.5, 0.8, 'Probabilities: {}'.format(probabilities), color='black',
             size=20, **text_params)
    plt.text(0.35, 0.7, 'A', color=colors[0], size=20, **text_params)
    plt.text(0.50, 0.7, 'C', color=colors[1], size=20, **text_params)
    plt.text(0.65, 0.7, 'G', color=colors[2], size=20, **text_params)
    plt.text(0.80, 0.7, 'T', color=colors[3], size=20, **text_params)
    plt.axis('off')
    plt.show()


def visualize_prediction(xi, yi, probabilities, predictions):
    """
    Displays visualization of prediction using heatmap representation.

    Shows the number of letters (A, C, G, T, D) for each position in example,
    predictions and probabilities for that example.

    Number of positions in example depends on the size of neighbourhood and
    is always odd. The middle position in the position for which the
    prediction is being calculated.

    :param xi: example
    :type xi: np.ndarray
    :param yi: ground truth
    :type yi: np.ndarray
    :param probabilities: predicted probabilites
    :type probabilities: np.ndarray
    :param predictions: predicted class, one-hot encoded
    :type predictions: np.ndarray
    """
    example = xi.T
    num_positions = example.shape[1]
    positions = _generate_x_axis_labels(num_positions)

    bases = ['A', 'C', 'G', 'T', 'D']
    output_bases = ['A', 'C', 'G', 'T', 'I', 'D']

    fig = plt.figure(figsize=(14, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])

    # Plot example.
    ax0 = plt.subplot(gs[0])
    im = ax0.imshow(example)

    # We want to show all ticks...
    ax0.set_xticks(np.arange(len(positions)))
    ax0.set_yticks(np.arange(len(bases)))
    # ... and label them with the respective list entries
    ax0.set_xticklabels(positions)
    ax0.set_yticklabels(bases)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax0.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(bases)):
        for j in range(len(positions)):
            ax0.text(j, i, example[i, j], ha="center", va="center", color="w")

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbarlabel = 'frequency'
    cbar = ax0.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    true_label = output_bases[np.argmax(yi)]
    ax0.set_title('Example, true_label = {}'.format(true_label))
    fig.tight_layout()

    # Plot prediction.
    ax1 = plt.subplot(gs[1])
    probabilities = np.array([probabilities]).T
    im = ax1.imshow(probabilities)

    ax1.set_yticks(np.arange(len(output_bases)))
    ax1.set_xticks(np.arange(0))
    # ... and label them with the respective list entries
    ax1.set_yticklabels(probabilities)
    for i in range(len(output_bases)):
        ax1.text(0, i, output_bases[i], ha="center", va="center", color="w")

    cbarlabel = 'Probabilities'
    cbar = ax1.figure.colorbar(im, ax=ax1)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    predicted_label = output_bases[np.argmax(predictions)]
    ax1.set_title('Predictions, label = {}'.format(predicted_label))

###############################################################
# WARNING: Following code is written for racon-hax data format.
###############################################################

class_colors = ['green', 'blue', 'orange', 'red', 'grey', 'black']


def color_bars(bars):
    """
    Sets colors to given bars.

    :param bars: Bars in display.
    :type bars: matplotlib.pyplot.bars
    """
    for bar, color in zip(bars, class_colors):
        bar.set_color(color)


def annotate_height_value_of_bars(bars, ax, format_type):
    """
    Adds heights at given bar plots.

    :param bars: Bars in display.
    :type bars: matplotlib.pyplot.bars
    :param ax: Canvas for plotting.
    :type ax: matplotlib.pyplot.ax
    :param format_type: Indicates only 'int' or 'float' for displaying purposes.
    :type format_type: str
    """
    if format_type not in ['int', 'float']:
        raise ValueError(
            'Expected format_type is \'int\' or \'float\', but {} given.'.format(
                format_type))
    for bar in bars:
        height = bar.get_height()
        if format_type == 'int':
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    '{:.1e}'.format(float(height)), ha='center', va='bottom')
        else:  # format_type == 'float'
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    '{:.4}'.format(height), ha='center', va='bottom')


def show_classes_statistics(labels, y_data):
    """
    WARNING: This function is for racon-hax data format.
    Displays textual info about data.

    :param labels: X-axis labels (classes).
    :type labels: list of str
    :param values: List of values to be plotted.
    :type values: list of float
    """
    print('Total number of data: {}'.format(int(np.sum(y_data))))

    for yi, label in zip(y_data, labels):
        print('Number of {}: {}'.format(label, int(yi)))


def plot_classes_statistics(labels, values, dataset_type, include_indels=True):
    """
    WARNING: This function is for racon-hax data format.

    :param labels: X-axis labels (classes).
    :type labels: list of str
    :param values: List of values to be plotted.
    :type values: list of np.ndarray
    :param dataset_type: Indicates whether this is 'train', 'valid' or 'test'
        set.
    :type dataset_type: str
    """
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    # Plot total number of classes - bar plot.
    ax_0 = plt.subplot(gs[0])
    bars_1 = plt.bar(labels, values, width=0.8)
    annotate_height_value_of_bars(bars_1, ax_0, 'int')
    color_bars(bars_1)
    ax_0.set_title(dataset_type + '- total number')

    # Plot relative to maximum number of classes - bar plot.
    ax_1 = plt.subplot(gs[1])
    max_value = np.max(values)
    values_relative = [yi / max_value for yi in values]
    bars_2 = plt.bar(labels, values_relative)
    annotate_height_value_of_bars(bars_2, ax_1, 'float')
    color_bars(bars_2)
    ax_1.set_title(dataset_type + '- relative to maximal number')

    # Plot relative number of classes - bar plot.
    ax_1 = plt.subplot(gs[2])
    explode = [0.08] * (6 if include_indels else 4)
    plt.pie(values, labels=labels, colors=class_colors, autopct='%1.1f%%',
            shadow=True, startangle=0, explode=explode)
    ax_1.set_title(dataset_type + '- relative number')


def dataset_classes_summary(y_data, dataset_type, include_indels=True):
    """
    WARNING: This function is for racon-hax data format.

    Creates and displays dataset summary including textual summary and
    graphical plots (bar and pie charts).

    :param y_data: Data for summary - should be labels (y).
    :type y_data: np.ndarray
    :param dataset_type: Indicates whether this is 'train', 'valid' or 'test'
        set.
    :type dataset_type: str
    """
    num_A = np.sum(y_data[:, 0])
    num_C = np.sum(y_data[:, 1])
    num_G = np.sum(y_data[:, 2])
    num_T = np.sum(y_data[:, 3])

    x_values = ['A', 'C', 'G', 'T']
    y_values = [num_A, num_C, num_G, num_T]

    if include_indels:
        num_I = np.sum(y_data[:, 4])
        num_D = np.sum(y_data[:, 5])

        x_values.append('I')
        x_values.append('D')

        y_values.append(num_I)
        y_values.append(num_D)

    show_classes_statistics(x_values, y_values)
    plot_classes_statistics(x_values, y_values, dataset_type, include_indels)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param cm: Confusion matrix.
    :type cm: sklearn.metrics.confusion_matrix
    :param classes: List of name of classes.
    :type classes: list of str
    :param title: Title of plot.
    :type title: str
    :param cmap: Color map.
    :type cmap: plt.cm
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
