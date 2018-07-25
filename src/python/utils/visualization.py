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
    ax0.set_yticklabels(labels)

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

    true_label = bases[np.argmax(yi)]
    ax0.set_title('Example, true_label = {}'.format(true_label))
    fig.tight_layout()

    # Plot prediction.
    ax1 = plt.subplot(gs[1])
    probabilities = np.array([probabilities]).T
    im = ax1.imshow(probabilities)

    output_bases = ['A', 'C', 'G', 'T', 'D', 'I']
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
