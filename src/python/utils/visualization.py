import numpy as np
import matplotlib.pyplot as plt

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
    neighbourhood_size = (num_labels - 1) // 2
    x_labels = list()
    for i in range(neighbourhood_size, 0, -1):
        x_labels.append('t - ' + str(i))
    x_labels.append('t')
    for i in range(1, neighbourhood_size + 1):
        x_labels.append('t + ' + str(i))
    return x_labels


def visualize_sample(xi, yi, probabilities, predictions):
    """
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
