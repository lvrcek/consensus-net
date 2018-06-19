import numpy as np
import os

from keras.models import load_model
from src.python.dataset import dataset

CONSENSUS_SUMMARY_CMD_1 = '{}/mummer3.23/dnadiff -p {}/dnadiff-output {} {} ' \
                          '2>> {}/err'
CONSENSUS_SUMMARY_CMD_2 = 'head -n 24 {}/dnadiff-output.report | tail -n 20'


def _convert_predictions_to_genome(predictions):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: '', 5: 'N'}
    genome = [mapping[prediction] for prediction in predictions]
    return genome


def _write_genome_to_fasta(contigs, fasta_file_path, contig_names):
    with open(fasta_file_path, 'w') as f:
        for contig, contig_name in zip(contigs, contig_names):
            f.write('>{} LN:{}\n'.format(contig_name, len(contig)))
            f.write('{}\n'.format(''.join(contig)))


def make_consensus(model_path, reference_path, pileup_generator,
                   neighbourhood_size, output_dir, tools_dir):
    # TODO(ajuric): Currently, y is also created while calculating consensus, due to
    # reuising existing code from training. But, here in inference y is not used.
    # This needs to be removed to reduce the unnecessary overhead.

    print('----> Create pileups from assembly. <----')
    X, y, X_save_paths, y_save_paths, contig_names = \
        pileup_generator.generate_pileups()

    print('----> Create dataset with neighbourhood from pileups. <----')
    X, y, X_save_paths, y_save_paths = \
        dataset.create_dataset_with_neighbourhood(
        X_save_paths,
        y_save_paths,
        neighbourhood_size,
        mode='inference',
        save_directory_path=output_dir)

    print('----> Reshape dataset for convolutional network. <----')
    X_list, y_list = dataset.read_dataset_and_reshape_for_conv(X_save_paths,
                                                       y_save_paths)

    print('----> Load model and make predictions (consensus). <----')
    model = load_model(model_path)

    contigs = list()
    for X, y, contig_name in zip(X_list, y_list, contig_names):
        probabilities = model.predict(X)
        predictions = np.argmax(probabilities, axis=1)

        contig = _convert_predictions_to_genome(predictions)
        contigs.append(contig)

    consensus_path = os.path.join(output_dir, 'consensus.fasta')
    _write_genome_to_fasta(contigs, consensus_path, contig_names)

    print('----> Create consensus summary. <----')
    os.system(CONSENSUS_SUMMARY_CMD_1.format(tools_dir, output_dir,
                                             reference_path,
                                             consensus_path, output_dir))
    os.system(CONSENSUS_SUMMARY_CMD_2.format(output_dir))


# @TODO(ajuric): Refactor this consensus methods.
def make_consensus_before_shapeing_tmp(X_path, y_path, model_path,
                                       output_dir, tools_dir,
                                       reference_path, contig):
    print('----> Reshape dataset for convolutional network. <----')
    X, y = dataset.read_dataset_and_reshape_for_conv(X_path, y_path)

    print('----> Load model and make predictions (consensus). <----')
    model = load_model(model_path)

    probabilities = model.predict(X)
    predictions = np.argmax(probabilities, axis=1)

    genome = _convert_predictions_to_genome(predictions)
    consensus_path = os.path.join(output_dir, 'consensus.fasta')
    _write_genome_to_fasta(genome, consensus_path, contig)

    print('----> Create consensus summary. <----')
    os.system(CONSENSUS_SUMMARY_CMD_1.format(tools_dir, output_dir,
                                             reference_path,
                                             consensus_path, output_dir))
    os.system(CONSENSUS_SUMMARY_CMD_2.format(output_dir))

# @TODO(ajuric): Refactor this consensus methods.
def make_consensus_only(X_path, y_path, model_path,
                                       output_dir, tools_dir,
                                       reference_path, contig):
    print('----> Load X and y. <----')
    X, y = np.load(X_path), np.load(y_path)

    print('----> Load model and make predictions (consensus). <----')
    model = load_model(model_path)

    probabilities = model.predict(X)
    predictions = np.argmax(probabilities, axis=1)

    genome = _convert_predictions_to_genome(predictions)
    consensus_path = os.path.join(output_dir, 'consensus.fasta')
    _write_genome_to_fasta(genome, consensus_path, contig)

    print('----> Create consensus summary. <----')
    os.system(CONSENSUS_SUMMARY_CMD_1.format(tools_dir, output_dir,
                                             reference_path,
                                             consensus_path, output_dir))
    os.system(CONSENSUS_SUMMARY_CMD_2.format(output_dir))
