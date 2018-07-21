import numpy as np
import os

from keras.models import load_model
from src.python.dataset import dataset

CONSENSUS_SUMMARY_CMD_1 = '{}/mummer3.23/dnadiff -p {}/dnadiff-output {} {} ' \
                          '2>> {}/err'
CONSENSUS_SUMMARY_CMD_2 = 'head -n 24 {}/dnadiff-output.report | tail -n 20'

RESULT_CMD = 'cp {}/dnadiff-output.report {}'


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
                   neighbourhood_size, output_dir, tools_dir, result_file_path):
    """
    Creates consensus which is polished by trained neural network.

    Pileup generator creates pileups by using it's own pileup strategy. Those
    pileups are used to create dataset (examples with given neighbourhood
    size). Given model makes predictions (contigs) for created dataset. Those
    contigs are concatenated the same way as in .sam file. At last, polished
    genome is compared with reference.

    :param model_path: Path to trained model.
    :type model_path: str
    :param reference_path: Path to reference.
    :type reference_path: str
    :param pileup_generator: Pileup Generator object which creates pileups
        using it's own strategy.
    :type pileup_generator: PileupGenerator
    :param neighbourhood_size: Number of neighbours to use from one size (eg.
        if you set this parameter to 3, it will take 3 neighbours from both
        sides so total number of positions in one example will be 7 -
        counting the middle position).
    :type neighbourhood_size: int
    :param output_dir: Path to output directory. There will all outputs be
        saved.
    :type output_dir: str
    :param tools_dir: Path to directory where are used tools are installed.
    :type tools_dir: str
    :param result_file_path: Path where will results be copied to.
    :type result_file_path: str
    """
    # TODO(ajuric): Currently, y is also created while calculating consensus,
    # due to reuising existing code from training. But, here in inference y
    # is not used. This needs to be removed to reduce the unnecessary overhead.

    if os.path.exists(output_dir):
        raise FileExistsError('Given directory already exists: {}! Provide '
                              'non-existing directory.')

    os.makedirs(output_dir)

    print('----> Create pileups from assembly. <----')
    X, y, contig_names = \
        pileup_generator.generate_pileups()

    print('----> Create dataset with neighbourhood from pileups. <----')
    X, y = dataset.create_dataset_with_neighbourhood(
            neighbourhood_size,
            mode='inference',
            X_list=X,
            y_list=y)

    print('----> Load model and make predictions (consensus). <----')
    model = load_model(model_path)

    contigs = list()
    for X, y, contig_name in zip(X, y, contig_names):
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
    os.system(RESULT_CMD.format(output_dir, result_file_path))


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
