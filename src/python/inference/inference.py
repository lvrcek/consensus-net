import numpy as np
import os

from keras.models import load_model
from ..dataset import dataset

CONSENSUS_SUMMARY_CMD_1 = '{}/mummer3.23/dnadiff -p {}/dnadiff-output {} {} ' \
                          '2>> {}/err'
CONSENSUS_SUMMARY_CMD_2 = 'head -n 24 {}/dnadiff-output.report | tail -n 20'


def _convert_predictions_to_genome(predictions):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    genome = [mapping[prediction] for prediction in predictions]
    return genome


def _write_genome_to_fasta(genome, fasta_file_path, contig_name):
    with open(fasta_file_path, 'w') as f:
        f.write('>{} LN:{}\n'.format(contig_name, len(genome)))
        f.write('{}'.format(''.join(genome)))


def make_consensus(model_path, assembly_fasta_path, bam_file_path, contig,
                   neighbourhood_size, output_dir, tools_dir,
                   include_indels=True):
    print('----> Create pileups from assembly. <----')
    X, y = dataset.generate_pileups(contig, bam_file_path,
                                    assembly_fasta_path,
                                    save_directory_path=output_dir,
                                    include_indels=include_indels)
    X_save_path = os.path.join(output_dir, 'X-pileups{}'.format(
        'indels' if include_indels else ''))
    y_save_path = os.path.join(output_dir, 'y-pileups')
    np.save(X_save_path, X)
    np.save(y_save_path, y)

    print('----> Create dataset with neighbourhood from pileups. <----')
    X, y = dataset.create_dataset_with_neighbourhood([X_save_path],
                                                     [y_save_path],
                                                     neighbourhood_size)
    X_save_path = os.path.join(output_dir, 'X-pileups-n{}{}'.format(
        neighbourhood_size, 'indels' if include_indels else ''))
    y_save_path = os.path.join(output_dir, 'y-pileups-n{}'.format(
        neighbourhood_size))
    np.save(X_save_path, X)
    np.save(y_save_path, y)

    print('----> Reshape dataset for convolutional network. <----')
    X, y = dataset.read_dataset_and_reshape_for_conv(X_save_path, y_save_path)

    print('----> Load model and make predictions (consensus). <----')
    model = load_model(model_path)

    probabilities = model.predict(X)
    predictions = np.argmax(probabilities, axis=1)

    genome = _convert_predictions_to_genome(predictions)
    consensus_path = os.path.join(output_dir, 'consensus.fasta')
    _write_genome_to_fasta(genome, consensus_path, contig)

    print('----> Create consensus summary. <----')
    os.system(CONSENSUS_SUMMARY_CMD_1.format(tools_dir, output_dir,
                                             assembly_fasta_path,
                                             consensus_path, output_dir))
    os.system(CONSENSUS_SUMMARY_CMD_2.format(output_dir))
