import numpy as np
import progressbar
import pysam
import pysamstats
import os

from sklearn.model_selection import train_test_split
from Bio import SeqIO

ALL_CPU = -1
modes = ['training', 'inference']


def read_dataset_and_reshape_for_conv(path_X, path_y, validation_size=None):
    """
    Reads X and y from given paths and reshapes them for applying in
    convolutional networks.

    Reshaping is done by splitting different letters in separate channels,
    eg. letter 'A' has it's own channel, letter 'C' has it's own channel, etc.

    :param path_X: path to X data
    :type path_X: str
    :param path_y: path to y data
    :type path_y: str
    :param validation_size: specifies percentage of dataset used for validation
    :type validation_size: float
    :return: If validation_size is None, returns just X and y reshaped. If
    validation_size is float, returns a tuple in following order: (X, y,
    X_train, X_validate, y_train, y_validate).
    :rtype: tuple of np.ndarray
    """
    if validation_size is not None:
        if validation_size < 0 or validation_size > 1.0:
            raise ValueError('Validation size must be float from [0, 1], but {}'
                             ' given.'.format(validation_size))

    X, y = np.load(path_X), np.load(path_y)
    print('X shape before reshaping:', X.shape)
    print('y shape before reshaping:', y.shape)

    new_X = list()
    neighbourhood_size = X[0].shape[0]
    # Number of columns is equal to the number of letters in dataset (A, C,
    # G, T, I, D, ...).
    num_columns = X[0].shape[1]
    num_data = X.shape[0]
    with progressbar.ProgressBar(max_value=num_data) as progress_bar:
        for i, xi in enumerate(X):
            new_xi = np.dstack(
                [xi[:, col_index].reshape(neighbourhood_size, 1)
                 for col_index in range(num_columns)]
            )
            new_X.append(new_xi)
            progress_bar.update(i)

    new_X = np.array(new_X)
    X = new_X
    print('X shape after reshaping:', X.shape)
    print('y shape after reshaping:', y.shape)

    if validation_size is None:
        return X, y
    else:
        print('Splitting to train and validation set.')
        X_train, X_validate, y_train, y_validate = train_test_split(
            X, y, test_size=validation_size)
        print('X_train shape:', X_train.shape)
        print('X_validate shape:', X_validate.shape)
        print('y_train:', y_train.shape)
        print('y_validate:', y_validate.shape)
        return X, y, X_train, X_validate, y_train, y_validate


def _calc_empty_rows(X):
    """
    Calculates which rows in X are empty rows (i.e. all numbers in that row
    are equal to 0).

    :param X: 2-D data
    :type X: np.ndarray
    :return: 1-D array with 1s on positions which correspond to empty rows.
    :rtype: np.ndarray
    """
    empty_row = np.zeros((1, X.shape[1])) # size is second axis of X
    empty_rows = [int(v) for v in np.all(empty_row == X, axis=1)]
    return empty_rows


def create_dataset_with_neighbourhood(X_paths, y_paths, neighbourhood_size):
    """
    Creates datasets by mixing all pileups with given neighbourhood_size.

    Datasets are concatenated after extracting neighbourhood_size positions in
    given datasets separately.

    Dataset at i-th position in X_paths should match given labels at i-th
    position in y_paths.

    :param X_paths: list of paths to X pileup dataset
    :type X_paths: list of str
    :param y_paths: list of paths to y pileup dataset
    :type y_paths: list of str
    :param neighbourhood_size: number of neighbours to use from one size (eg.
        if you set this parameter to 3, it will take 3 neighbours from both
        sides so total number of positions in one example will be 7 -
        counting the middle position)
    :type neighbourhood_size: float
    :return:
    :rtype tuple of np.ndarray
    """
    if not len(X_paths) == len(y_paths):
        raise ValueError('Number of X_paths and y_paths should be the same!')

    new_X, new_y = list(), list()
    for X_path, y_path in zip(X_paths, y_paths):
        print('Parsing ', X_path, ' and ', y_path)

        X, y = np.load(X_path), np.load(y_path)
        # Removing last column which everything which was not 'A' nor 'C' nor
        #  'G' nor 'T'.
        y = y[:, :4]

        empty_rows = _calc_empty_rows(X)

        print('Creating dataset with neighbourhood ...')
        with progressbar.ProgressBar(max_value=X.shape[0]) as progress_bar:
            # TODO(ajuric): Check if this can be speed up.
            for i in range(X.shape[0]):
                progress_bar.update(i)
                if empty_rows[i] == 1:
                    continue  # current row is empty row
                if i < neighbourhood_size or \
                   i >= X.shape[0] - neighbourhood_size:
                    # Current position is not suitable to build an example.
                    continue
                zeros_to_left = np.sum(empty_rows[i - neighbourhood_size:i])
                zeros_to_right = np.sum(
                    empty_rows[i + 1:i + neighbourhood_size + 1])
                if zeros_to_left == 0 and zeros_to_right == 0:
                    new_X.append(
                        X[i - neighbourhood_size:i + neighbourhood_size + 1])
                    new_y.append(y[i])

    return np.array(new_X), np.array(new_y)


# Parameters '-L100 -Sw5 -m0' are suggested params from minimap tool. See
# https://github.com/lh3/minimap/blob/master/README.md.
OVERLAP_CMD = '{}/minimap/minimap -t {} -L100 -Sw5 -m0 {} {} > {}/ovl.paf ' \
               '2>> {}/err'

LAYOUT_CMD_1 = '{}/miniasm/miniasm -f {} {}/ovl.paf > {}/lay.gfa 2>> {}/err'
LAYOUT_CMD_2 = 'awk \'$1 ~/S/ {print ">"$2"\n"$3}\' {}/lay.gfa > ' \
               '{}/iter0.fasta'

CONSENSUS_ITER1_CMD_1 = '{}/miniamp/minimap -t {} {}/iter0.fasta {} > ' \
                        '{}/iter1.paf 2>> {}/err'
CONSENSUS_ITER1_CMD_2 = '/usr/bin/time -v -a -o {}/time_memory_new.txt ' \
                        '{}/racon/build/bin/racon -t {} {} {}/iter1.paf ' \
                        '{}/iter0.fasta > {}/iter1.fasta 2>> {}/err'

CONSENSUS_ITER2_CMD_1 = '{}/miniamp/minimap -t {} {}/iter1.fasta {} > ' \
                        '{}/iter2.paf 2>> {}/err'
CONSENSUS_ITER2_CMD_2 = '/usr/bin/time -v -a -o {}/time_memory_new.txt ' \
                        '{}/racon/build/bin/racon -t {} {} {}/iter2.paf ' \
                        '{}/iter1.fasta > {}/iter2.fasta 2>> {}/err'

ASSEMBLY_SUMMARY_CMD_1 = '{}/mummer3.23/dnadiff -p {}/out {} {}/iter2.fasta ' \
                         '2>> {}/err'
ASSEMBLY_SUMMARY_CMD_2 = 'head -n 24 {}/out.report | tail -n 20'


def _create_assembly(reads_path, reference_path, output_dir,
                     tools_dir, num_threads):
    """
    Creates assembly as described by OLC paradigm: overlap, layout and
    consensus phase.

    :param reads_path: path to reads
    :type reads_path: str
    :param reference_path: path to reference
    :type reference_path: str
    :param output_dir: output directory
    :type output_dir: str
    :param tools_dir: tools directory (specifies where the tools which are used
    during assembly creation are installed)
    :type tools_dir: str
    :param num_threads: number of threads to use
    :type num_threads: int
    """
    print('##### Creating assembly #####')

    print('-----> 1. Overlap phase. <-----')
    os.system(OVERLAP_CMD.format(tools_dir, num_threads, reads_path,
                                 reads_path, output_dir, output_dir))

    print('-----> 2. Layout phase. <-----')
    os.system(LAYOUT_CMD_1.format(tools_dir, reads_path, output_dir,
                                  output_dir, output_dir))
    os.system(LAYOUT_CMD_2.format(output_dir, output_dir))

    print('-----> 3. Consensus phase. <-----')
    print('----------> 3.1. First iteration. <----------')
    os.system(CONSENSUS_ITER1_CMD_1.format(tools_dir, output_dir, reads_path,
                                           output_dir, output_dir))
    os.system(CONSENSUS_ITER1_CMD_2.format(output_dir, tools_dir,
                                           num_threads, reads_path,
                                           output_dir, output_dir,
                                           output_dir, output_dir))

    print('----------> 3.2. Second iteration. <----------')
    os.system(CONSENSUS_ITER2_CMD_1.format(tools_dir, output_dir, reads_path,
                                           output_dir, output_dir))
    os.system(CONSENSUS_ITER2_CMD_2.format(output_dir, tools_dir,
                                           num_threads, reads_path,
                                           output_dir, output_dir,
                                           output_dir, output_dir))

    print('-----> 4. Assembly summary. <-----')
    os.system(ASSEMBLY_SUMMARY_CMD_1.format(tools_dir, output_dir,
                                            reference_path, output_dir,
                                            output_dir))
    os.system(ASSEMBLY_SUMMARY_CMD_2.format(output_dir))


ALIGN_READS_TO_REF_CMD_1 = '{}/minimap2/minimap2/ -ax {} -t {} {} {} >' \
                           '{}/reads-to-ref.sam'
ALIGN_READS_TO_REF_CMD_2 = '{}/samtools-1.3.1/samtools view -b ' \
                           '{}/reads-to-ref.sam > {}/reads-to-ref.bam'
ALIGN_READS_TO_REF_CMD_3 = '{}/samtools-1.3.1/samtools sort ' \
                           '{}/reads-to-ref.bam > {}/reads-to-ref-sorted.bam'
ALIGN_READS_TO_REF_CMD_4 = '{}/samtools-1.3.1/samtools index' \
                           '{}/reads-to-ref-sorted.bam'


def _align_reads_to_reference(reads_path, reference_path, output_dir,
                              tools_dir, num_threads, reads_type):
    """
    Aligns reads to reference.

    Those alignments are later used for generating pileups.

    :param reads_path: path to reads
    :type reads_path: str
    :param reference_path: path to reference
    :type reference_path: str
    :param output_dir: output directory
    :type output_dir: str
    :param tools_dir: tools directory (specifies where the tools which are used
    during assembly creation are installed)
    :type tools_dir: str
    :param num_threads: number of threads to use
    :type num_threads: int
    :param reads_type: one of 'pb' or 'ont', indicates which technology was
    used to create reads (PacBio or Oxford Nanopore)
    :type reads_type: str
    """
    reads_types_to_cmd_mapping = {'pb': 'map-pb', 'ont': 'map-ont'}
    if reads_type not in reads_types_to_cmd_mapping:
        raise ValueError('Supported reads_types are {}, but {} '
                         'given.'.format(reads_types_to_cmd_mapping.keys(),
                                         reads_type))

    command_type = reads_types_to_cmd_mapping[reads_type]

    print('##### Align reads to reference. #####')

    print('-----> 1. Align reads. <-----')
    os.system(ALIGN_READS_TO_REF_CMD_1.format(tools_dir, command_type,
                                              num_threads, reference_path,
                                              reads_path, output_dir))

    print('-----> 2. Convert .sam to .bam. <-----')
    os.system(ALIGN_READS_TO_REF_CMD_2.format(tools_dir, output_dir,
                                              output_dir))

    print('-----> 3. Sort alignments. <-----')
    os.system(ALIGN_READS_TO_REF_CMD_3.format(tools_dir, output_dir,
                                              output_dir))

    print('-----> 4. Create index. <-----')
    os.system(ALIGN_READS_TO_REF_CMD_4.format(tools_dir, output_dir))

ALIGN_READS_TO_ASM_CMD_1 = '{}/minimap2/minimap2/ -ax {} -t {} {}/iter2.fasta' \
                           '{} > {}/reads-to-asm.sam'
ALIGN_READS_TO_ASM_CMD_2 = '{}/samtools-1.3.1/samtools view -b ' \
                           '{}/reads-to-asm.sam > {}/reads-to-asm.bam'
ALIGN_READS_TO_ASM_CMD_3 = '{}/samtools-1.3.1/samtools sort ' \
                           '{}/reads-to-asm.bam > {}/reads-to-asm-sorted.bam'
ALIGN_READS_TO_ASM_CMD_4 = '{}/samtools-1.3.1/samtools index' \
                           '{}/reads-to-asm-sorted.bam'


def _align_reads_to_assembly(reads_path, output_dir,
                             tools_dir, num_threads, reads_type):
    """
    Aligns reads to assembly.

    Those alignments are later used for generating pileups.

    :param reads_path: path to reads
    :type reads_path: str
    :param output_dir: output directory
    :type output_dir: str
    :param tools_dir: tools directory (specifies where the tools which are used
    during assembly creation are installed)
    :type tools_dir: str
    :param num_threads: number of threads to use
    :type num_threads: int
    :param reads_type: one of 'pb' or 'ont', indicates which technology was
    used to create reads (PacBio or Oxford Nanopore)
    :type reads_type: str
    """
    reads_types_to_cmd_mapping = {'pb': 'map-pb', 'ont': 'map-ont'}
    if reads_type not in reads_types_to_cmd_mapping:
        raise ValueError('Supported reads_types are {}, but {} '
                         'given.'.format(reads_types_to_cmd_mapping.keys(),
                                         reads_type))

    command_type = reads_types_to_cmd_mapping[reads_type]

    print('##### Align reads to assembly. #####')

    print('-----> 1. Align reads. <-----')
    os.system(ALIGN_READS_TO_ASM_CMD_1.format(tools_dir, command_type,
                                              num_threads,
                                              reads_path, output_dir))

    print('-----> 2. Convert .sam to .bam. <-----')
    os.system(ALIGN_READS_TO_ASM_CMD_2.format(tools_dir, output_dir,
                                              output_dir))

    print('-----> 3. Sort alignments. <-----')
    os.system(ALIGN_READS_TO_ASM_CMD_3.format(tools_dir, output_dir,
                                              output_dir))

    print('-----> 4. Create index. <-----')
    os.system(ALIGN_READS_TO_ASM_CMD_4.format(tools_dir, output_dir))


def generate_assembly_and_alignments(reads_path, reference_path, output_dir,
                                     tools_dir, reads_type,
                                     num_threads=-1):
    """
    Generates assembly from given reads and aligns reads to given reference
    and created assembly.

    :param reads_path: path to reads
    :type reads_path: str
    :param reference_path: path to reference
    :type reference_path: str
    :param output_dir: output directory
    :type output_dir: str
    :param tools_dir: tools directory (specifies where the tools which are used
    during assembly creation are installed)
    :type tools_dir: str
    :param num_threads: number of threads to use, set to -1 to use all cpu
    available
    :type num_threads: int
    :param reads_type: one of 'pb' or 'ont', indicates which technology was
    used to create reads (PacBio or Oxford Nanopore)
    :type reads_type: str
    """

    if num_threads == ALL_CPU:
        num_threads = os.cpu_count()
    elif num_threads == 0 or num_threads < -2 or num_threads > os.cpu_count():
        raise ValueError('Number of threads -1 or from 1 to {}, but {} '
                         'given.'.format(os.cpu_count(), num_threads))

    _create_assembly(reads_path, reference_path, output_dir, tools_dir,
                     num_threads)
    _align_reads_to_reference(reads_path, reference_path, output_dir,
                              tools_dir, num_threads, reads_type)
    _align_reads_to_assembly(reads_path, output_dir, tools_dir, num_threads,
                             reads_type)


def _generate_pileups(bam_file_path, reference_fasta_path, include_indels=True):
    """
    Generate pileups from reads alignment to reference.

    Pileups are generated for all contigs.

    :param bam_file_path: path to .bam file containing alignments
    :type bam_file_path: str
    :param reference_fasta_path: path to .fasta file
    :type reference_fasta_path: str
    :param include_indels: flag which indicates whether to include indels in
        pileups
    :type include_indels: bool
    :return: pileups (X)
    :rtype: tuple of np.ndarray and list of str
    """
    bam_file = pysam.AlignmentFile(bam_file_path)

    if include_indels:
        info_of_interest = ['A', 'C', 'G', 'T', 'insertions', 'deletions']
    else:
        info_of_interest = ['A', 'C', 'G', 'T']

    pileups = [np.zeros(
        (bam_file.get_reference_length(contig_name),
         len(info_of_interest)
         )) for contig_name in bam_file.references]

    total_length = np.sum(
        [bam_file.get_reference_length(contig_name) for contig_name in
         bam_file.references])
    with progressbar.ProgressBar(max_value=total_length) as progress_bar:
        for contig_id, contig_name in enumerate(bam_file.references):
            for record in pysamstats.stat_variation(
                    bam_file, chrom=contig_name, fafile=reference_fasta_path):
                progress_bar.update(record['pos'])
                for i, info in enumerate(info_of_interest):
                    pileups[contig_id][record['pos']][i] += record[info]

    return pileups, bam_file.references


def _generate_ground_truth(reference_fasta_path, ordered_contigs):
    """
    Generates ground truth - nucleus bases from reference.

    It parses all contigs in same order as when generating pileups to make
    sure that every X corresponds to correct y.

    :param reference_fasta_path: path to .fasta file
    :type reference_fasta_path: str
    :param ordered_contigs: list of contigs
    :type ordered_contigs: list of str
    :return: nucleus bases from reference (y)
    :rtype: np.ndarray
    """
    record_dict = SeqIO.to_dict(SeqIO.parse(reference_fasta_path, 'fasta'))
    total_options = 5
    y_oh = [np.zeros((len(record_dict[contig_name]), total_options)) for
            contig_name in ordered_contigs]
    # Last number in shape - 5 - is for letters other than A, C, G and T.
    mapping = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3}

    total_length = np.sum(
        len(record_dict[contig_name]) for contig_name in ordered_contigs)
    with progressbar.ProgressBar(max_value=total_length) as progress_bar:
        for contig_id, contig_name in enumerate(ordered_contigs):
            contig = record_dict[contig_name]
            print(contig_name, len(contig))
            for position, base in enumerate(contig.seq):
                progress_bar.update(position)
                y_oh[contig_id][position][mapping.get(base, -1)] = 1
    return y_oh


def generate_pileups(bam_file_path, reference_fasta_path, mode,
                     save_directory_path=None, include_indels=True):
    """
    Generates pileups from given alignment stored in bam file.

    Mode must be one of 'training' or 'inference' string indicating pileups
    generation mode. If 'training' is selected, pileups from different
    contigs are all be concatenated. If 'inference' is selected, pileups from
    different contig will be hold separate to enable to make consensus with
    same number of contigs.

    If save_directory_path is provided, generated pileups are stored in that
    directory.

    If include_indels is set to True, indels will also we included in
    pileups. Otherwise, only nucleus bases will be in pileup (A, C, G and T).

    :param bam_file_path: path to .bam file containing alignments
    :type bam_file_path: str
    :param reference_fasta_path: path to .fasta file
    :type reference_fasta_path: str
    :param mode: either 'training' or 'inference' string, representing the
        mode for pileups generation
    :type mode: str
    :param save_directory_path: path to directory for storing pileups
    :type save_directory_path: str
    :param include_indels: flag which indicates whether to include indels in
        pileups
    :type include_indels: bool
    :return: pileups (X) and matching nucleus bases from reference (y) with
        concatenated contigs for inference mode, or separate contigs for
        training mode
    :rtype: tuple of np.ndarray or tuple of list of np.ndarray
    """
    if not mode in modes:
        raise ValueError('You must provide either \'training\' or '
                         '\'inference\' mode, but \'{}\' given.'.format(mode))

    if save_directory_path is not None:
        if os.path.exists(save_directory_path):
            raise ValueError('You must provide non-existing save output '
                             'directory, {} given.'.format(save_directory_path))
        else:
            os.makedirs(save_directory_path)

    print('##### Generate pileups from read alignments to reference. #####')

    print('-----> 1. Generate pileups. <-----')
    X, ordered_contigs = _generate_pileups(bam_file_path,
                                           reference_fasta_path,
                                           include_indels=include_indels)

    print('-----> 2. Generate ground truth. <-----')
    y_oh = _generate_ground_truth(reference_fasta_path, ordered_contigs)

    total_pileups = len(X)
    if mode == 'training': # training mode
        X, y_oh = np.concatenate(X, axis=0), np.concatenate(y_oh, axis=0)
        total_pileups = 1
    else:  # inference mode
        pass  # nothing to do

    if save_directory_path is not None:
        X_save_paths = [
            os.path.join(
                save_directory_path,
                'pileups-X-{}-ref{}'.format(
                    i, '-indels' if include_indels else ''))
            for i in range(total_pileups)]
        y_save_paths = [os.path.join(save_directory_path,
                        'pileups-y-{}-ref'.format(i))
                        for i in range(total_pileups)]
        for X_save_path, y_save_path in zip(X_save_paths, y_save_paths):
            np.save(X_save_path, X)
            np.save(y_save_path, y_oh)

    return X, y_oh
