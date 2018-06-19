import numpy as np
import progressbar
import os

from sklearn.model_selection import train_test_split

ALL_CPU = -1


def read_dataset_and_reshape_for_conv(X_list=None, y_list=None, X_paths=None,
                                      y_paths=None, validation_size=None):
    """
    Reads X and y from given paths and reshapes them for applying in
    convolutional networks.

    If X_list and y_list are given, X_paths and y_paths are ignored.

    Reshaping is done by splitting different letters in separate channels,
    eg. letter 'A' has it's own channel, letter 'C' has it's own channel, etc.

    :param X_list: list of X pileup dataset
    :type X_list: list of np.ndarray
    :param y_list: list of y pileup dataset
    :type y_list: list of np.ndarray
    :param X_paths: list of paths to X data
    :type X_paths: list of str
    :param y_paths: list of paths to y data
    :type y_paths: list of str
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
        if X_paths is not None:
            if not len(X_paths) == 1:
                raise ValueError(
                    'Validation size can only be provided if there is only '
                    'one X path and y_path.')

    if not ((X_list is None and y_list is None)
            or (X_paths is None and y_paths is None)):
        raise ValueError('Either X_list and y_list or X_paths and y_paths '
                         'must be provided!')

    if X_list is None and y_list is None:
        X_list = [np.load(X_path) for X_path in X_paths]
        y_list = [np.load(y_path) for y_path in y_paths]

    reshaped_X_list, reshaped_y_list = list(), list()
    for X, y in zip(X_list, y_list):
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

        reshaped_X_list.append(X), reshaped_y_list.append(y)

    if validation_size is None:
        return reshaped_X_list, reshaped_y_list
    else:
        # There is only one X and y (because, all datasets are concatenated
        # for training).
        X, y = reshaped_X_list[0], reshaped_y_list[0]
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
    empty_row = np.zeros((1, X.shape[1]))  # size is second axis of X
    empty_rows = [int(v) for v in np.all(empty_row == X, axis=1)]
    return empty_rows


def create_dataset_with_neighbourhood(neighbourhood_size, mode, X_list=None,
                                      y_list=None, X_paths=None,
                                      y_paths=None, save_directory_path=None):
    """
    Creates datasets by mixing all pileups with given neighbourhood_size.

    If X_list and y_list are given, X_paths and y_paths are ignored.

    Datasets are concatenated after extracting neighbourhood_size positions in
    given datasets separately if 'training' mode is selected. If 'inference'
    mode is selected, X_paths are assumed to be paths to different contigs of
    same

    Dataset at i-th position in X_paths should match given labels at i-th
    position in y_paths.

    If save_directory_path is provided, generated pileups are stored in that
    directory.

    :param neighbourhood_size: number of neighbours to use from one size (eg.
        if you set this parameter to 3, it will take 3 neighbours from both
        sides so total number of positions in one example will be 7 -
        counting the middle position)
    :type neighbourhood_size: int
    :param mode: either 'training' or 'inference' string, representing the
        mode for pileups generation
    :type mode: str
    :param X_list: list of X pileup dataset
    :type X_list: list of np.ndarray
    :param y_list: list of y pileup dataset
    :type y_list: list of np.ndarray
    :param X_paths: list of paths to X pileup dataset
    :type X_paths: list of str
    :param y_paths: list of paths to y pileup dataset
    :type y_paths: list of str
    :param save_directory_path: path to directory for storing dataset
    :type save_directory_path: str
    :return:
    :rtype tuple of np.ndarray or tuple of np.array and list of str
    """
    _check_mode(mode)

    # If training mode is selected, all pileups will be concatenated.
    total_pileups = 1 if mode == 'training' else len(X_paths)

    X_save_paths, y_save_paths = None, None
    if save_directory_path is not None:
        X_save_paths, y_save_paths = _generate_save_paths(neighbourhood_size,
                                                          save_directory_path,
                                                          total_pileups)

    X_neighbourhood_list, y_neighbourhood_list = list(), list()

    if not ((X_list is None and y_list is None)
            or (X_paths is None and y_paths is None)):
        raise ValueError('Either X_list and y_list or X_paths and y_paths '
                         'must be provided!')

    if X_list is None and y_list is None:
        X_list = [np.load(X_path) for X_path in X_paths]
        y_list = [np.load(y_path) for y_path in y_paths]

    for pileup_pair, (curr_X, curr_y) in enumerate(zip(X_list, y_list)):
        print('Parsing pileup pair {}'.format(pileup_pair))

        # curr_X, curr_y = np.load(X_path), np.load(y_path)
        # Removing last column which contains everything which was not 'A' nor
        # 'C' nor 'G' nor 'T'.
        curr_y = curr_y[:, :-1]
        new_curr_X, new_curr_y = list(), list()
        empty_rows = _calc_empty_rows(curr_X)

        print('Creating dataset with neighbourhood ...')
        with progressbar.ProgressBar(max_value=curr_X.shape[0]) as progress_bar:
            # TODO(ajuric): Check if this can be speed up.
            for i in range(curr_X.shape[0]):
                progress_bar.update(i)
                if empty_rows[i] == 1:
                    continue  # current row is empty row
                if i < neighbourhood_size or \
                   i >= curr_X.shape[0] - neighbourhood_size:
                    # Current position is not suitable to build an example.
                    continue
                zeros_to_left = np.sum(empty_rows[i - neighbourhood_size:i])
                zeros_to_right = np.sum(
                    empty_rows[i + 1:i + neighbourhood_size + 1])
                if zeros_to_left == 0 and zeros_to_right == 0:
                    new_curr_X.append(
                        curr_X[
                            i - neighbourhood_size:
                            i + neighbourhood_size + 1])
                    new_curr_y.append(curr_y[i])

        X_neighbourhood_list.append(np.array(new_curr_X))
        y_neighbourhood_list.append(np.array(new_curr_y))

    if mode == 'training':
        X_neighbourhood_list = [np.concatenate(X_neighbourhood_list, axis=0)]
        y_neighbourhood_list = [np.concatenate(y_neighbourhood_list, axis=0)]
    else:  # inference mode
        pass  # nothing to do

    if save_directory_path is not None:
        for X_save_path, y_save_path, Xi, yi in zip(
            X_save_paths, y_save_paths, X_neighbourhood_list,
                y_neighbourhood_list):
            np.save(X_save_path, Xi)
            np.save(y_save_path, yi)
        return X_neighbourhood_list, \
               y_neighbourhood_list, \
               X_save_paths, \
               y_save_paths

    return X_neighbourhood_list, y_neighbourhood_list


def _generate_save_paths(neighbourhood_size, save_directory_path,
                         total_pileups):
    """
    Generates a list of save paths for dataset X and y.

    :param neighbourhood_size: number of neighbours to use from one size (eg.
        if you set this parameter to 3, it will take 3 neighbours from both
        sides so total number of positions in one example will be 7 -
        counting the middle position)
    :type neighbourhood_size: int
    :param save_directory_path: path to directory for storing dataset
    :type save_directory_path: str
    :param total_pileups: total number of pileups to be generated a the end;
        determines the number of save paths to be generated
    :type total_pileups: int
    :return: tuple of list of str
    """
    # Creating save file paths.
    X_save_paths = [os.path.join(
        save_directory_path,
        'X-pileups-n{}-{}.npy'.format(neighbourhood_size, i))
                    for i in range(total_pileups)]
    y_save_paths = [os.path.join(
        save_directory_path,
        'y-pileups-n{}-{}.npy'.format(neighbourhood_size, i))
                    for i in range(total_pileups)]

    for X_save_path, y_save_path in zip(X_save_paths, y_save_paths):
        if os.path.exists(X_save_path) or os.path.exists(y_save_path):
            raise ValueError('Pileups already exists in given save '
                             'directory. Either provide other save '
                             'directory or empty this one.')
    return X_save_paths, y_save_paths

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


def _check_mode(mode):
    """
    Checks if given mode is supported: 'training' or 'inference'.

    :param mode: mode to check
    :type mode: str
    :raise ValueError: if selected mode is not supported
    """
    modes = ['training', 'inference']
    if mode not in modes:
        raise ValueError('You must provide either \'training\' or '
                         '\'inference\' mode, but \'{}\' given.'.format(mode))
