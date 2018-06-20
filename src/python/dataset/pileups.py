from abc import ABC, abstractmethod
from multiprocessing import Pool
from shutil import rmtree
from time import time

import numpy as np
import os
import progressbar
import pysam
import pysamstats


class PileupGenerator(ABC):
    """
    Base class for generating pileups from read alignments.

    Usage:
        X, y = PileupGenerator(reference_fasta_path, mode).generate()
    """

    def __init__(self, reference_fasta_path, mode, save_directory_path=None):
        """

        Mode must be one of 'training' or 'inference' string indicating pileups
        generation mode. If 'training' is selected, pileups from different
        contigs are all be concatenated. If 'inference' is selected, pileups
        from different contig will be hold separate to enable to make
        consensus with same number of contigs.

        If save_directory_path is provided, generated pileups are stored in that
        directory. Also, paths to those stored files are returned in
        generate_pileups() method.

        :param reference_fasta_path: path to reference stored in fasta file
            format
        :type reference_fasta_path: str
        :param mode: either 'training' or 'inference' string, representing the
            mode for pileups generation
        :type mode: str
        :param save_directory_path: path to directory for storing pileups
        :type save_directory_path: str
        """
        self.reference_fasta_path = reference_fasta_path
        self.mode = mode
        self.save_directory_path = save_directory_path

        self._check_mode()

    def _check_mode(self):
        """
        Checks if given mode is supported: 'training' or 'inference'.

        :raise ValueError: if selected mode is not supported
        """
        modes = ['training', 'inference']
        if self.mode not in modes:
            raise ValueError(
                'You must provide either \'training\' or \'inference\' mode, '
                'but \'{}\' given.'.format(self.mode))

    def _check_save_directory_path(self):
        """
        Checks if given save directory path doesn't exists. If given
        directory doesnt' exists, it creates it.

        :raise ValueError: if given save directory exists
        """
        if self.save_directory_path is not None:
            if os.path.exists(self.save_directory_path):
                raise ValueError(
                    'You must provide non-existing save output directory, '
                    '{} given.'.format(self.save_directory_path))
            else:
                os.makedirs(self.save_directory_path)

    def generate_pileups(self):
        """
        Generates pileups from given alignments.

        :return: Pileups (X) and matching nucleus bases from reference (y) with
            concatenated contigs for inference mode, or separate contigs for
            training mode; also, if save_directory_path is provided, list of
            paths to saved files is returned in tuple (X, y_oh,
            contig_names, X_save_paths, y_save_paths). In both cases list of
            contig names is also returned.
        :rtype tuple of np.ndarray and list of str or tuple of np.array and
            list of str
        """
        self._check_mode()
        self._check_save_directory_path()

        X, y_oh, contig_names = self._generate_pileups()

        total_pileups = len(X)
        if self.mode == 'training':  # training mode
            X, y_oh = [np.concatenate(X, axis=0)], [
                np.concatenate(y_oh, axis=0)]
            total_pileups = 1
        else:  # inference mode
            pass  # nothing to do

        if self.save_directory_path is not None:
            return self._save_data(X, y_oh, contig_names, total_pileups)
        else:
            return X, y_oh, contig_names

    def _save_data(self, X, y_oh, contig_names, total_pileups):
        """
        Saves data to given save_directory path.

        :param X: generated pileups
        :type X: list of np.ndarray
        :param y_oh: generated ground truth
        :type y_oh: list of np.ndarray
        :param contig_names: list of contig names
        :type contig_names: list of str
        :param total_pileups: total number of pileups
        :type total_pileups: int
        :return: generated pileups and ground truths
        :rtype: tuple of np.ndarrays and list of str
        """
        X_save_paths = [
            os.path.join(
                self.save_directory_path,
                'pileups-X-{}.npy'.format(i))
            for i in range(total_pileups)]
        y_save_paths = [os.path.join(self.save_directory_path,
                                     'pileups-y-{}.npy'.format(i))
                        for i in range(total_pileups)]
        for X_save_path, y_save_path, Xi, yi in zip(
                X_save_paths, y_save_paths, X, y_oh):
            np.save(X_save_path, Xi)
            np.save(y_save_path, yi)
        return X, y_oh, X_save_paths, y_save_paths, contig_names

    @abstractmethod
    def _generate_pileups(self):
        """
        Abstract method for generating pileups.

        :return: generated pileups, ground truth and contig names
        :rtype tuple of np.ndarray and list of str
        """
        pass


class PysamstatsNoIndelGenerator(PileupGenerator):
    """
    Pileup generator which uses Pysamstats as backend
    (https://github.com/alimanfoo/pysamstats).

    This version doesn't include indels nor in pileups nor in ground truth.

    One example is:
        x_i = (num_A, num_C, num_G, num_T)
        y_i = ground truth class one-hot encoded (one of A, C, G or T)
    """

    def __init__(self, bam_file_path, reference_fasta_path, mode,
                 save_directory_path=None):
        """
        :param bam_file_path: path to .bam file containing alignments
        :type bam_file_path: str
        :param reference_fasta_path: path to reference stored in fasta file
            format
        :type reference_fasta_path: str
        :param mode: either 'training' or 'inference' string, representing the
            mode for pileups generation
        :type mode: str
        :param save_directory_path: path to directory for storing pileups
        :type save_directory_path: str
        """
        PileupGenerator.__init__(self, reference_fasta_path, mode,
                                 save_directory_path=save_directory_path)
        self.bam_file_path = bam_file_path

    def _generate_pileups(self):
        bam_file = pysam.AlignmentFile(self.bam_file_path)

        info_of_interest = ['A', 'C', 'G', 'T']

        # Last number in shape - 5 - is for letters other than A, C, G and T.
        mapping = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3,
                   't': 3}
        total_options = len(info_of_interest) + 1

        pileups = [np.zeros(
            (bam_file.get_reference_length(contig_name),
             len(info_of_interest)
             )) for contig_name in bam_file.references]

        y_oh = [np.zeros(
            (bam_file.get_reference_length(contig_name),
             total_options
             )) for contig_name in bam_file.references]

        total_length = np.sum(
            [bam_file.get_reference_length(contig_name) for contig_name in
             bam_file.references])
        progress_counter = 0
        contig_names = bam_file.references
        with progressbar.ProgressBar(max_value=total_length) as progress_bar:
            for contig_id, contig_name in enumerate(contig_names):
                for record in pysamstats.stat_variation(
                        bam_file, chrom=contig_name,
                        fafile=self.reference_fasta_path):
                    progress_bar.update(progress_counter)
                    progress_counter += 1

                    curr_position = record['pos']

                    for i, info in enumerate(info_of_interest):
                        pileups[contig_id][curr_position][i] += record[info]

                    y_oh[contig_id][curr_position][
                        mapping.get(record['ref'], -1)] = 1

        return pileups, y_oh, contig_names


class PysamstatsIndelGenerator(PileupGenerator):
    """
    Pileup generator which uses Pysamstats as backend
    (https://github.com/alimanfoo/pysamstats).

    This version includes indels both in pileups and ground truth.

    One example is:
        x_i = (num_A, num_C, num_G, num_T, num_I, num_D)
        y_i = ground truth class one-hot encoded (one of A, C, G, T, I or D)
    """

    def __init__(self, bam_file_path, reference_fasta_path, mode,
                 save_directory_path=None):
        """
        :param bam_file_path: path to .bam file containing alignments
        :type bam_file_path: str
        :param reference_fasta_path: path to reference stored in fasta file
            format
        :type reference_fasta_path: str
        :param mode: either 'training' or 'inference' string, representing the
            mode for pileups generation
        :type mode: str
        :param save_directory_path: path to directory for storing pileups
        :type save_directory_path: str
        """
        PileupGenerator.__init__(self, reference_fasta_path, mode,
                                 save_directory_path=save_directory_path)
        self.bam_file_path = bam_file_path

    def _generate_pileups(self):
        bam_file = pysam.AlignmentFile(self.bam_file_path)

        info_of_interest = ['A', 'C', 'G', 'T', 'insertions', 'deletions']
        indel_positions = [4, 5]

        # Last number in shape - 5 - is for letters other than A, C, G and T.
        mapping = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3,
                   't': 3}
        total_options = len(info_of_interest) + 1

        pileups = [np.zeros(
            (bam_file.get_reference_length(contig_name),
             len(info_of_interest)
             )) for contig_name in bam_file.references]

        y_oh = [np.zeros(
            (bam_file.get_reference_length(contig_name),
             total_options
             )) for contig_name in bam_file.references]

        total_length = np.sum(
            [bam_file.get_reference_length(contig_name) for contig_name in
             bam_file.references])
        progress_counter = 0
        contig_names = bam_file.references
        with progressbar.ProgressBar(max_value=total_length) as progress_bar:
            for contig_id, contig_name in enumerate(contig_names):
                for record in pysamstats.stat_variation(
                        bam_file, chrom=contig_name,
                        fafile=self.reference_fasta_path):
                    progress_bar.update(progress_counter)
                    progress_counter += 1

                    curr_position = record['pos']

                    for i, info in enumerate(info_of_interest):
                        pileups[contig_id][curr_position][i] += record[info]

                    pileup_argmax = np.argmax(
                        pileups[contig_id][curr_position])
                    if pileup_argmax in indel_positions:
                        y_oh[contig_id][curr_position][pileup_argmax] = 1
                    else:
                        y_oh[contig_id][curr_position][
                            mapping.get(record['ref'], -1)] = 1

        return pileups, y_oh, contig_names


class RaconMSAGenerator(PileupGenerator):
    """
    Pileup generator which uses MSA algorithm in order to give moge
    informative pileups.

    As MSA algorithm, Racon tool is used: https://github.com/isovic/racon

    One example is:
        x_i = (num_A, num_C, num_G, num_T, num_D)
        y_i = ground truth class one-hot encoded (one of A, C, G, T, I or D)

    Before pileup generation, Racon tool produces MSA which is stored in
    textual file. Every six lines in that file represent the following:
        - contig
        - number of As
        - number of Cs
        - number of Gs
        - number of Ts
        - number of Ds
    """
    _RACON_CMD = '{}/racon-hax/racon_hax -t {} {} {} {} > {}'

    def __init__(self, reads_path, sam_file_path, reference_fasta_path, mode,
                 tools_dir, racon_hax_output_dir, save_directory_path=None,
                 num_threads=1):
        """

        :param reads_path: path to fastq file containg reads
        :type reads_path: str
        :param sam_file_path: path to .sam file containing alignments
        :type sam_file_path: str
        :param reference_fasta_path: path to reference stored in fasta file
            format
        :type reference_fasta_path: str
        :param mode: either 'training' or 'inference' string, representing the
            mode for pileups generation
        :type mode: str
        :param tools_dir: path to root directory containing tools (like Racon)
        :type tools_dir: str
        :param racon_hax_output_dir: path to directory where Racon output
            will be temporary stored
        :type racon_hax_output_dir: str
        :param save_directory_path: path to directory for storing pileups
        :type save_directory_path: str
        :param num_threads: number of threads for running the Racon tool
        :type num_threads: int
        """
        PileupGenerator.__init__(self, reference_fasta_path, mode,
                                 save_directory_path=save_directory_path)
        self.reads_path = reads_path
        self.sam_file_path = sam_file_path
        self.num_threads = num_threads
        self.tools_dir = tools_dir
        self.racon_hax_output_dir = racon_hax_output_dir

    @staticmethod
    def parse_line(line):
        """
        Parses given line by splitting numbers are casting them to int.

        Given line is one output line from Racon tool.

        :param line: line to be parsed
        :type line: str
        :return: integers parsed from given line
        :rtype: list of int
        """
        return [int(v) for v in line.strip().split()]

    def _parse_racon_hax_output(self, racon_hax_output_path):
        """
        Every pileup has 5 rows A, C, G, T and D. At some column i, number of
        As, Cs, Gs, Ts and Ds correspond to number of those letters at
        position i on reference in pileup.

        :param racon_hax_output_path:
        :return:
        """

        references, pileups = list(), list()
        with Pool(self.num_threads) as pool:
            with open(racon_hax_output_path) as f:
                while True:  # loop for multiple contigs
                    reference = f.readline().strip()
                    if len(reference) == 0:  # EOF
                        break

                    lines = [f.readline() for _ in range(5)]
                    pileup = np.array(pool.map(self.parse_line, lines))

                    references.append(reference)
                    pileups.append(pileup)

        return references, pileups

    @staticmethod
    def _generate_contig_names(num_contigs):
        return ['contig_'.format(i) for i in range(num_contigs)]

    def _generate_pileups(self):
        timestamp = str(int(time()))
        racon_hax_output_path = os.path.join(self.racon_hax_output_dir,
                                             'racon-hex-{}.txt'.format(
                                                 timestamp))
        os.makedirs(self.racon_hax_output_dir)

        # Generate racon_hax output (MSA algorithm).
        os.system(
            RaconMSAGenerator._RACON_CMD.format(
                self.tools_dir,
                self.num_threads,
                self.reads_path,
                self.sam_file_path,
                self.reference_fasta_path,
                racon_hax_output_path
            ))

        # Parse the racon_hax output.
        references, pileups = self._parse_racon_hax_output(
            racon_hax_output_path)

        # Remove racon_hax output.
        rmtree(self.racon_hax_output_dir)

        num_contigs = len(references)
        contig_names = self._generate_contig_names(num_contigs)

        # D - deletions; I - insertions.
        y_classes = ['A', 'C', 'G', 'T', 'I', 'D']
        mapping = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3,
                   't': 3}

        # Parse all contigs.
        total_options = len(y_classes) + 1
        y_oh = [np.zeros(
            (len(reference), total_options)) for reference in references]

        # Transpose pileups to make one example to have shape: (1,
        # num_features).
        pileups = [pileup.T for pileup in pileups]

        total_length = np.sum([len(reference) for reference in references])
        progress_counter = 0
        with progressbar.ProgressBar(max_value=total_length) as progress_bar:
            for contig_id, reference in enumerate(references):
                for position, base in enumerate(reference):
                    progress_bar.update(progress_counter)
                    progress_counter += 1

                    num_Ds = pileups[contig_id][position][4]
                    num_bases = np.max(pileups[contig_id][position][:4])

                    if base == '-':  # insertion
                        y_oh[contig_id][position][4] = 1  # 4 is insertion id
                    elif num_Ds > num_bases:  # deletion
                        y_oh[contig_id][position][5] = 1  # 5 is insertion id
                    else:
                        y_oh[contig_id][position][
                            mapping.get(base, -1)] = 1

        return pileups, y_oh, contig_names
