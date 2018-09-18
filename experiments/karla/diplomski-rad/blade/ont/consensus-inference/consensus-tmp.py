import numpy as np

import sys

module_path = '/home/diplomski-rad/consensus-net/'
if module_path not in sys.path:
    print('Adding src module.')
    sys.path.append(module_path)
    
import src

module_path = '/home/diplomski-rad/consensus-net/src/python/inference/'
if module_path not in sys.path:
    print('Adding inference module.')
    sys.path.append(module_path)

import inference




model_path = '/home/diplomski-rad/blade/ont/datasets/n20-s-cerevisiae/model-2.h5'
assembly_paths = ['/home/diplomski-rad/blade/ont/s-cerevisiae-r7-288c/iter2.fasta',
                  '/home/diplomski-rad/blade/ont/s-cerevisiae-r9-288c/iter2.fasta',
                  '/home/diplomski-rad/blade/ont/escherichia-coli-ecoli_map006/iter2.fasta']
reference_paths = ['/home/data/oxford_nanopore/saccharomyces/cerevisiae/saccharomyces_cerevisiae_reference.fasta',
                   '/home/data/oxford_nanopore/saccharomyces/cerevisiae/saccharomyces_cerevisiae_reference.fasta',
                   '/home/data/oxford_nanopore/bacteria/escherichia/coli/escherichia_coli_reference.fasta']
bam_file_paths = ['/home/diplomski-rad/blade/ont/s-cerevisiae-r7-288c/reads-to-asm-sorted.bam',
                  '/home/diplomski-rad/blade/ont/s-cerevisiae-r9-288c/reads-to-asm-sorted.bam',
                  '/home/diplomski-rad/blade/ont/escherichia-coli-ecoli_map006/reads-to-asm-sorted.bam']
neighbourhood_size = 20
output_dirs = ['./s-cerv-r7-288c-n20-model-2',
               './s-cerv-r9-288c-n20-model-2',
               './-e-coli-n20-model-2']
tools_dir = '/home/diplomski-rad/'
include_indels = True

for assembly_path, reference_path, bam_file_path, output_dir in zip(assembly_paths, reference_paths, bam_file_paths, output_dirs):
    inference.make_consensus(
        model_path, 
        assembly_path,
        reference_path,
        bam_file_path,
        neighbourhood_size,
        output_dir,
        tools_dir,
        include_indels=include_indels)
    
model_path = '/home/diplomski-rad/blade/ont/datasets/n20-s-cerevisiae/model-12.h5'
assembly_paths = ['/home/diplomski-rad/blade/ont/s-cerevisiae-r7-288c/iter2.fasta',
                  '/home/diplomski-rad/blade/ont/s-cerevisiae-r9-288c/iter2.fasta',
                  '/home/diplomski-rad/blade/ont/escherichia-coli-ecoli_map006/iter2.fasta']
reference_paths = ['/home/data/oxford_nanopore/saccharomyces/cerevisiae/saccharomyces_cerevisiae_reference.fasta',
                   '/home/data/oxford_nanopore/saccharomyces/cerevisiae/saccharomyces_cerevisiae_reference.fasta',
                   '/home/data/oxford_nanopore/bacteria/escherichia/coli/escherichia_coli_reference.fasta']
bam_file_paths = ['/home/diplomski-rad/blade/ont/s-cerevisiae-r7-288c/reads-to-asm-sorted.bam',
                  '/home/diplomski-rad/blade/ont/s-cerevisiae-r9-288c/reads-to-asm-sorted.bam',
                  '/home/diplomski-rad/blade/ont/escherichia-coli-ecoli_map006/reads-to-asm-sorted.bam']
neighbourhood_size = 20
output_dirs = ['./s-cerv-r7-288c-n20-model-12',
               './s-cerv-r9-288c-n20-model-12',
               './-e-coli-n20-model-12']
tools_dir = '/home/diplomski-rad/'
include_indels = True

for assembly_path, reference_path, bam_file_path, output_dir in zip(assembly_paths, reference_paths, bam_file_paths, output_dirs):
    inference.make_consensus(
        model_path, 
        assembly_path,
        reference_path,
        bam_file_path,
        neighbourhood_size,
        output_dir,
        tools_dir,
        include_indels=include_indels)