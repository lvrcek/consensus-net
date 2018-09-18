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

module_path = '/home/diplomski-rad/consensus-net/src/python/dataset/'
if module_path not in sys.path:
    print('Adding dataset module.')
    sys.path.append(module_path)

import pileups

# model 11
model_path = '/home/diplomski-rad/blade/pb/datasets/n20-racon-hax/model-11.h5'
reads_paths = [
    '/home/data/pacific_biosciences/bacteria/escherichia/coli/NCTC86.fastq',
    '/home/data/pacific_biosciences/bacteria/morganella/morgani/NCTC235.fastq',
    '/home/data/pacific_biosciences/bacteria/salmonella/enterica/NCTC92.fastq',
    '/home/data/pacific_biosciences/bacteria/salmonella/enterica/NCTC129.fastq',
    '/home/data/pacific_biosciences/bacteria/klebsiela/pneumoniae/NCTC204.fastq'
]
assembly_paths = [
    '/home/diplomski-rad/blade/pb/escherichia-coli-NCTC86/iter2.fasta',
    '/home/diplomski-rad/blade/pb/morganela-morgani-NCTC235/iter2.fasta',
    '/home/diplomski-rad/blade/pb/salmonella-enterica-NCTC92/iter2.fasta',
    '/home/diplomski-rad/blade/pb/salmonella-enterica-NCTC129/iter2.fasta',
    '/home/diplomski-rad/blade/pb/klebsiela-pneumoniae-NCTC204-BROKEN/iter2.fasta'
]
reference_paths = [
    '/home/data/pacific_biosciences/bacteria/escherichia/coli/escherichia_coli_reference.fasta',
    '/home/data/pacific_biosciences/bacteria/morganella/morgani/morganella_morganii_reference.fasta',
    '/home/data/pacific_biosciences/bacteria/salmonella/enterica/salmonella_enterica_reference.fasta',
    '/home/data/pacific_biosciences/bacteria/salmonella/enterica/salmonella_enterica_reference.fasta',
    '/home/data/pacific_biosciences/bacteria/klebsiela/pneumoniae/klebsiella_pneumoniae_reference.fasta'
]
sam_file_paths = [
    '/home/diplomski-rad/blade/pb/escherichia-coli-NCTC86/reads-to-asm.sam',
    '/home/diplomski-rad/blade/pb/morganela-morgani-NCTC235/reads-to-asm.sam',
    '/home/diplomski-rad/blade/pb/salmonella-enterica-NCTC92/reads-to-asm.sam',
    '/home/diplomski-rad/blade/pb/salmonella-enterica-NCTC129/reads-to-asm.sam',
    '/home/diplomski-rad/blade/pb/klebsiela-pneumoniae-NCTC204-BROKEN/reads-to-asm.sam'
]
neighbourhood_size = 20
output_dirs = [
    './e-coli-NCTC86-all-contigs-n20-model-11-racon-hax-new',
    './m-morgani-NCTC235-all-contigs-n20-model-11-racon-hax-new',
    './s-enterica-NCTC92-all-contigs-n20-model-11-racon-hax-new',
    './s-enterica-NCTC129-all-contigs-n20-model-11-racon-hax-new',
    './k-pneumoniae-NCTC204-all-contigs-n20-model-11-racon-hax-new'
]
tools_dir = '/home/diplomski-rad/'
racon_hax_output_dir = './racon-hax-tmp'
num_threads = 12

for reads_path, assembly_path, reference_path, sam_file_path, output_dir in \
    zip(reads_paths, assembly_paths, reference_paths, sam_file_paths, output_dirs):
    
    pileup_generator = pileups.RaconMSAGenerator(
        reads_path,
        sam_file_path,
        assembly_path,
        mode='inference',
        tools_dir=tools_dir,
        racon_hax_output_dir=racon_hax_output_dir,
        num_threads=num_threads
    )
    
    inference.make_consensus(
        model_path, 
        reference_path,
        pileup_generator,
        neighbourhood_size,
        output_dir,
        tools_dir)
    
    
model_path = '/home/diplomski-rad/blade/pb/datasets/n20-racon-hax/model-7.h5'
reads_paths = [
    '/home/data/pacific_biosciences/bacteria/escherichia/coli/NCTC86.fastq',
    '/home/data/pacific_biosciences/bacteria/morganella/morgani/NCTC235.fastq',
    '/home/data/pacific_biosciences/bacteria/salmonella/enterica/NCTC92.fastq',
    '/home/data/pacific_biosciences/bacteria/salmonella/enterica/NCTC129.fastq',
    '/home/data/pacific_biosciences/bacteria/klebsiela/pneumoniae/NCTC204.fastq'
]
assembly_paths = [
    '/home/diplomski-rad/blade/pb/escherichia-coli-NCTC86/iter2.fasta',
    '/home/diplomski-rad/blade/pb/morganela-morgani-NCTC235/iter2.fasta',
    '/home/diplomski-rad/blade/pb/salmonella-enterica-NCTC92/iter2.fasta',
    '/home/diplomski-rad/blade/pb/salmonella-enterica-NCTC129/iter2.fasta',
    '/home/diplomski-rad/blade/pb/klebsiela-pneumoniae-NCTC204-BROKEN/iter2.fasta'
]
reference_paths = [
    '/home/data/pacific_biosciences/bacteria/escherichia/coli/escherichia_coli_reference.fasta',
    '/home/data/pacific_biosciences/bacteria/morganella/morgani/morganella_morganii_reference.fasta',
    '/home/data/pacific_biosciences/bacteria/salmonella/enterica/salmonella_enterica_reference.fasta',
    '/home/data/pacific_biosciences/bacteria/salmonella/enterica/salmonella_enterica_reference.fasta',
    '/home/data/pacific_biosciences/bacteria/klebsiela/pneumoniae/klebsiella_pneumoniae_reference.fasta'
]
sam_file_paths = [
    '/home/diplomski-rad/blade/pb/escherichia-coli-NCTC86/reads-to-asm.sam',
    '/home/diplomski-rad/blade/pb/morganela-morgani-NCTC235/reads-to-asm.sam',
    '/home/diplomski-rad/blade/pb/salmonella-enterica-NCTC92/reads-to-asm.sam',
    '/home/diplomski-rad/blade/pb/salmonella-enterica-NCTC129/reads-to-asm.sam',
    '/home/diplomski-rad/blade/pb/klebsiela-pneumoniae-NCTC204-BROKEN/reads-to-asm.sam'
]
neighbourhood_size = 20
output_dirs = [
    './e-coli-NCTC86-all-contigs-n20-model-7-racon-hax-new',
    './m-morgani-NCTC235-all-contigs-n20-model-7-racon-hax-new',
    './s-enterica-NCTC92-all-contigs-n20-model-7-racon-hax-new',
    './s-enterica-NCTC129-all-contigs-n20-model-7-racon-hax-new',
    './k-pneumoniae-NCTC204-all-contigs-n20-model-7-racon-hax-new'
]
tools_dir = '/home/diplomski-rad/'
racon_hax_output_dir = './racon-hax-tmp'
num_threads = 12

for reads_path, assembly_path, reference_path, sam_file_path, output_dir in \
    zip(reads_paths, assembly_paths, reference_paths, sam_file_paths, output_dirs):
    
    pileup_generator = pileups.RaconMSAGenerator(
        reads_path,
        sam_file_path,
        assembly_path,
        mode='inference',
        tools_dir=tools_dir,
        racon_hax_output_dir=racon_hax_output_dir,
        num_threads=num_threads
    )
    
    inference.make_consensus(
        model_path, 
        reference_path,
        pileup_generator,
        neighbourhood_size,
        output_dir,
        tools_dir)