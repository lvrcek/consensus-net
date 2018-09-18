#! /usr/bin/env bash

# Handling input params.
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters."
    echo "Usage: workflow-pb.sh <READS_PATH> <REFERENCE_PATH> <OUTPUT_DIR>"
    exit 1
fi

if [ -d "$3" ]; then
  echo "OUTPUT_DIR alread exists. Provide non-existing directory."
  exit 1
fi

# Create output directory.
mkdir $3
echo "Output directory created."
# Use all available threads.
# threads=$(nproc)
threads=12
echo "Number of available threads to use: $threads"
# Tools directory.
tools="/home/diplomski-rad"

####################################################
# 1. Create assembly
####################################################
echo "1. Create assembly"
#   1.1. Overlap
echo "1.1. Overlap"
$tools/minimap/minimap \
    -t $threads \
    -L100 \
    -Sw5 \
    -m0 \
     $1 $1 > $3/ovl.paf 2>> $3/err
#   1.2. Layout
echo "1.2. Layout"
$tools/miniasm/miniasm \
    -f \
    $1 $3/ovl.paf > $3/lay.gfa 2>> $3/err
awk '$1 ~/S/ {print ">"$2"\n"$3}' $3/lay.gfa > $3/iter0.fasta
#   1.3. Consensus
echo "1.3. Consensus"
#     1.3.1. First iteration
echo "1.3.1. First iteration"
$tools/minimap/minimap \
    -t $threads \
    $3/iter0.fasta $1 > $3/iter1.paf 2>> $3/err
/usr/bin/time \
    -v \
    -a \
    -o $3/time_memory_new.txt \
     $tools/racon/build/bin/racon \
         -t $threads \
         $1 $3/iter1.paf $3/iter0.fasta > $3/iter1.fasta 2>> $3/err
#    1.3.2. Second iteration
echo "1.3.2. Second iteration"
$tools/minimap/minimap \
    -t $threads \
    $3/iter1.fasta $1 > $3/iter2.paf 2>> $3/err
/usr/bin/time \
    -v \
    -a \
    -o $3/time_memory_new.txt \
    $tools/racon/build/bin/racon \
        -t $threads \
        $1 $3/iter2.paf $3/iter1.fasta > $3/iter2.fasta 2>> $3/err
#   1.4. Assembly summary
echo "1.4. Assembly summary"
$tools/mummer3.23/dnadiff -p $3/out $2 $3/iter2.fasta 2>> $3/err
head -n 24 $3/out.report | tail -n 20

####################################################
# 2. Align reads to reference
####################################################
echo "2. Align reads to reference"
#   2.1. Align reads
echo "2.1. Align reads"
$tools/minimap2/minimap2 \
    -ax map-ont \
    -t $threads \
    $2 $1 > $3/reads-to-ref.sam
#   2.2. Convert .sam to .bam
echo "2.2. Convert .sam to .bam"
$tools/samtools-1.3.1/samtools view \
    -b \
    $3/reads-to-ref.sam > $3/reads-to-ref.bam
#   2.3. Sort alignments
echo "2.3. Sort alignments"
$tools/samtools-1.3.1/samtools sort \
    $3/reads-to-ref.bam > $3/reads-to-ref-sorted.bam
#   2.4. Create index
echo "2.4. Create index"
$tools/samtools-1.3.1/samtools index $3/reads-to-ref-sorted.bam

####################################################
# 3. Align reads to assembly
####################################################
echo "3. Align reads to assembly"
#   3.1. Align reads
echo "3.1. Align reads"
$tools/minimap2/minimap2 \
    -ax map-ont \
    -t $threads \
    $3/iter2.fasta $1 > $3/reads-to-asm.sam
#   3.2. Convert .sam to .bam
echo "3.2. Convert .sam to .bam"
$tools/samtools-1.3.1/samtools view \
    -b \
    $3/reads-to-asm.sam > $3/reads-to-asm.bam
#   3.3. Sort alignments
echo "3.3. Sort alignments"
$tools/samtools-1.3.1/samtools sort \
    $3/reads-to-asm.bam > $3/reads-to-asm-sorted.bam
#   3.4. Create index
echo "3.4. Create index"
$tools/samtools-1.3.1/samtools index $3/reads-to-asm-sorted.bam
