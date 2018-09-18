#! /usr/bin/env bash

echo $1

if [ -z $2 ]; then
    echo "Input should be <READS_PATH> <REFERENCE_PATH>"
    exit 1
fi

threads=12

../minimap2/minimap2 -t $threads -L100 -Sw5 -m0 $1 $1 > ovl.paf 2>> err

../miniasm/miniasm -f $1 ovl.paf > lay.gfa 2>> err

awk '$1 ~/S/ {print ">"$2"\n"$3}' lay.gfa > iter0.fasta

../minimap/minimap -t $threads iter0.fasta $1 > iter1.paf 2>> err

/usr/bin/time -v -a -o time_memory_new.txt ../racon/build/bin/racon -w 1500 -t $threads $1 iter1.paf iter0.fasta > iter1.fasta 2>> err

../minimap/minimap -t $threads iter1.fasta $1 > iter2.paf 2>> err

/usr/bin/time -v -a -o time_memory_new.txt ../racon/build/bin/racon -w 1500 -t $threads $1 iter2.paf iter1.fasta > iter2.fasta 2>> err

/home/rvaser/mummer/dnadiff $2 iter2.fasta 2>> err

head -n 24 out.report | tail -n 20

#rm ovl.paf
#rm lay.gfa
#rm iter*.fasta
#rm iter*.paf
#rm out.*
#rm err
