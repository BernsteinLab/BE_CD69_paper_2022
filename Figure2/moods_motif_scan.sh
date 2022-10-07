#!/bin/bash -l

### make bed file of our ROI
awk '{OFS="\t"}{print "chr12","9764556","9765505"}' > ROI.bed

### extract corresponding sequence from hg38 fasta file
bedtools getfasta -fi ../reference_files/hg38.fa -bed ROI.bed > ROI.fa

#### now motif scan w/ moods, using PFMs from Hocomoco v11
## use lo-bg values from vierstra et al. 2020 and a p-value cutoff of 1.0e-04
python scripts/moods-dna.py --sep ';' -s ~/BE_paper_2022/analyses/Figure2/ROI.fa --p-value 0.0001 --lo-bg 2.977e-01 2.023e-01 2.023e-01 2.977e-01 -m ~/BE_CD69_paper_2022/reference/hocomoco_motifs/pfm/*.pfm -s ~/BE_paper_2022/analyses/Figure2/ROI.fa -o motif_scan.out

### parse resulting file into bed and adjust coordinates appropriately
cat motif_scan.out | sed 's(;(\t(g' | awk '{OFS="\t"}{print "chr12",$4+9764556,$4+9764556+length($7),$6,$2}' > hocomoco_motifs.bed
