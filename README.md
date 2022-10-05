# BE_CD69_paper_2022

This repository contains code used for analysis and a subset of the figure generation for the preprint "Integrative dissection of gene regulatory elements at base resolution". 

Raw data in fastq format is available on GEO under the following accessions:
 - WT Jurkat resting/stim ATAC-seq: GSE155555(Nasser et al. Nature 2021)
 - WT Jurkat resting/stim RNA-seq: GSE90718(Brignall et al. 2017) 
 - WT CD4 resting.stim: GSE124867(Todd and Cutler 2020)
 - edited Jurkat ATAC, ChIP, RNA-seq: GSE206377(this study)

ATAC-processing
 - ATAC-seq data above was aligned and processed using the ENCODE ATAC-seq pipeline. See `processing scripts` for the encode_atac_copy.wdl containing a version of the pipeline w/ minor modifications to return intermediate files. Bigwigs in the figures were generated using DeepTools - see the deeptools_bam_cov.wdl
 
RNA-processing
 - RNA-seq data was aligned and processed using a custom pipeline - see `processing_scripts/RNA_processing_PE.wdl`
 - gene quants were generated using Salmon v1.7.0
 
ChIP-processing
 - ChIP-seq data was aligned and processed using the ENCODE ChIP-seq pipeline- see `processing_scripts/ENCODE_chip_seq.wdl

Remainder of code is grouped by figure or analysis type. 
`data_and_analysis/` contains code for:
 - ATAC-seq differential accessibility analysis
 - RNA-seq differential expression analysis
 - Enformer base importance score calculation, and TFModisco analyses
 - ChIP-seq motif enrichment analysis
 - motif spacing analysis
 
 `Figure_n` folders contain code to generate the raw figure PDFs for (Figs 1B, 1D, S1D, 2A-C,3A, 3C-F, S3D, S3F, 4A_ii, 4B-F) before further color/sizing adjustments in illustrator. 

`enformer_fine_tuning` contains: 
 - training scripts and utilities for fine-tuning an Enformer model on stim vs. resting ATAC-seq data
 - checkpoint contains the weights for the best fine-tuned model at iteration 24

`references/` contains: 
 - gene annotation files
 - blacklists
 - motif PWMs/annotations used in the paper

**required packages**
 - to do
