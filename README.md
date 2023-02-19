# BE_CD69_paper_2022

This repository contains code used for analysis for the preprint "Integrative dissection of gene regulatory elements at base resolution". 

Raw data in fastq format is available on GEO under the following accessions:
 - WT Jurkat resting/stim ATAC-seq: GSE155555(Nasser et al. Nature 2021)
 - WT Jurkat resting/stim RNA-seq: GSE90718(Brignall et al. 2017) 
 - WT CD4 resting.stim: GSE124867(Todd and Cutler 2020)
 - edited Jurkat ATAC, ChIP, RNA-seq: GSE206377(this study)

ATAC-alignment and peak calling
 - ATAC-seq data above was aligned and processed using the ENCODE ATAC-seq pipeline. See `alignment_processing_wdls` for the encode_atac_copy.wdl containing a version of the pipeline w/ minor modifications to return intermediate files. Bigwigs in the figures were generated using DeepTools - see the deeptools_bam_cov.wdl
 
RNA-quantification
 - RNA-seq data was aligned and processed using a custom pipeline - see `alignment_processing_wdls/RNA_processing_PE.wdl`
 - gene quants were generated using Salmon v1.7.0
 
ChIP-alignment and peak calling
 - ChIP-seq data was aligned and processed using the ENCODE ChIP-seq pipeline- see `alignment_processing_wdls/ENCODE_chip_seq.wdl

Remainder of code is grouped by figure or analysis type. 
`data_and_analysis/` contains code for:
 - sample ATAC-seq differential accessibility analysis
 - sample RNA-seq differential expression analysis
 - Enformer base importance calculation: 
  - original model: `enformer_analysis/Enformer_Jurkat_CAGE_gradients_Fig1B_D_Enformer_scores/Enformer_Jurkat_CAGE_gradients_Fig1B_D_Enformer_scores.ipynb`
  - fine tuned model: `enformer_analysis/Enformer_finetuned_gradients_Fig3D_S1D_fine_tuned/Enformer_finetuned_gradients_Fig3D_S1D_fine_tuned.ipynb` 
 - Enformer prediction benchmarking against experimental results for CRISPR-i: 
   - `enformer_analysis/benchmarking_CRISPRi_FigS1D_E`) and BE (`enformer_analysis/benchmarking_BE_FigS2H_I`
 - TFModisco analyses:
   - `enformer_analysis/Enformer_compute_gradients_script_FigS2J.py`, `enformer_analysis/Enformer_TFModisco_FigS2J.py`
 

`enformer_fine_tuning` contains: 
 - training scripts and utilities for fine-tuning an Enformer model on stim vs. resting ATAC-seq data and differential
 - note that the checkpoint containing the weights for the best fine-tuned model is available at 
   - gs://picard-testing-176520/be_paper_finetuning//models/enformer_fine_tuning_230119_LR15e-05_LR20.001_WD15e-07_WD25e-07_WD25e-07_enformer_fine_tuning_230119/final

**required packages**
 - to do
