# Motif Enrichment In Ranked Lists Of Peaks
This project analyzes the relative enrichment of transcription factor binding motifs found in peaks at the top or bottom of a given ranking/score. 
The design is based on [MOODS](https://github.com/jhkorhonen/MOODS/tree/master/python) and [statsmodels](https://www.statsmodels.org/stable/index.html).

# Installation
To install this package with conda run:
`conda install -c bioconda -c npdeloss meirlop`  
Assuming bioconda is not in your channels.

# Usage
```
usage: meirlop [-h] (--fa scored_fasta_file | --bed bed_file)
               [--fi reference_fasta_file] [--jobs jobs] [--scan] [--html]
               [--kmer max_k] [--length] [--covariates covariates_table_file]
               [--score score_column]
               motif_matrix_file output_dir

Determine enrichment of motifs in a list of scored sequences.

positional arguments:
  motif_matrix_file     A motif matrices file in JASPAR format.
  output_dir            Create this directory and write output to it.

optional arguments:
  -h, --help            show this help message and exit
  --fa scored_fasta_file
                        A scored fasta file, where sequence headers are of
                        form: ">sequence_name sequence_score". Mutually
                        exclusive with --bed.
  --bed bed_file        A 6-column tab-separated bed file, with columns of
                        form: "chrom start end name score strand" Mutually
                        exclusive with --fa. Requires --fi.
  --fi reference_fasta_file
                        A reference fasta file for use with bed_file.
                        Sequences will be extracted from this fasta using
                        coordinates from bed_file. Required if using --bed.
  --jobs jobs           The number of jobs for multithreading. Note:
                        statsmodels may exceed this during logistic
                        regression.
  --scan                Set this flag to write motif scanning results table to
                        output directory.
  --html                Set this flag to write motif html results table to
                        output directory. Includes motif weblogos.
  --kmer max_k          Set maximum length of kmers to consider during
                        regression. Frequency of kmers will be used as a
                        covariate in logistic regression. Default = 2
  --length              Set this flag to incorporate sequence length as a
                        covariate in logistic regression.
  --covariates covariates_table_file
                        Supply an optional tab-separated file containing
                        additional covariates to incorporate in logistic
                        regression. Columns should be labeled, and the first
                        column should match sequence names in the fasta file.
  --score score_column  Name a column in covariates_table_file to use as the
                        sequence score. Use if you don't want to include score
                        in your FASTA file. By default, sequence score is
                        drawn from the FASTA sequence header.
```
