# Motif Enrichment In Ranked Lists Of Peaks
This project analyzes the relative enrichment of transcription factor binding motifs found in peaks at the top or bottom of a given ranking/score. 
The design is based on [MOODS](https://github.com/jhkorhonen/MOODS/tree/master/python) and [statsmodels](https://www.statsmodels.org/stable/index.html).

# Installation
To install this package with conda run:
`conda install -c bioconda -c npdeloss meirlop`  
Assuming bioconda is not in your channels.

# Usage
```
usage: meirlop [-h] [--jobs jobs] [--scan] [--kmer max_k]
               scored_fasta_file motif_matrix_file output_dir

Determine enrichment of motifs in a list of scored sequences.
positional arguments:
  scored_fasta_file  A scored fasta file, where sequence headers are of form:
                     >sequence_name sequence_score
  motif_matrix_file  A motif matrix file in JASPAR format.
  output_dir         Create this directory and write output to it.

optional arguments:
  -h, --help         show this help message and exit
  --jobs jobs        The number of jobs for multithreading. Note: statsmodels
                     may exceed this during logistic regression.
  --scan             Set this flag to write motif scanning results table to
                     output directory.
  --kmer max_k       Set maximum length of kmers to consider during
                     regression. Frequency of kmers will be used as a
                     covariate in logistic regression. Default = 2

```
