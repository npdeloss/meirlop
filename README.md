# Motif Enrichment In Ranked Lists Of Peaks
This project analyzes the relative enrichment of transcription factor binding motifs found in peaks at the top or bottom of a given ranking/score. 
The design is based on [MOODS](https://github.com/jhkorhonen/MOODS/tree/master/python) and [statsmodels](https://www.statsmodels.org/stable/index.html).

To get started using MEIRLOP, see our [quickstart guide](https://nbviewer.jupyter.org/github/npdeloss/meirlop/blob/master/notebooks/quickstart.ipynb) ([html preview](notebooks/quickstart.html), [jupyter notebook](notebooks/quickstart.ipynb)), which gives you a quick idea of what data formats go into and out of MEIRLOP so you can use it for your own experiments. To run this demo for yourself, you'll need to copy our `notebooks` directory with its `archived_data` subdirectory. For a full usage example that generates and downloads all the data it needs on the fly, see our [walkthrough](https://nbviewer.jupyter.org/github/npdeloss/meirlop/blob/master/notebooks/walkthrough.ipynb) ([html preview](notebooks/walkthrough.html), [jupyter notebook](notebooks/walkthrough.ipynb)), which recreates commands and output for the section "MEIRLOP identifies enriched TF binding motifs in DNase I Hypersensitive Sites" from the MEIRLOP manuscript.

# Installation
To install this package with conda run:
`conda install -c bioconda -c conda-forge -c npdeloss meirlop`  
Assuming bioconda and conda-forge are not in your channels.

# Usage
## MEIRLOP
```
usage: meirlop [-h] (--fa scored_fasta_file | --bed bed_file)
               [--fi reference_fasta_file] [--jobs jobs] [--scan] [--html]
               [--svg] [--sortabs] [--norevcomp] [--kmer max_k] [--length]
               [--gc] [--covariates covariates_table_file]
               [--score score_column] [--pval scan_pval_threshold]
               [--pcount scan_pseudocount]
               [--padj logistic_regression_padj_threshold]
               motif_matrix_file output_dir

Determine enrichment of motifs in a list of scored sequences.

positional arguments:
  motif_matrix_file     A motif matrices file in JASPAR format. As a start,
                        one can be obtained through the JASPAR website at:
                        http://jaspar.genereg.net/downloads/
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
  --svg                 Set this flag to draw motif logos in html output as
                        svg. Useful for figures, but slower for browsers.
  --sortabs             Set this flag to sort enrichment results by the
                        absolute value of the enrichment coefficient.
  --norevcomp           Set this flag to disable searching for reverse
                        complement of motifs.
  --kmer max_k          Set length of kmers to consider during regression.
                        Principal components based on frequency of kmers will
                        be used as a covariates in logistic regression. Set to
                        0 to disable. Default = 2
  --length              Set this flag to incorporate sequence length as a
                        covariate in logistic regression. Multiple covariates
                        will be reduced to principal components.
  --gc                  Set this flag to incorporate GC content as a covariate
                        in logistic regression. Recommend setting --kmer to 0
                        if using --gc. Multiple covariates will be reduced to
                        principal components.
  --covariates covariates_table_file
                        Supply an optional tab-separated file containing
                        additional covariates to incorporate in logistic
                        regression. Columns should be labeled, and the first
                        column should match sequence names in the fasta file.
                        Multiple covariates will be reduced to principal
                        components.
  --score score_column  Name a column in covariates_table_file to use as the
                        sequence score. Use if you don't want to include score
                        in your FASTA file. By default, sequence score is
                        drawn from the FASTA sequence header.
  --pval scan_pval_threshold
                        Set p-value threshold for motif scanning hits on
                        sequences. Defaults to 0.001.
  --pcount scan_pseudocount
                        Set motif matrix pseudocount for motif scanning on
                        sequences. Defaults to 0.001.
  --padj logistic_regression_padj_threshold
                        Set adjusted p-value threshold for logistic regression
                        results. Defaults to 0.05.
```

## MoDiPlot
```
usage: modiplot [-h] [--jobs jobs]
                [--motifslugs motif_slugs_file | --top [n_top_motifs]]
                [--alphafactor [alpha_factor] | --alphaoverride
                alpha_override] [--pointsize pointsize] [--separate]
                [--formats formats [formats ...]] [--fwdcolor fwd_color]
                [--revcolor rev_color] [--width width] [--height height]
                [--dpi dpi] [--nopickle]
                output_dir

Make a plot of the distribution of a motif within a set of scored sequences (a
motif distribution plot, AKA "MoDiPlot") after analysis with meirlop (--scan
is required to generate the necessary scan_results.tsv).

positional arguments:
  output_dir            Read the contents of this meirlop output dir and place
                        plots here.

optional arguments:
  -h, --help            show this help message and exit
  --jobs jobs           The number of jobs for multithreading.
  --motifslugs motif_slugs_file
                        A 2-column tab-separated file with two columns,
                        "motif_id" and "slugname". These column names must be
                        in the header. This table translates motif IDs
                        submitted to meirlop into filename-compatible "slugs"
                        to assign useful filenames to motif plots, and
                        determines which motifs to plot. Mutually exclusive
                        with --top.
  --top [n_top_motifs]  The number of top motif enrichment results from
                        meirlop lr_results to plot. Mutually exclusive with
                        --motifslugs. Default = 10
  --alphafactor [alpha_factor]
                        Factor multiplied against max motif count over
                        position to determine automatically assigned point
                        alpha (transparency) for plotting many motif
                        instances. Mutually exclusive with --alphaoverride.
                        Default = 4.0
  --alphaoverride alpha_override
                        Override automatic alpha calculation with this
                        constant. (See --alphafactor) Mutually exclusive with
                        --alphafactor.
  --pointsize pointsize
                        Size of points to plot.
  --separate            Plot +/- motif orientations separately.
  --formats formats [formats ...]
                        List of output formats for plots. Default: Output
                        plots in SVG and PNG formats.
  --fwdcolor fwd_color  Color of points for motifs in + orientation.
  --revcolor rev_color  Color of points for motifs in - orientation.
  --width width         Width of figures to output, in inches.
  --height height       Height of figures to output, in inches.
  --dpi dpi             DPI of figures to output.
  --nopickle            Do not store motif distributions in a pickle file.
                        They can take a while to write, but might come in
                        handy in the future.
```

## DEPP
```
usage: depp [-h] [--motifslugs motif_slugs_file | --top [n_top_motifs] |
            --all] [--formats formats [formats ...]] [--width width]
            [--height height] [--dpi dpi]
            output_dir

Make a plot of changes (deltas) of motif enrichment across positions within
scored sequences (a delta enrichment positionality plot, AKA "DEPP") after
analysis with meirlop (--scan is required to generate the necessary
scan_results.tsv).

positional arguments:
  output_dir            Read the contents of this meirlop output dir and place
                        plots here.

optional arguments:
  -h, --help            show this help message and exit
  --motifslugs motif_slugs_file
                        A 2-column tab-separated file with two columns,
                        "motif_id" and "slugname". These column names must be
                        in the header. This table translates motif IDs
                        submitted to meirlop into filename-compatible "slugs"
                        to assign useful filenames to motif plots, and
                        determines which motifs to plot. Mutually exclusive
                        with --top.
  --top [n_top_motifs]  The number of top motif enrichment results from
                        meirlop lr_results to plot. Mutually exclusive with
                        --motifslugs. Default = 10
  --all                 Plot all motifs from lr_results. Warning: This can
                        take a while.
  --formats formats [formats ...]
                        List of output formats for plots. Default: Output
                        plots in SVG and PNG formats.
  --width width         Width of figures to output, in inches.
  --height height       Height of figures to output, in inches.
  --dpi dpi             DPI of figures to output.
```
