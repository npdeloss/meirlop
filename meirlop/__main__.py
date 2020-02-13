import argparse
import sys
import shlex
import os
import os.path
import pickle
import json
from tqdm import tqdm
import pandas as pd

from . import analyze_scored_fasta_data_with_lr, dict_to_df
from . import read_scored_fasta, read_motif_matrices
from . import get_scored_sequences
from . import get_html_for_lr_results_df

def main():
    parser = argparse.ArgumentParser(prog = 'meirlop', 
                                     description = (
                                         'Determine enrichment '
                                         'of motifs in a list of '
                                         'scored sequences.'
                                     ))
    setup_parser(parser)
    args = parser.parse_args()
    if args.bed_file and (args.reference_fasta_file is None):
        parser.error('--bed requires --fi')
    args.func(args)

def setup_parser(parser):
    input_type = parser.add_mutually_exclusive_group(required = True)
    input_type.add_argument('--fa', 
                            metavar = 'scored_fasta_file', 
                            dest = 'scored_fasta_file', 
                            type = argparse.FileType('r'), 
                            help = (
                                'A scored fasta file, '
                                'where sequence headers are of form: \n'
                                '">sequence_name sequence_score". '
                                'Mutually exclusive with --bed. '
                            ))
    
    input_type.add_argument('--bed', 
                            metavar = 'bed_file', 
                            dest = 'bed_file', 
                            type = argparse.FileType('r'), 
                            help = (
                                'A 6-column tab-separated bed file, '
                                'with columns of form: '
                                '"chrom start end name score strand" '
                                'Mutually exclusive with --fa. '
                                'Requires --fi.'
                            ))
    
    parser.add_argument('--fi', 
                        metavar = 'reference_fasta_file', 
                        dest = 'reference_fasta_file', 
                        type = argparse.FileType('r'), 
                        help = (
                            'A reference fasta file for use with bed_file. '
                            'Sequences will be extracted from this fasta '
                            'using coordinates from bed_file. '
                            'Required if using --bed. '
                        ))
    
    parser.add_argument('motif_matrix_file', 
                        metavar = 'motif_matrix_file', 
                        type = argparse.FileType('r'), 
                        help = (
                            'A motif matrices file '
                            'in JASPAR format. '
                            'As a start, one can be '
                            'obtained through the '
                            'JASPAR website at: '
                            'http://jaspar.genereg.net/downloads/'
                        ))
    
    parser.add_argument('output_dir', 
                        metavar = 'output_dir', 
                        type = str, 
                        help = (
                            'Create this directory and write output to it.'
                        ))
    
    parser.add_argument('--jobs', 
                        metavar = 'jobs', 
                        dest = 'jobs', 
                        type = int, 
                        default = 1, 
                        help = ( 
                            'The number of jobs for multithreading. \n'
                            'Note: statsmodels may exceed this during '
                            'logistic regression.'
                        ))
    
    parser.add_argument('--scan', 
                        dest = 'save_scan', 
                        action='store_true', 
                        help = (
                            'Set this flag to write motif ' 
                            'scanning results table to ' 
                            'output directory.'
                        ))
    
    parser.add_argument('--html', 
                        dest = 'save_html', 
                        action='store_true', 
                        help = (
                            'Set this flag to write motif ' 
                            'html results table to ' 
                            'output directory. '
                            'Includes motif weblogos.'
                        ))
    
    parser.add_argument('--sortabs', 
                        dest = 'sortabs', 
                        action='store_true', 
                        help = (
                            'Set this flag to sort ' 
                            'enrichment results by ' 
                            'the absolute value of '
                            'the enrichment coefficient.'
                        ))
    
    parser.add_argument('--norevcomp', 
                        dest = 'norevcomp', 
                        action='store_true', 
                        help = (
                            'Set this flag to disable ' 
                            'searching for ' 
                            'reverse complement of motifs.'
                        ))
    
    parser.add_argument('--kmer', 
                        metavar = 'max_k', 
                        dest = 'max_k', 
                        type = int, 
                        default = 2, 
                        help = (
                            'Set length of kmers to '
                            'consider during regression. '
                            'Principal components based on '
                            'frequency of kmers will be used '
                            'as a covariates in logistic regression. '
                            'Set to 0 to disable. '
                            'Default = 2'
                        ))
    
    parser.add_argument('--length', 
                        dest = 'use_length', 
                        action='store_true', 
                        help = (
                            'Set this flag to incorporate '
                            'sequence length as a covariate '
                            'in logistic regression. '
                            'Multiple covariates will be reduced '
                            'to principal components.'
                        ))
    
    parser.add_argument('--gc', 
                        dest = 'use_gc', 
                        action='store_true', 
                        help = (
                            'Set this flag to incorporate '
                            'GC content as a covariate '
                            'in logistic regression. '
                            'Recommend setting --kmer to 0 '
                            'if using --gc. '
                            'Multiple covariates will be reduced '
                            'to principal components.'
                        ))
    
    parser.add_argument('--covariates', 
                        metavar = 'covariates_table_file', 
                        dest = 'covariates_table_file', 
                        type = argparse.FileType('r'), 
                        help = (
                            'Supply an optional tab-separated '
                            'file containing '
                            'additional covariates '
                            'to incorporate '
                            'in logistic regression. '
                            'Columns should be labeled, '
                            'and the first column should '
                            'match sequence names in '
                            'the fasta file. '
                            'Multiple covariates will be reduced '
                            'to principal components.'
                        ))
    
    parser.add_argument('--score', 
                        metavar = 'score_column', 
                        dest = 'score_column', 
                        default = None, 
                        type = str, 
                        help = (
                            'Name a column in ' 
                            'covariates_table_file '
                            'to use as the sequence score. '
                            'Use if you don\'t want to '
                            'include score in your FASTA file. '
                            'By default, sequence score is drawn '
                            'from the FASTA sequence header.'
                            
                        ))
    
    parser.add_argument('--pval', 
                        metavar = 'scan_pval_threshold', 
                        dest = 'pval', 
                        default = 0.001, 
                        type = float, 
                        help = (
                            'Set p-value threshold for ' 
                            'motif scanning hits on sequences. '
                            'Defaults to 0.001.'
                        ))
    
    parser.add_argument('--pcount', 
                        metavar = 'scan_pseudocount', 
                        dest = 'pseudocount', 
                        default = 0.001, 
                        type = float, 
                        help = (
                            'Set motif matrix pseudocount for ' 
                            'motif scanning on sequences. '
                            'Defaults to 0.001.'
                        ))
    
    parser.add_argument('--padj', 
                        metavar = 'logistic_regression_padj_threshold', 
                        dest = 'padj_thresh', 
                        default = 0.05, 
                        type = float, 
                        help = (
                            'Set adjusted p-value threshold for ' 
                            'logistic regression results. '
                            'Defaults to 0.05.'
                        ))
    
    parser.set_defaults(func = run_meirlop)
    
def run_meirlop(args):
    scored_fasta_file = args.scored_fasta_file
    bed_file = args.bed_file
    reference_fasta_file = args.reference_fasta_file
    motif_matrix_file = args.motif_matrix_file
    save_scan = args.save_scan
    save_html = args.save_html
    sortabs = args.sortabs
    norevcomp = args.norevcomp
    max_k = args.max_k
    use_length = args.use_length
    use_gc = args.use_gc
    covariates_table_file = args.covariates_table_file
    output_dir = args.output_dir
    score_column = args.score_column
    n_jobs = args.jobs
    pval = args.pval
    pseudocount = args.pseudocount
    padj_thresh = args.padj_thresh
    
    cmdline = 'meirlop ' + ' '.join(map(shlex.quote, sys.argv[1:]))
    
    revcomp = (False==norevcomp)
    
    user_covariates_df = None
    if covariates_table_file is not None:
        user_covariates_df = pd.read_csv(covariates_table_file, sep = '\t')
    
    os.environ['OMP_NUM_THREADS'] = f'{n_jobs}'
    os.environ['MKL_NUM_THREADS'] = f'{n_jobs}'
    
    if scored_fasta_file is not None:
        sequence_dict, score_dict = read_scored_fasta(scored_fasta_file)[:-1]
    elif (bed_file is not None) and (reference_fasta_file is not None):
        sequence_dict, score_dict = get_scored_sequences(bed_file, 
                                                         reference_fasta_file)
    
    peak_length_dict = {k: len(v)
                            for k,v 
                            in sequence_dict.items()}
    peak_length_df = dict_to_df(peak_length_dict, 
                                'peak_id', 
                                'peak_length')
    
    motif_matrix_dict = read_motif_matrices(motif_matrix_file)[0]
    motif_length_dict = {
        motif_id: motif_matrix.shape[1] 
        for motif_id, motif_matrix 
        in motif_matrix_dict.items()
    }
    motif_length_df = dict_to_df(motif_length_dict, 
                                 'motif_id', 
                                 'motif_length')
    if score_column is not None:
        peak_id_column = user_covariates_df.columns[0]
        score_dict = (user_covariates_df
                      .set_index(peak_id_column)[score_column]
                      .to_dict())
        user_covariates_df = user_covariates_df.drop(columns = [score_column])
        if user_covariates_df.shape[1] <= 1:
            user_covariates_df = None
    
    (lr_results_df, 
     lr_input_df, 
     motif_peak_set_dict, 
     scan_results_df) = analyze_scored_fasta_data_with_lr(
        sequence_dict, 
        score_dict, 
        motif_matrix_dict, 
        max_k = max_k, 
        use_length = use_length, 
        use_gc = use_gc, 
        user_covariates_df = user_covariates_df, 
        pval = pval, 
        pseudocount = pseudocount, 
        padj_thresh = padj_thresh, 
        n_jobs = n_jobs, 
        revcomp = revcomp)
    
    outpath = os.path.normpath(output_dir)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    outpath_lr_results = os.path.normpath(output_dir + '/lr_results.tsv')
    outpath_lr_input = os.path.normpath(output_dir + '/lr_input.tsv')
    outpath_peak_length = os.path.normpath(output_dir + '/peak_lengths.tsv')
    outpath_motif_length = os.path.normpath(output_dir + '/motif_lengths.tsv')
    outpath_motif_peak_set_dict = os.path.normpath(output_dir + '/motif_peak_set_dict.p')
    outpath_motif_peak_set_json = os.path.normpath(output_dir + '/motif_peak_set_dict.json')
    outpath_scan_results = os.path.normpath(output_dir + '/scan_results.tsv')
    outpath_html_results = os.path.normpath(output_dir + '/lr_results.html')
    outpath_cmdline_txt = os.path.normpath(output_dir + '/cmdline.txt')
    
    lr_results_df = lr_results_df[['motif_id',
                                   'coef','abs_coef',
                                   'std_err',
                                   'ci_95_pct_lower','ci_95_pct_upper',
                                   'auc',
                                   'pval','padj','padj_sig', 'num_peaks']]
    sortcol = 'coef'
    if sortabs == True:
        sortcol = 'abs_coef'
    print(f'Sorting by padj_sig and {sortcol}')
    lr_results_df = lr_results_df.sort_values(by = ['padj_sig', sortcol], 
                                              ascending = False)
    
    lr_results_df.to_csv(outpath_lr_results, sep = '\t', index = False)
    lr_input_df.to_csv(outpath_lr_input, sep = '\t', index = False)
    peak_length_df.to_csv(outpath_peak_length, sep = '\t', index = False)
    motif_length_df.to_csv(outpath_motif_length, sep = '\t', index = False)
    if save_scan:
        scan_results_df.to_csv(outpath_scan_results, sep = '\t', index = False)
    with open(outpath_motif_peak_set_dict, 'wb') as outpath_motif_peak_set_dict_file:
        pickle.dump(motif_peak_set_dict, outpath_motif_peak_set_dict_file)
    with open(outpath_motif_peak_set_json, 'w') as outpath_motif_peak_set_json_file:
        json.dump(motif_peak_set_dict, outpath_motif_peak_set_json_file)
    with open(outpath_cmdline_txt, 'w') as outpath_cmdline_txtfile:
        outpath_cmdline_txtfile.write(cmdline+'\n')
    
    if save_html:
        print('exporting html report with sequence logos')
        html = get_html_for_lr_results_df(lr_results_df, motif_matrix_dict, output_dir, n_jobs = n_jobs, cmdline = cmdline, sortcol = sortcol)
        with open(outpath_html_results, 'w') as html_file:
            html_file.write(html)

if __name__ == '__main__':
    main()
