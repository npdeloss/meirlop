import argparse
import sys
import os
import os.path
import pickle

from . import analyze_scored_fasta_data_with_lr, read_scored_fasta, read_motif_matrices

def main():
    parser = argparse.ArgumentParser(prog = 'meirlop', 
                                     description = (
                                         'Determine enrichment '
                                         'of motifs in a list of '
                                         'scored sequences.'
                                     ))
    setup_parser(parser)
    args = parser.parse_args()
    args.func(args)

def setup_parser(parser):
    parser.add_argument('scored_fasta_file', 
                        metavar = 'scored_fasta_file',
                        type = argparse.FileType('r'), 
                        help = (
                            'A scored fasta file, '
                            'where sequence headers are of form: \n'
                            '>sequence_name sequence_score'
                        ))
    
    parser.add_argument('motif_matrix_file', 
                        metavar = 'motif_matrix_file', 
                        type = argparse.FileType('r'), 
                        help = (
                            'A motif matrix file '
                            'in JASPAR format.'
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
    
    parser.add_argument('--kmer', 
                        metavar = 'max_k', 
                        dest = 'max_k', 
                        type = int, 
                        default = 2, 
                        help = (
                            'Set maximum length of kmers to '
                            'consider during regression. '
                            'Frequency of kmers will be used '
                            'as a covariate in logistic regression. '
                            'Default = 2'
                        ))
    
    parser.add_argument('--length', 
                        dest = 'use_length', 
                        action='store_true', 
                        help = (
                            'Set this flag to incorporate '
                            'sequence length as a covariate '
                            'in logistic regression. '
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
                            'the fasta file.'
                        ))
    
    parser.add_argument('--score', 
                        metavar = 'score_column', 
                        dest = 'score_column', 
                        default = None, 
                        type = str, 
                        help = (
                            'Name a column in ' 
                            'covariates_table_column '
                            'to use as the sequence score. '
                            'By default, sequence score is drawn '
                            'from the FASTA sequence header.'
                            'Use if you don\'t want to '
                            'include score in your FASTA file.'
                        ))
    
    parser.set_defaults(func = run_meirlop)
    
def run_meirlop(args):
    scored_fasta_file = args.scored_fasta_file
    motif_matrix_file = args.motif_matrix_file
    save_scan = args.save_scan
    max_k = args.max_k
    use_length = args.use_length
    covariates_table_file = args.covariates_table_file
    output_dir = args.output_dir
    score_column = args.score_column
    n_jobs = args.jobs
    
    user_covariates_df = None
    if covariates_table_file is not None:
        user_covariates_df = pd.read_table(user_covariates_df)
    
    os.environ['OMP_NUM_THREADS'] = f'{n_jobs}'
    os.environ['MKL_NUM_THREADS'] = f'{n_jobs}'
    
    sequence_dict, score_dict = read_scored_fasta(scored_fasta_file)[:-1]
    motif_matrix_dict = read_motif_matrices(motif_matrix_file)[0]
    
    if score_column is not None:
        peak_id_column = user_covariates_df.columns[0]
        score_dict = (user_covariates_df
                      .set_index(peak_id_column)[score_column]
                      .to_dict())
        user_covariates_df = user_covariates_df.drop(columns = [score_column])
    
    (lr_results_df, 
     lr_input_df, 
     motif_peak_set_dict, 
     scan_results_df) = analyze_scored_fasta_data_with_lr(
        sequence_dict, 
        score_dict, 
        motif_matrix_dict, 
        max_k = max_k, 
        use_length = use_length, 
        n_jobs = n_jobs)
    
    outpath = os.path.normpath(output_dir)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    outpath_lr_results = os.path.normpath(output_dir + '/lr_results.tsv')
    outpath_lr_input = os.path.normpath(output_dir + '/lr_input.tsv')
    outpath_motif_peak_set_dict = os.path.normpath(output_dir + '/motif_peak_set_dict.p')
    outpath_scan_results = os.path.normpath(output_dir + '/scan_results.tsv')
    
    lr_results_df.to_csv(outpath_lr_results, sep = '\t', index = False)
    lr_input_df.to_csv(outpath_lr_input, sep = '\t', index = False)
    if save_scan:
        scan_results_df.to_csv(outpath_scan_results, sep = '\t', index = False)
    pickle.dump(motif_peak_set_dict, open(outpath_motif_peak_set_dict, 'wb'))

if __name__ == '__main__':
    main()
