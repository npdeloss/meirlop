from timeit import default_timer as timer
import datetime
import logging

from tqdm import tqdm

import pandas as pd
import numpy as np

import statsmodels.api as smapi
import statsmodels.formula.api as sm
from statsmodels.stats.multitest import multipletests as mt

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .sequence_characterization import get_background, get_frequency_ratio_df
from .motif_scanning import scan_motifs_parallel, format_scan_results

def analyze_scored_fasta_data_with_lr(
    sequence_dict, 
    score_dict, 
    motif_matrix_dict, 
    alphabet = list('ACGT'), 
    max_pct_degenerate = 50, 
    pval = 0.001, 
    pseudocount = 0.001, 
    max_k = 2, 
    use_length = False, 
    use_gc = False, 
    user_covariates_df = None, 
    padj_method = 'fdr_bh', 
    padj_thresh = 0.05, 
    min_set_size = 1, 
    max_set_size = np.inf, 
    progress_wrapper = tqdm, 
    n_jobs = 1):
    start = timer()
    print('importing peak data')
    print(datetime.datetime.now())
    
    peak_sequence_dict = {k: v.upper() 
                          for 
                          k, v 
                          in sequence_dict.items() 
                          if (len([nuc 
                                   for nuc in v 
                                   if nuc not in alphabet])
                              /(1.0 * len(v))) < (max_pct_degenerate/100)}
    
    peak_score_dict = {sequence_id: score_dict[sequence_id] 
                       for sequence_id 
                       in peak_sequence_dict.keys()}
    
    peak_score_df = dict_to_df(peak_score_dict, 
                               'peak_id', 
                               'peak_score')
    
    end = timer()
    runtime = end - start
    print(f'{runtime} seconds')
    print(datetime.datetime.now())
    print('scanning for motifs')
    bg = get_background(''.join(peak_sequence_dict.values()), 
                        alphabet = alphabet, 
                        as_counts = False)
    
    scan_results = scan_motifs_parallel(motif_matrix_dict, 
                                        peak_sequence_dict, 
                                        bg = bg, 
                                        pval = pval, 
                                        pseudocount = pseudocount, 
                                        n_jobs = n_jobs, 
                                        progress_wrapper = progress_wrapper)
    (scan_results_df, 
     motif_peak_set_dict) = format_scan_results(scan_results)
    
    covariate_dfs = []
    
    if max_k > 0:

        end = timer()
        runtime = end - start
        print(f'{runtime} seconds')
        print(datetime.datetime.now())
        print('calculating kmer frequency ratios')

        frequency_ratio_df = get_frequency_ratio_df(
            sequence_dict, 
            alphabet = alphabet, 
            max_k = max_k, 
            n_jobs = n_jobs, 
            remove_redundant = True, 
            progress_wrapper = progress_wrapper)
        frequency_ratio_df = (frequency_ratio_df
                              .rename(
                                  columns = {'sequence_id': 'peak_id'}
                              ))
        frequency_ratio_df = frequency_ratio_df.sort_values(by = 'peak_id')
        covariate_dfs.append(frequency_ratio_df)

    if use_length:
        peak_length_dict = {k: len(v) 
                            for k,v 
                            in peak_sequence_dict}
        peak_length_df = dict_to_df(peak_length_dict, 
                                   'peak_id', 
                                   'peak_length')
        peak_length_df = peak_length_df.sort_values(by = 'peak_id')
        covariate_dfs.append(peak_length_df)
    if use_gc:
        end = timer()
        runtime = end - start
        print(f'{runtime} seconds')
        print(datetime.datetime.now())
        print('calculating GC ratios')

        gc_ratio_df = get_frequency_ratio_df(
            sequence_dict, 
            alphabet = alphabet, 
            max_k = 1, 
            n_jobs = n_jobs, 
            remove_redundant = False, 
            progress_wrapper = progress_wrapper)
        gc_ratio_df = (gc_ratio_df
                              .rename(
                                  columns = {'sequence_id': 'peak_id'}
                              ))
        gc_ratio_df = gc_ratio_df.sort_values(by = 'peak_id')
        gc_ratio_df['ratio_gc'] = gc_ratio_df['kmer_ratio_G'] + gc_ratio_df['kmer_ratio_C']
        gc_ratio_df = gc_ratio_df[['peak_id', 'ratio_gc']].copy()
        covariate_dfs.append(gc_ratio_df)
    if user_covariates_df is not None:
        user_covariates_df_cp = user_covariates_df.copy()
        user_covariates_df_columns = list(user_covariates_df_cp.columns)
        user_covariates_df_cp = (user_covariates_df_cp
                                 .rename(columns = {
                                     user_covariates_df_columns[0]: 'peak_id'
                                 }))
        user_covariates_df_cp = (user_covariates_df_cp
                                 .rename(columns = {
                                     colname: 'user_covariate_' + colname
                                     for colname 
                                     in user_covariates_df_columns[1:]
                                 }))
        user_covariates_df_cp = (user_covariates_df_cp
                                 .sort_values(by = 'peak_id'))
        covariate_dfs.append(user_covariates_df_cp)
    
    covariates_df = pd.concat([(df
                                .set_index('peak_id')) 
                               for df 
                               in covariate_dfs], 
                              axis = 1).reset_index()
    
    lr_input_df = peak_score_df.merge(covariates_df)
    end = timer()
    runtime = end - start
    print(f'{runtime} seconds')
    print(datetime.datetime.now())
    print('performing logistic regression')
    
    lr_results_df = analyze_peaks_with_lr(
        peak_score_df, 
        motif_peak_set_dict, 
        covariates_df, 
        padj_method = padj_method, 
        padj_thresh = padj_thresh, 
        min_set_size = min_set_size, 
        max_set_size = max_set_size, 
        progress_wrapper = tqdm)
    
    end = timer()
    runtime = end - start
    print(f'{runtime} seconds')
    print(datetime.datetime.now())
    print('returning logistic regression results')
    
    return lr_results_df, lr_input_df, motif_peak_set_dict, scan_results_df

def dict_to_df(data_dict, key_column, val_column):
    return pd.DataFrame(data = list(data_dict
                                    .items()), 
                        columns = [key_column, 
                                   val_column])

def analyze_peaks_with_lr(peak_score_df,
                          peak_set_dict,
                          peak_covariates_df = None,
                          padj_method = 'fdr_bh',
                          padj_thresh = 0.05,
                          min_set_size = 1,
                          max_set_size = np.inf,
                          progress_wrapper = tqdm):

    lr_df = preprocess_lr_df(peak_score_df, peak_covariates_df)
    peak_id_colname = lr_df.columns[0]
    score_colname = lr_df.columns[1]
    cov_colnames = list(lr_df.columns[2:])
    def process_motif_id(motif_id):
        return ((motif_id,) 
                + compute_logit_regression_for_peak_set(
                    peak_set_dict[motif_id],
                    lr_df,
                    peak_id_colname,
                    score_colname,
                    cov_colnames)[:-1])
    
    valid_peak_ids = [key 
                      for key, val 
                      in peak_set_dict.items() 
                      if (min_set_size <= len(val)) 
                      and (len(val) <= max_set_size)]
    
    result_tups = [process_motif_id(motif_id) 
                   for motif_id 
                   in progress_wrapper(valid_peak_ids)]
    
    results_df = pd.DataFrame(result_tups, 
                              columns = ['motif_id', 
                                         'coef', 
                                         'std_err', 
                                         'ci_95_pct_lower', 
                                         'ci_95_pct_upper', 
                                         'pval', 
                                         'auc'])
    results_df['padj'] = mt(results_df['pval'], 
                            method = padj_method)[1]
    
    results_df['padj_sig'] = ((results_df['padj'] < padj_thresh)
                              .astype(int))
    results_df['abs_coef'] = np.abs(results_df['coef'])
    results_df = results_df.sort_values(by = ['abs_coef', 
                                              'padj_sig'], 
                                        ascending = False)
    
    return results_df

def preprocess_lr_df(peak_score_df, 
                     peak_covariates_df = None):
    if peak_covariates_df is None:
        peak_data_df = peak_score_df.copy()
        peak_id_colname = peak_score_df.columns[0]
        peak_score_colname = peak_data_df.columns[1]
        peak_covariate_colnames = []
    else:
        peak_data_df = peak_score_df.merge(peak_covariates_df)
        peak_id_colname = peak_score_df.columns[0]
        peak_score_colname = peak_data_df.columns[1]
        peak_covariate_colnames = list(peak_covariates_df.columns[1:])

    ss = StandardScaler(with_mean = True, with_std = True)
    X = ss.fit_transform(peak_data_df[[peak_score_colname] 
                                      + peak_covariate_colnames])
    lr_df = pd.DataFrame(X, columns = [peak_score_colname] 
                         + peak_covariate_colnames)
    lr_df.insert(0, 
                 peak_id_colname, 
                 peak_score_df[peak_id_colname])
    # lr_df['intercept'] = 1.0
    return lr_df

def compute_logit_regression_for_peak_set(peak_set,
                                          lr_df,
                                          peak_id_colname,
                                          score_colname,
                                          cov_colnames):
    y = lr_df[peak_id_colname].isin(peak_set)
    indep_var_cols = [score_colname] + cov_colnames
    X = lr_df[indep_var_cols]
    # X = smapi.add_constant(lr_df[indep_var_cols])
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    coef = result.params[score_colname]
    std_err = result.bse[score_colname]
    pval = result.pvalues[score_colname]
    ci = result.conf_int()
    
    (ci_95_pct_lower, 
     ci_95_pct_upper) = (ci[0][score_colname], ci[1][score_colname])
    
    y_score = result.predict(X.values)
    auc = roc_auc_score(y_true = y, 
                        y_score = y_score)
    
    return coef, std_err, ci_95_pct_lower, ci_95_pct_upper, pval, auc, result

