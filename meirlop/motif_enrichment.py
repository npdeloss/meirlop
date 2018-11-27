import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from tqdm import tqdm

import datetime
from timeit import default_timer as timer

# def analyze_peaks_with_lr(peak_score_df,
#                           peak_set_dict,
#                           peak_covariates_df,
#                           min_set_size = 1,
#                           max_set_size = np.inf,
#                           n_jobs = 1):

def analyze_peaks_with_prerank(peak_score_df,
                               peak_set_dict,
                               peak_strata_df,
                               min_set_size = 1,
                               max_set_size = np.inf,
                               nperm = 10,
                               nshuf = 100,
                               rs = np.random.RandomState(),
                               n_jobs = 1,
                               progress_wrapper = tqdm):
    peak_data_df = peak_score_df.merge(peak_strata_df)
    peak_id_col = peak_data_df.columns[0]
    peak_score_col = peak_data_df.columns[1]
    peak_batch_cols = list(peak_strata_df.columns[1:])


    peak_data_df = peak_data_df.sort_values(by = peak_score_col, ascending = False)

    peak_id_to_peak_idx = {v:i for i, v in enumerate(list(peak_data_df[peak_id_col]))}
    peak_idx_to_peak_id = {v:k for k, v in peak_id_to_peak_idx.items()}

    motif_peak_idx_set_dict = {motif_id: {peak_id_to_peak_idx[peak_id]
                                          for peak_id
                                          in peak_ids}
                               for motif_id, peak_ids
                               in peak_set_dict.items()}

    peak_data_df[peak_id_col] = peak_data_df[peak_id_col].map(peak_id_to_peak_idx).astype(int)
    # start = timer()
    # print(datetime.datetime.now())
    # print('Permuting peak data')
    # print(peak_data_df.shape)
    shuffled_permuted_peak_data = append_shuffled_permuted_peak_data(peak_data_df,
                                                                     score_col = peak_score_col,
                                                                     batch_cols = peak_batch_cols,
                                                                     nperm = nperm,
                                                                     nshuf = nshuf,
                                                                     rs = rs,
                                                                     n_jobs = n_jobs,
                                                                     progress_wrapper = progress_wrapper)
    # end = timer()
    # runtime = end - start
    # print(f'{runtime} seconds')
    # print(datetime.datetime.now())
    # print('Permuted peak data')
    peak_data_shuf_df, peak_id_cols, null_perm_mask_vector = shuffled_permuted_peak_data
    # print(peak_data_shuf_df.shape)
    peak_data_shuf_df = peak_data_shuf_df.sort_values(by = peak_score_col, ascending = False)
    correl_vector = np.abs(peak_data_shuf_df[peak_score_col].values)

    peak_idx_matrix = peak_data_shuf_df[peak_id_cols].values.T
    enrichment_score_results_df = compute_enrichment_scores(motif_peak_idx_set_dict,
                                                            min_set_size,
                                                            max_set_size,
                                                            correl_vector,
                                                            peak_idx_matrix,
                                                            null_perm_mask_vector,
                                                            n_jobs,
                                                            progress_wrapper)
    enrichment_score_results_df['fdr_sig'] = (enrichment_score_results_df['fdr'] < 0.05).astype(int)
    enrichment_score_results_df['abs_nes'] = np.abs(enrichment_score_results_df['nes'])
    enrichment_score_results_df = enrichment_score_results_df.sort_values(by = ['fdr_sig', 'abs_nes'], ascending = False).reset_index(drop = True)
    return enrichment_score_results_df

def applyParallel(dfGrouped, func, n_jobs):
    retLst = Parallel(n_jobs=n_jobs)(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

def append_peak_id_perms(df, suffix, nperm, rs):
    """
    Helper function to append permutations of the peak id as additional columns
    Used to create stratified permutations
    """
    df_cp = df.copy().reset_index(drop = True)
    df_ids = list(df_cp[df_cp.columns[0]].copy(deep=True))
    for i in range(nperm):
        shuffle_colname = f'{df_cp.columns[0]}{suffix}{i}'
        if df_cp.shape[0] > 1:
            df_cp[shuffle_colname] = list(rs.permutation(df_ids))
        else:
            df_cp[shuffle_colname] = df_cp[df_cp.columns[0]]
    return df_cp

def append_peak_id_perms_batched(df, batch_cols, suffix, nperm, rs, n_jobs):
    perm_df = applyParallel(df.groupby(batch_cols), lambda group: append_peak_id_perms(group.copy(), suffix, nperm, rs), n_jobs)
    return perm_df

def append_shuffled_permuted_peak_data(peak_data_df,
                                       score_col = 'peak_score',
                                       batch_cols = ['peak_strata'],
                                       perm_suffix = '_perm_',
                                       nperm = 10,
                                       shuf_suffix = '_shuf_',
                                       nshuf = 100,
                                       rs = np.random.RandomState(),
                                       n_jobs = 1,
                                       progress_wrapper = tqdm):

    # Shuffle to account for ties
    # print(nperm, nshuf, n_jobs)
    peak_data_shufs_df = append_peak_id_perms_batched(peak_data_df,
                                                      batch_cols = [score_col],
                                                      suffix = shuf_suffix,
                                                      nperm = nshuf,
                                                      rs = rs,
                                                      n_jobs = n_jobs)

    get_shuf_colname  = lambda shuf: peak_data_shufs_df.columns[0]+shuf_suffix+str(shuf)
    peak_data_common_cols = [peak_data_shufs_df.columns[0]] + [score_col] + batch_cols



    get_shuf_sub_df = lambda shuf: peak_data_shufs_df[peak_data_common_cols + [get_shuf_colname(shuf)]]

    # Permute, accounting for strata. Only shuffle within strata
    permute_peak_data_shuf = lambda shuf: append_peak_id_perms_batched(peak_data_shufs_df[[get_shuf_colname(shuf)] + peak_data_common_cols].copy(),
                                                                       batch_cols = batch_cols,
                                                                       suffix = perm_suffix,
                                                                       nperm = nperm,
                                                                       rs = rs,
                                                                       n_jobs = n_jobs)
    peak_data_perms_dfs = [permute_peak_data_shuf(shuf) for shuf in progress_wrapper(range(nshuf))]
    merge_and_set_index = lambda shuf: get_shuf_sub_df(shuf).merge(peak_data_perms_dfs[shuf]).set_index(peak_data_common_cols)
    peak_data_perm_df = pd.concat([merge_and_set_index(shuf) for shuf in progress_wrapper(range(nshuf))], axis = 1)

    # Merge with original data
    peak_data_with_null_perms_and_shufs_df = peak_data_df.merge(peak_data_perm_df.reset_index())

    # Get names of columns with permuted data
    peak_id_cols = [peak_data_df.columns[0]+shuf_suffix+str(shuf) for shuf in range(nshuf)]
    peak_id_cols = peak_id_cols + [peak_data_df.columns[0]+shuf_suffix+str(shuf)+perm_suffix+str(perm)
                                   for shuf in range(nshuf)
                                   for perm in range(nperm)]
    null_perm_mask_vector = np.array(([0] * nshuf) + ([1] * (len(peak_id_cols) - nshuf)))

    return peak_data_with_null_perms_and_shufs_df, peak_id_cols, null_perm_mask_vector

def compute_enrichment_score(tag_indicator, correl_vector, null_perm_mask_vector, weighted_score_type = 1, scale = False, single = False):
    weighted_abs_correl_vector = np.abs(correl_vector) ** weighted_score_type

    axis = 1

    no_tag_indicator = 1 - tag_indicator

    n = len(correl_vector)
    n_hit = np.sum(tag_indicator, axis=axis)
    n_miss = n - n_hit

    sum_correl_tag = np.sum(weighted_abs_correl_vector * tag_indicator, axis=axis)

    norm_tag =  1.0/sum_correl_tag
    norm_no_tag = 1.0/n_miss

    res = np.cumsum(((tag_indicator * weighted_abs_correl_vector).T * norm_tag).T - (no_tag_indicator.T * norm_no_tag).T, axis=axis)

    if scale:
        res = res / n
    if single:
        es_vec = res.sum(axis = axis)
    else:
        max_es, min_es =  res.max(axis=axis), res.min(axis=axis)
        es_vec = np.where(np.abs(max_es) > np.abs(min_es), max_es, min_es)

    alt_es_vec = es_vec[np.where(1 - null_perm_mask_vector)]
    avg_es = np.average(alt_es_vec)

    null_es_vec = es_vec[np.where(null_perm_mask_vector)]

    pos_null_es_vec = null_es_vec[np.where(null_es_vec >= 0)]
    neg_null_es_vec = null_es_vec[np.where(null_es_vec < 0)]

    inv_avg_pos_null_es_vec = 1 / np.average(pos_null_es_vec) if len(pos_null_es_vec) > 0 else 0
    inv_avg_neg_null_es_vec = 1 / np.average(neg_null_es_vec) if len(neg_null_es_vec) > 0 else 0

    nes_vec = np.where(es_vec >= 0, es_vec * inv_avg_pos_null_es_vec, -es_vec * inv_avg_neg_null_es_vec)

    alt_nes_vec = nes_vec[np.where(1 - null_perm_mask_vector)]
    null_nes_vec = nes_vec[np.where(null_perm_mask_vector)]

    avg_nes = np.average(alt_nes_vec)

    upper_n_more_extreme = np.sum(np.where(avg_es > pos_null_es_vec, 0, 1))
    lower_n_more_extreme = np.sum(np.where(avg_es < neg_null_es_vec, 0, 1))

    # print(upper_n_more_extreme, pos_null_es_vec.size, lower_n_more_extreme, neg_null_es_vec.size)
    upper_nom_pval = 0 if pos_null_es_vec.size == 0 else upper_n_more_extreme / pos_null_es_vec.size
    lower_nom_pval = 0 if neg_null_es_vec.size == 0 else lower_n_more_extreme / neg_null_es_vec.size

    n_more_extreme = upper_n_more_extreme if avg_es >= 0 else lower_n_more_extreme
    nom_pval = upper_nom_pval if avg_es >= 0 else lower_nom_pval

    return avg_es, avg_nes, nom_pval, n_more_extreme, alt_es_vec, null_es_vec, alt_nes_vec, null_nes_vec, res

def get_tag_indicator_from_peak_idx_set_parallel(peak_idx_set, peak_idx_matrix, n_jobs):
    process_subarray = lambda subarray: [int(peak_idx in peak_idx_set) for peak_idx in subarray]
    return np.array(Parallel(n_jobs=n_jobs)(delayed(process_subarray)(subarray) for subarray in peak_idx_matrix))

def compute_enrichment_scores(motif_peak_idx_set_dict, min_set_size, max_set_size, correl_vector, peak_idx_matrix, null_perm_mask_vector, n_jobs, progress_wrapper = tqdm):

    get_tag_indicator_for_motif_id = lambda motif_id: get_tag_indicator_from_peak_idx_set_parallel(motif_peak_idx_set_dict[motif_id],
                                                                                                   peak_idx_matrix,
                                                                                                   n_jobs)

    get_enrichment_score_result = lambda motif_id: compute_enrichment_score(get_tag_indicator_for_motif_id(motif_id),
                                                                            correl_vector,
                                                                            null_perm_mask_vector)

    get_enrichment_score_result_subset = lambda motif_id: (motif_id, ) + get_enrichment_score_result(motif_id)

    get_enrichment_score_result_tup = lambda motif_id: tuple([item
                                                              for i, item
                                                              in enumerate(get_enrichment_score_result_subset(motif_id))
                                                              if i in {0,1,2,3,4,8}])

    motif_ids_in_size_limit = [motif_id
                               for motif_id
                               in motif_peak_idx_set_dict.keys()
                               if (min_set_size <= len(set(motif_peak_idx_set_dict[motif_id]))) and (len(set(motif_peak_idx_set_dict[motif_id])) <= max_set_size)]

    enrichment_score_results_tups_tmp = [get_enrichment_score_result_tup(motif_id) for motif_id in progress_wrapper(motif_ids_in_size_limit)]
    null_nes_vec = np.array([null_nes for result in enrichment_score_results_tups_tmp for null_nes in result[-1]])
    pos_null_nes_vec = null_nes_vec[np.where(null_nes_vec >= 0)]
    neg_null_nes_vec = null_nes_vec[np.where(null_nes_vec < 0)]
    nes_vec = np.array([result[2] for result in enrichment_score_results_tups_tmp])

    get_fdr_for_pos_nes = lambda nes: (np.sum(np.where(nes > pos_null_nes_vec, 0, 1))/pos_null_nes_vec.size, np.sum(np.where(nes > pos_null_nes_vec, 0, 1)))
    get_fdr_for_neg_nes = lambda nes: (np.sum(np.where(nes < neg_null_nes_vec, 0, 1))/neg_null_nes_vec.size, np.sum(np.where(nes < neg_null_nes_vec, 0, 1)))
    get_fdr_for_nes = lambda nes: get_fdr_for_pos_nes(nes) if nes >= 0 else get_fdr_for_neg_nes(nes)
    fdrs = [get_fdr_for_nes(nes) for nes in progress_wrapper(nes_vec)]
    enrichment_score_results_tups = [tup[:-1] + (fdrs[i]) for i, tup in progress_wrapper(enumerate(enrichment_score_results_tups_tmp))]

    del enrichment_score_results_tups_tmp
    del null_nes_vec
    del pos_null_nes_vec
    del neg_null_nes_vec
    del nes_vec
    del fdrs

    enrichment_score_results_df = pd.DataFrame(enrichment_score_results_tups,
                                               columns = ['motif_id', 'es', 'nes', 'pval', 'n_more_extreme', 'fdr', 'n_all_null_nes_more_extreme'])
    return enrichment_score_results_df
