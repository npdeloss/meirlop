import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from tqdm import tqdm

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
    peak_batch_cols = peak_strata_df.columns[1:]
    peak_data_df = peak_data_df.sort_values(by = peak_score_col, ascending = False)

    peak_id_to_peak_idx = {v:i for i, v in enumerate(set(list(peak_data_df[peak_id_col])))}
    peak_idx_to_peak_id = {v:k for k, v in peak_id_to_peak_idx.items()}

    motif_peak_idx_set_dict = {motif_id: {peak_id_to_peak_idx[peak_id]
                                          for peak_id
                                          in peak_ids}
                               for motif_id, peak_ids
                               in peak_set_dict.items()}

    peak_data_df[peak_id_col] = peak_data_df[peak_id_col].map(peak_id_to_peak_idx)

    shuffled_permuted_peak_data = append_shuffled_permuted_peak_data(peak_data_df,
                                                                     score_col = peak_score_col,
                                                                     batch_cols = peak_batch_cols,
                                                                     nperm = nperm,
                                                                     nshuf = nshuf,
                                                                     rs = rs,
                                                                     n_jobs = n_jobs)
    peak_data_shuf_df, peak_id_cols, null_perm_mask_vector = shuffled_permuted_peak_data
    peak_data_shuf_df = peak_data_shuf_df.sort_values(by = score_col, ascending = False)
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
                                       nperm = 100,
                                       shuf_suffix = '_shuf_',
                                       nshuf = 100,
                                       rs = np.random.RandomState(),
                                       n_jobs = 1):

    # Shuffle to account for ties
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
    peak_data_perms_dfs = [permute_peak_data_shuf(shuf) for shuf in range(nshuf)]
    merge_and_set_index = lambda shuf: get_shuf_sub_df(shuf).merge(peak_data_perms_dfs[shuf]).set_index(peak_data_common_cols)
    peak_data_perm_df = pd.concat([merge_and_set_index(shuf) for shuf in range(nshuf)], axis = 1)

    # Merge with original data
    peak_data_with_null_perms_and_shufs_df = peak_data_df.merge(peak_data_perm_df.reset_index())

    # Get names of columns with permuted data
    peak_id_cols = [peak_data_df.columns[0]+shuf_suffix+str(shuf) for shuf in range(nshuf)]
    peak_id_cols = peak_id_cols + [peak_data_df.columns[0]+shuf_suffix+str(shuf)+perm_suffix+str(perm)
                                   for shuf in range(nshuf)
                                   for perm in range(nperm)]
    null_perm_mask_vector = np.array(([0] * nshuf) + ([1] * (len(peak_id_cols) - nshuf)))

    return peak_data_with_null_perms_and_shufs_df, peak_id_cols, null_perm_mask_vector

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
                               if (min_set_size <= len(set(peak_set_dict[motif_id]))) and (len(set(peak_set_dict[motif_id])) <= max_set_size)]

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
