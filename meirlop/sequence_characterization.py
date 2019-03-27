import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from tqdm import tqdm

def get_background(seq, 
                   alphabet = 'ACGT', 
                   as_counts = False):
    counts = {nuc: 0
              for nuc
              in list(alphabet)}
    for nuc in list(seq):
        if nuc in counts:
            counts[nuc] = counts[nuc] + 1
    if as_counts:
        counts_list = [counts[nuc]
                       for nuc
                       in list(alphabet)]
        return counts_list
    total = sum(counts.values())
    bg = tuple([counts[nuc]/total
                for nuc
                in alphabet])

    return bg

def quotient(a, b):
    return int((a - (a % b))/b)

def number_to_pattern(index, 
                      k, 
                      alphabet = list('ACGT')):
    number_to_symbol = alphabet
    
    if k == 1:
        return number_to_symbol[index]
    prefix_index = quotient(index, len(alphabet))
    r = index % len(alphabet)
    symbol = number_to_symbol[r]
    prefix_pattern = number_to_pattern(prefix_index, k - 1)
    return prefix_pattern + symbol

def pattern_to_number(pattern, alphabet = list('ACGT')):
    symbol_to_number = {v:i for i, v in enumerate(alphabet)}
    if len(pattern) == 0:
        return 0
    symbol = pattern[-1]
    prefix = pattern[0:len(pattern)-1]
    return (len(alphabet) 
            * pattern_to_number(prefix) 
            + symbol_to_number[symbol])

def compute_frequency_array(text, k, alphabet = list('ACGT')):
    frequency_array = [0 for i in range(len(alphabet)**k)]
    
    for i in range(len(text) - k + 1):
        pattern = text[i:i+k]
        if len(set(list(pattern)) | set(list(alphabet))) <= len(alphabet):
            j = pattern_to_number(pattern)
            frequency_array[j] = frequency_array[j] + 1
    return frequency_array

def compute_frequency_ratio_array(sequence, k, alphabet = list('ACGT')):
    frequency_array = np.array(compute_frequency_array(sequence, k, alphabet = alphabet))
    frequency_ratio_array = list(frequency_array/np.sum(frequency_array))
    return frequency_ratio_array

def get_frequency_ratio_df(
    sequence_dict, 
    max_k = 2, 
    alphabet = list('ACGT'), 
    n_jobs = 1, 
    remove_redundant = False, 
    progress_wrapper = tqdm):

    get_frequency_ratio_tup = lambda sequence_id, sequence, k: ((k, sequence_id) 
                                                                + tuple(compute_frequency_ratio_array(
                                                                    sequence, k, 
                                                                    alphabet = alphabet)))

    frequency_ratio_tups = [Parallel(n_jobs = 20)(delayed(get_frequency_ratio_tup)(sequence_id, 
                                                                                   sequence, k) 
                                                  for sequence_id, sequence 
                                                  in progress_wrapper(sequence_dict.items())) 
                            for k in [max_k]]
                            # for k in range(1, max_k + 1)]

    get_frequency_ratio_kmers = lambda k: [number_to_pattern(index, k, alphabet) 
                                           for index 
                                           in range(len(alphabet)**k)]

    get_frequency_ratio_df_columns = lambda k: (['k', 'sequence_id'] 
                                                + [f'kmer_ratio_{kmer}'
                                                   for kmer 
                                                   in get_frequency_ratio_kmers(k)])

    get_frequency_ratio_df_from_tup = lambda k: pd.DataFrame(
        frequency_ratio_tups[0], 
        # frequency_ratio_tups[k-1], 
        columns = get_frequency_ratio_df_columns(k))

    frequency_ratio_dfs = [(get_frequency_ratio_df_from_tup(k)
                            .sort_values(by = 'sequence_id')
                            .set_index('sequence_id')
                            .drop(columns = ['k'])) 
                           for k 
                           in [max_k]]
                           # in range(1, max_k + 1)]
    
    frequency_ratio_df = frequency_ratio_dfs[0].reset_index()
    # frequency_ratio_df = pd.concat(frequency_ratio_dfs, axis = 1).reset_index()
    if remove_redundant:
        columns_to_drop = ['kmer_ratio_' + number_to_pattern((len(alphabet)**k)-1, 
                                                             k, 
                                                             alphabet) 
                           for k in [max_k]]
                           # for k in range(1, max_k + 1)]
#         if max_k > 1:
#             columns_to_drop = (columns_to_drop 
#                                + ['kmer_ratio_' + kmer 
#                                   for k in range(2, max_k + 1) 
#                                   for kmer 
#                                   in [number_to_pattern(number, k, alphabet) 
#                                       for number 
#                                       in range(len(alphabet)**k)] 
#                                   if len(set(kmer)) == 1])
        frequency_ratio_df = frequency_ratio_df.drop(columns = columns_to_drop)
    return frequency_ratio_df
