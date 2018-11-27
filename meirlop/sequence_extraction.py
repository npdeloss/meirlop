from math import ceil

import pandas as pd

from Bio import SeqIO

import pybedtools
import pybedtools.featurefuncs
from pybedtools import BedTool

from .io import read_fasta

def get_background(seq, alphabet = 'ACGT', as_counts = False):
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

def get_gc_pct(seq, alphabet_positions = [1,2]):
    bg = get_background(seq)
    gc_pct = 100.0 * sum([bg[pos] for pos in alphabet_positions]) / sum(bg)

    return gc_pct

def get_gc_pct_bin(seq, num_bins = 20, alphabet_positions = [1,2]):
    gc_pct = get_gc_pct(seq, alphabet_positions = alphabet_positions)
    gc_pct_bin = round(round((gc_pct/100) * num_bins) * (100/num_bins))

    return gc_pct_bin

def get_centered_peak_sequences(peaks_df,
                                genome_fa_file,
                                sequence_length = 2000,
                                peak_bed_columns = ['chr','start','end', 'name','score','strand'],
                                start_offset = -1):
    """
    Helper function to get sequences of sequence_length centered on the middle of each peak in peaks_df,
    using the reference in genome_fa_filename
    """

    def widen(feature, width):
        feature.start = max(0, feature.start - int(ceil(width/2)))
        feature.end = feature.end + int(ceil(width/2))
        return feature

    genome_fa_dict, genome_fa_chromsizes_dict = read_fasta(genome_fa_file, chromsizes = True)
    genome_fa_bt = BedTool(genome_fa_file)
    peaks_df_cp = peaks_df.copy()
    start_colname = peak_bed_columns[1]
    peaks_df_cp[start_colname] = peaks_df_cp[start_colname] + start_offset
    peak_bt = BedTool(peaks_df_cp[
        peak_bed_columns
    ].to_string(header = False,
                index = False),
                      from_string=True)

    peak_bt = peak_bt.each(pybedtools.featurefuncs.midpoint).each(widen,
                                                                  width = sequence_length).saveas()
    peak_bt = peak_bt.truncate_to_chrom(genome_fa_chromsizes_dict).saveas()

    peak_sequence_lines = [line.strip()
                           for line
                           in open(peak_bt.sequence(fi = genome_fa_bt,
                                                    tab = True, s = True,
                                                    name = True).seqfn).readlines()]

    peak_sequence_dict = {'('.join(line.strip().split('(')[:-1]): line.split('\t')[1].strip()
                          for line
                          in peak_sequence_lines}
    peak_sequence_bed_df = pd.DataFrame(data = [tuple(entry.split('\t'))
                                                for entry
                                                in str(peak_bt).split('\n')],
                                        columns = ['chr',
                                                   'start',
                                                   'end',
                                                   'name',
                                                   'score',
                                                   'strand'])

    return peak_sequence_dict, peak_sequence_bed_df
