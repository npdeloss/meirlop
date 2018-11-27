import pandas as pd
import numpy as np

from Bio import SeqIO
import Bio.motifs.jaspar as jaspar
from Bio.Alphabet.IUPAC import IUPACUnambiguousDNA as dna_alphabet
from Bio.Seq import Seq

import MOODS.scan
import MOODS.tools
import MOODS.parsers

def get_scored_sequence_dict(fasta_filename, description_delim = ' '):
    fasta_records = list(SeqIO.parse(open(fasta_filename),'fasta'))
    sequence_dict = {rec.id: str(rec.seq)
                     for rec
                     in fasta_records}
    description_dict = {rec.id: rec.description.split(description_delim)
                        for rec
                        in fasta_records}
    score_dict = {key: float(val[1])
                  for key, val
                  in description_dict.items()}
    return sequence_dict, score_dict, description_dict

def get_motif_matrices(motifs_filename):
    motifs_bs = jaspar.read(open(motifs_filename, 'r'), format = 'jaspar')
    motif_matrix_dict = {f'{motif.matrix_id} {motif.name}':
                         np.array([list(motif.pwm[nuc])
                                   for nuc
                                   in 'ACGT'])
                         for motif
                         in motifs_bs}
    motif_consensus_dict = {f'{motif.matrix_id} {motif.name}':
                            str(motif.consensus)
                            for motif
                            in motifs_bs}

    return motif_matrix_dict, motif_consensus_dict

def get_motif_fwd_rev_matrices(motif_matrix_dict):
    motif_fwd_matrix_dict = {(motif_id, '+'): motif_matrix
                             for motif_id, motif_matrix
                             in motif_matrix_dict.items()}
    motif_rev_matrix_dict = {(motif_id, '-'): MOODS.tools.reverse_complement(motif_matrix)
                             for motif_id, motif_matrix
                             in motif_matrix_dict.items()}
    motif_fwd_rev_matrix_dict = {**motif_fwd_matrix_dict, **motif_rev_matrix_dict}
    return motif_fwd_rev_matrix_dict

def get_motif_bg_lo_matrix_threshold(motif_matrix, bg, pval = 0.01, pseudocount = 0.001):
    lo_matrix = MOODS.tools.log_odds(motif_matrix, bg, pseudocount)
    threshold = MOODS.tools.threshold_from_p(motif_matrix, bg, pval, len(bg))
    return lo_matrix, threshold

def get_motif_bg_lo_matrices_thresholds(motif_matrix, bg, pval = 0.01, pseudocount = 0.001):
    motif_lo_matrix_threshold_dict = {motif_id: get_motif_bg_lo_matrix_threshold(motif_matrix,
                                                                                 bg,
                                                                                 pval = pval,
                                                                                 pseudocount = pseudocount)
                                      for motif_id, motif_matrix
                                      in motif_matrix.items()}
    motif_lo_matrix_dict = {motif_id: tup[0]
                            for motif_id, tup
                            in motif_lo_matrix_threshold_dict.items()}
    motif_threshold_dict = {motif_id: tup[1]
                            for motif_id, tup
                            in motif_lo_matrix_threshold_dict.items()}
    return motif_lo_matrix_dict, motif_threshold_dict

def scan_motifs(motif_matrix_dict,
                peak_sequence_dict,
                bg = (0.25, 0.25, 0.25, 0,25),
                pval = 0.01,
                pseudocount = 0.001,
                window_size = 7):

    motif_fwd_rev_matrix_dict = get_motif_fwd_rev_matrices(motif_matrix_dict)

    motif_bg_lo_matrices_thresholds = get_motif_bg_lo_matrices_thresholds(
        motif_fwd_rev_matrix_dict,
        bg,
        pval = pval,
        pseudocount = pseudocount)
    motif_lo_matrix_dict, motif_threshold_dict = motif_bg_lo_matrices_thresholds

    peak_ids = list(peak_sequence_dict.keys())
    seqs = [peak_sequence_dict[peak_id]
            for peak_id in peak_ids]

    motif_ids = list(motif_lo_matrix_dict.keys())
    matrices = [motif_lo_matrix_dict[motif_id]
                for motif_id in motif_ids]
    thresholds = [motif_threshold_dict[motif_id]
                  for motif_id in motif_ids]

    scanner = MOODS.scan.Scanner(window_size)
    scanner.set_motifs(matrices, bg, thresholds)

    scans_by_peak = {peak_id: scanner.scan(seq)
                     for peak_id, seq
                     in peak_sequence_dict.items()}
    results = [(peak_id,) + motif_ids[i] + (result.pos, result.score)
               for peak_id, seq
               in peak_sequence_dict.items()
               for i, results
               in enumerate(scanner.scan(seq))
               for result
               in results]

    return results

def format_scan_results(scan_results, dedup_strands = True):
    scan_results_df = pd.DataFrame(data = scan_results,
                                   columns = [
                                   'peak_id',
                                   'motif_id',
                                   'motif_orientation',
                                   'instance_position',
                                   'instance_score'
                                   ])
    if dedup_strands:
        scan_results_df = (scan_results_df
                           .sort_values(by = 'instance_score',
                                        ascending = False)
                           .drop_duplicates([
                           'peak_id',
                           'motif_id',
                           'instance_position'
                           ]))

    motif_peak_set_dict = {motif_id: list(sorted(set(group['peak_id'])))
                           for motif_id, group
                           in scan_results_df.groupby('motif_id')}

    return scan_results_df, motif_peak_set_dict
