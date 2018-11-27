import pandas as pd
import numpy as np

from Bio import SeqIO
import Bio.motifs.jaspar as jaspar
from Bio.Alphabet.IUPAC import IUPACUnambiguousDNA as dna_alphabet
from Bio.Seq import Seq

import tempfile

def read_fasta(fasta_file, chromsizes = False):
    """
    Helper function to load sequences from a fasta file into a dict
    """
    fa_dict = {sequence.id: str(sequence.seq) for sequence in SeqIO.parse(fasta_file,'fasta')}
    if chromsizes:
        fa_chromsizes_dict = {key: (0, len(value)) for key, value in fa_dict.items()}
        return fa_dict, fa_chromsizes_dict
    else:
        return fa_dict

def write_fasta(sequence_dict, fasta_file):
    fasta_string = '\n'.join([f'>{key}\n{val}'
                              for key, val
                              in sequence_dict.items()]) + '\n'
    fasta_file.write(fasta_string)
    return fasta_string

def write_scored_fasta(sequence_dict, score_dict, fasta_file, reverse = True, other_dicts = []):
    sorted_keys = sorted(list(sequence_dict.keys()),
                         key = lambda k: score_dict[k], reverse = reverse)
    tups = [(key, score_dict[key]) + tuple([other_dict[key]
                                            for other_dict in other_dicts])
            for key in sorted_keys]
    rekey = lambda tup: ' '.join(map(str, list(tup)))
    rekeyed_sequence_dict = {rekey(tup): sequence_dict[tup[0]] for tup in tups}
    rekeys = [rekey(tup) for tup in tups]
    # Keep sorted by score
    fasta_string = '\n'.join([f'>{rekey}\n{rekeyed_sequence_dict[rekey]}' for rekey in rekeys]) + '\n'
    fasta_file.write(fasta_string)
    return fasta_string
    # return write_fasta(rekeyed_sequence_dict, fasta_file)

def read_scored_fasta(fasta_file, description_delim = ' '):
    fasta_records = list(SeqIO.parse(fasta_file, 'fasta'))
    sequence_dict = {rec.id: str(rec.seq)
                     for rec
                     in fasta_records}
    description_dict = {rec.id: tuple(rec.description.split(description_delim))
                        for rec
                        in fasta_records}
    score_dict = {key: float(val[1])
                  for key, val
                  in description_dict.items()}
    return sequence_dict, score_dict, description_dict

def read_motif_matrices(motifs_file):
    motifs_bs = jaspar.read(motifs_file, format = 'jaspar')
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

def write_gmt(peak_set_dict, gmt_file):
    gmt_string = '\n'.join([f'{motif_id}\t{motif_id}\t'+'\t'.join(peak_set)
                            for motif_id, peak_set
                            in peak_set_dict.items()]) + '\n'
    gmt_file.write(gmt_string)
    return gmt_string

def read_gmt(gmt_file):
    peak_set_dict = {entry[0]: entry[2:]
                     for entry
                     in [line.strip().split('\t')
                         for line
                         in gmt_file.readlines()]}
    return peak_set_dict

def dict_to_df(data_dict, key_column, val_column):
    return pd.DataFrame(data = list(data_dict.items()), columns = [key_column, val_column])
