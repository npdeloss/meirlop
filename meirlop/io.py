import numpy as np

from Bio import SeqIO
import Bio.motifs.jaspar as jaspar

def read_fasta(fasta_file, chromsizes = False):
    fa_dict = {sequence.id: str(sequence.seq) 
               for sequence 
               in SeqIO.parse(fasta_file,'fasta')}
    if chromsizes:
        fa_chromsizes_dict = {key: (0, len(value)) 
                              for key, value 
                              in fa_dict.items()}
        return fa_dict, fa_chromsizes_dict
    else:
        return fa_dict

def write_scored_fasta(sequence_dict, 
                       score_dict, 
                       fasta_file, 
                       reverse = True, 
                       other_dicts = []):
    sorted_keys = sorted(list(sequence_dict.keys()),
                         key = lambda k: score_dict[k], 
                         reverse = reverse)
    tups = [(key, score_dict[key]) + tuple([other_dict[key]
                                            for other_dict 
                                            in other_dicts])
            for key in sorted_keys]
    rekey = lambda tup: ' '.join(map(str, list(tup)))
    rekeyed_sequence_dict = {rekey(tup): sequence_dict[tup[0]] 
                             for tup in tups}
    rekeys = [rekey(tup) for tup in tups]
    
    fasta_string = '\n'.join(
        [f'>{rekey}\n{rekeyed_sequence_dict[rekey]}' 
         for rekey in rekeys]) + '\n'
    fasta_file.write(fasta_string)
    return fasta_string


def read_scored_fasta(fasta_file, description_delim = ' '):
    fasta_records = list(SeqIO.parse(fasta_file, 'fasta'))
    sequence_dict = {rec.id: str(rec.seq)
                     for rec
                     in fasta_records}
    description_dict = {rec.id: tuple(rec
                                      .description
                                      .split(description_delim))
                        for rec
                        in fasta_records 
                        if len(rec
                               .description
                               .split(description_delim)) > 1}
    score_dict = {key: float(val[1])
                  for key, val
                  in description_dict.items()}
    return sequence_dict, score_dict, description_dict

def read_motif_matrices(motifs_file, alphabet = list('ACGT')):
    motifs_bs = jaspar.read(motifs_file, format = 'jaspar')
    motif_matrix_dict = {f'{motif.matrix_id} {motif.name}':
                         np.array([list(motif.pwm[nuc])
                                   for nuc
                                   in alphabet])
                         for motif
                         in motifs_bs}
    motif_consensus_dict = {f'{motif.matrix_id} {motif.name}':
                            str(motif.consensus)
                            for motif
                            in motifs_bs}

    return motif_matrix_dict, motif_consensus_dict