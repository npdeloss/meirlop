# Motif Enrichment In Ranked Lists Of Peaks
This project analyzes the relative enrichment of transcription factor binding motifs found in peaks at the top or bottom of a given ranking/score. The design is based on [MOODS](https://github.com/jhkorhonen/MOODS/tree/master/python) and [GSEApy](https://github.com/zqfang/GSEApy).

# Usage
Full usage example available through a [jupyter notebook](notebooks/usage.ipynb)

```
start = timer()
print(datetime.datetime.now())

from meirlop import analyze_peaks_with_prerank
from tqdm import tqdm_notebook

n_jobs = 20
rs = np.random.RandomState(1234)

analysis_results = analyze_peaks_with_prerank(peak_score_df = score_df,
                                              peak_set_dict = motif_peak_set_dict,
                                              peak_strata_df = strata_df,
                                              min_set_size = 1,
                                              max_set_size = np.inf,
                                              nperm = 10,
                                              nshuf = 100,
                                              rs = rs,
                                              n_jobs = n_jobs,
                                              progress_wrapper = tqdm_notebook)

enrichment_score_results_df, shuffled_permuted_peak_data, peak_idx_to_peak_id = analysis_results

end = timer()
runtime = end - start
print(f'{runtime} seconds')
print(datetime.datetime.now())
```
