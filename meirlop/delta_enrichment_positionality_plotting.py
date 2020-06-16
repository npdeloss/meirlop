import argparse
import os
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from scipy.stats import norm
import statsmodels.api as smapi
import statsmodels.formula.api as sm
from statsmodels.stats.multitest import multipletests as mt

from .motif_enrichment import preprocess_lr_df
from .motif_distribution_plotting import load_meirlop_output_dfs, get_motif_id_slugname_df

def main():
    parser = argparse.ArgumentParser(prog = 'depp', 
                                     description = (
                                         'Make a plot of changes (deltas) '
                                         'of motif enrichment '
                                         'across positions within '
                                         'scored sequences '
                                         '(a delta enrichment '
                                         'positionality plot, '
                                         'AKA "DEPP") '
                                         'after analysis with '
                                         'meirlop '
                                         '(--scan is required to generate ' 
                                         'the necessary scan_results.tsv).'
                                     ))
    setup_parser(parser)
    args = parser.parse_args()
    args.func(args)

def setup_parser(parser):
    motif_listing = parser.add_mutually_exclusive_group()
    
    parser.add_argument('output_dir', 
                        metavar = 'output_dir', 
                        type = str, 
                        help = (
                            'Read the contents of this meirlop '
                            'output dir and place plots here.'
                        ))
    
    motif_listing.add_argument('--motifslugs', 
                               metavar = 'motif_slugs_file', 
                               dest = 'motif_slugs_file', 
                               type = argparse.FileType('r'), 
                               help = (
                                   'A 2-column tab-separated file with '
                                   'two columns, '
                                   '"motif_id" and "slugname". '
                                   'These column names '
                                   'must be in the header. '
                                   'This table translates '
                                   'motif IDs submitted to meirlop '
                                   'into filename-compatible "slugs" '
                                   'to assign useful filenames '
                                   'to motif plots, and '
                                   'determines which motifs to plot. '
                                   'Mutually exclusive with --top.'
                               ))
    
    motif_listing.add_argument('--top', 
                               metavar = 'n_top_motifs', 
                               dest = 'n_top_motifs', 
                               type = int, 
                               nargs = '?',
                               default = 10, 
                               const = None, 
                               help = (
                                   'The number of top motif enrichment '
                                   'results from meirlop '
                                   'lr_results to plot. '
                                   'Mutually exclusive with --motifslugs. '
                                   'Default = 10'
                               ))
    
    motif_listing.add_argument('--all', 
                               dest = 'plot_all', 
                               action = 'store_true', 
                               help = (
                                   'Plot all motifs from lr_results. '
                                   'Warning: This can take a while.'
                               ))
    
    parser.add_argument('--nscale', 
                        metavar = 'norm_scale', 
                        dest = 'norm_scale', 
                        type = float, 
                        default = 1.0, 
                        help = (
                            'Set standard deviation '
                            'of the gaussian distribution '
                            'whose pdf is used to weight '
                            'motif positions surrounding '
                            'a motif neighborhood. '
                            'Larger values lead to '
                            '"softer" exclusion '
                            'of motifs from a neighborhood. '
                            'Default = 1.0'
                        ))
    
    parser.add_argument('--formats', 
                        metavar = 'formats', 
                        dest = 'formats', 
                        nargs = '+', 
                        default = ['svg', 'png'], 
                        help = (
                            'List of output formats for plots. '
                            'Default: Output plots '
                            'in SVG and PNG formats.'
                        ))
    
    parser.add_argument('--width', 
                        metavar = 'width', 
                        dest = 'width', 
                        type = float, 
                        default = 10.0, 
                        help = (
                            'Width of figures to output, in inches.'
                        ))
    
    parser.add_argument('--height', 
                        metavar = 'height', 
                        dest = 'height', 
                        type = float, 
                        default = 10.0, 
                        help = (
                            'Height of figures to output, in inches.'
                        ))
    
    parser.add_argument('--dpi', 
                        metavar = 'dpi', 
                        dest = 'dpi', 
                        default = 300,
                        type = int, 
                        help = (
                            'DPI of figures to output.'
                        ))
    
    parser.set_defaults(func = run_depp)
    
def run_depp(args):
    
    if args.motif_slugs_file is not None:
        motif_id_slugname_df = pd.read_csv(
            args.motif_slugs_file, 
            sep = '\t'
        )
    else:
        motif_id_slugname_df = None
    
    (
        delta_enrichment_positionality_profiles_by_motif_id
    ) = plot_delta_enrichment_positionalities(
        args.output_dir, 
        motif_id_slugname_df = motif_id_slugname_df, 
        n_top = args.n_top_motifs, 
        plot_all = args.plot_all,
        norm_scale = args.norm_scale,
        progress_wrapper = tqdm, 
        plot_dpi = args.dpi, 
        figsize = (args.width, args.height), 
        plot_formats = args.formats
    )
    
    outpath_dict = os.path.normpath(args.output_dir + '/delta_enrichment_positionality_profiles_by_motif_id.p')
    with open(outpath_dict, 'wb') as outpath_dict_file:
        pickle.dump(delta_enrichment_positionality_profiles_by_motif_id, outpath_dict_file)
    
    print('Done')
  
    
def plot_delta_enrichment_positionality(
    scan_results_and_lengths_subset_df, 
    lr_input_df, 
    output_prefix, 
    double_negative = True, 
    norm_scale = 1.0, 
    figsize = (10, 10), 
    plot_formats = ['svg', 'png'],
    plot_tight = True,
    plot_dpi = 300, 
    close_fig = True
):
    if plot_tight:
        plt.tight_layout()
    motif_id = list(scan_results_and_lengths_subset_df['motif_id'])[0]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    peak_length = scan_results_and_lengths_subset_df['peak_length'].max()
#     num_points = int(np.round(peak_length / 3.0))
#     norm_locs = np.linspace(
#         -peak_length / 2.0,
#         peak_length / 2.0,
#         num_points
#     )
    norm_locs = np.arange(-peak_length/2.0, peak_length/2.0 + 1, 1.0)
    lr_input_df_subset = [
        lr_input_df['peak_id']
        .isin(list(set(scan_results_and_lengths_subset_df['peak_id'])))
    ]
    preprocessed_lr_df = (
        preprocess_lr_df(
            lr_input_df[
                lr_input_df.columns[:2]
            ],
            lr_input_df[
                [lr_input_df.columns[0]] + 
                list(lr_input_df.columns[2:])
            ]
        )
    )

    norm_loc_columns = []
    for i, norm_loc in enumerate(norm_locs):
        scan_results_and_lengths_subset_df[('norm_pdf', i)] = norm.pdf(
            scan_results_and_lengths_subset_df[
                'instance_peak_center_distance'
            ],
            loc = norm_loc,
            scale = norm_scale
        )
        if double_negative:
            scan_results_and_lengths_subset_df[('norm_pdf', i)] = (
                1.0 - scan_results_and_lengths_subset_df[('norm_pdf', i)]
            )
        norm_loc_columns.append(('norm_pdf', i))

    motif_peak_id_norm_pdf_df = (
        scan_results_and_lengths_subset_df[['peak_id'] + norm_loc_columns]
        .groupby('peak_id')
        .max()
        .reset_index()
    )

    lr_df = (
        motif_peak_id_norm_pdf_df
        .merge(preprocessed_lr_df, how = 'left')
        .fillna(0.0)
    )

    indep_var_cols = list(preprocessed_lr_df.columns[1:])
    score_colname = indep_var_cols[0]
    lr_results = []
    for i, norm_loc in enumerate(norm_locs):
        y = lr_df[('norm_pdf', i)]
        X = lr_df[indep_var_cols]
        model = smapi.OLS(y, X)
        result = model.fit(disp=0)
        coef = result.params[score_colname]
        std_err = result.bse[score_colname]
        pval = result.pvalues[score_colname]
        ci = result.conf_int()
        (
            ci_95_pct_lower,
            ci_95_pct_upper
        ) = (
            ci[0][score_colname],
            ci[1][score_colname]
        )

        # y_score = result.predict(X.values)
        result_tup = (
            coef,
            std_err,
            ci_95_pct_lower,
            ci_95_pct_upper,
            pval,
            result
        )
        lr_results.append(result_tup)

    positional_enrichment_results_df = pd.DataFrame(
        [
            (norm_loc,) + tup[:-1]
            for norm_loc, tup
            in zip(list(norm_locs), lr_results)
        ],
        columns = [
            'Motif Position',
            'Positional Enrichment Coefficient',
            'Standard Error',
            '95% CI Upper',
            '95% CI Lower',
            'P-value'
        ]
    )
    
    if double_negative:
        positional_enrichment_results_df[
            'Positional Enrichment Coefficient'
        ] = (
            0.0 -
            positional_enrichment_results_df[
                'Positional Enrichment Coefficient'
            ]
        )
        positional_enrichment_results_df['95% CI Upper'] = (
            0.0 - positional_enrichment_results_df['95% CI Upper']
        )
        positional_enrichment_results_df['95% CI Lower'] = (
            0.0 - positional_enrichment_results_df['95% CI Lower']
        )
    else:
        original_columns = list(positional_enrichment_results_df.columns)
        positional_enrichment_results_df = (
            positional_enrichment_results_df.rename(columns = {
                '95% CI Upper': '95% CI Lower',
                '95% CI Lower': '95% CI Upper'
            })
        )
        positional_enrichment_results_df = (
            positional_enrichment_results_df[original_columns]
        )
    
    positional_enrichment_results_df['Adjusted P-value'] = mt(
        positional_enrichment_results_df['P-value'],
        method = 'fdr_bh'
    )[1]

    positional_enrichment_results_stack_df = (
        positional_enrichment_results_df[[
            'Motif Position',
            'Positional Enrichment Coefficient',
            '95% CI Upper',
            '95% CI Lower'
        ]]
        .set_index('Motif Position')
        .stack()
        .reset_index()
        .rename(
            columns={
                'level_1': 'Measurement',
                0: 'Coefficient'
            }
        )
    )
    
    sns.lineplot(
        data = positional_enrichment_results_stack_df,
        x = 'Motif Position',
        y = 'Coefficient',
        hue = 'Measurement',
        hue_order = [
            'Positional Enrichment Coefficient',
            '95% CI Upper',
            '95% CI Lower'
        ],
        ax = ax
    )
    plt.title((
        f'Changed enrichment coefficient of motif {motif_id} '
        f'as a function of motif position in sequence'))
    positional_enrichment_results_filepath = os.path.normpath(f'{output_prefix}.tsv')
    positional_enrichment_results_df.to_csv(
        positional_enrichment_results_filepath, 
        sep = '\t', 
        index = False
    )
    
    fig_filepath = f'{output_prefix}'
    
    if plot_tight:
        bbox_inches = 'tight'
    else:
        bbox_inches = None
    
    if (len(plot_formats) > 0) and (fig_filepath != None):
        for fmt in plot_formats:
            fig.savefig(
                os.path.normpath(
                    f'{fig_filepath}.{fmt}'
                ), 
                bbox_inches = bbox_inches, 
                dpi = plot_dpi
            )
    
    if close_fig:
        plt.close(fig)
    
    return (
        fig, 
        positional_enrichment_results_df, 
        os.path.normpath(f'{fig_filepath}.{plot_formats[0]}'), 
        positional_enrichment_results_filepath
    )

def plot_delta_enrichment_positionalities(
    output_dir,
    motif_id_slugname_df = None,
    n_top = 10,
    plot_all = False,
    double_negative = True,
    norm_scale = 1.0,
    plot_dpi = 300,
    figsize = (10, 10),
    plot_formats = ['.svg', '.png'],
    progress_wrapper = tqdm
):
    print(f'Plotting motifs from {output_dir}')
    (
        lr_results_df, 
        lr_input_df, 
        peak_length_df, 
        motif_length_df, 
        scan_results_df, 
        html_logos
    ) = load_meirlop_output_dfs(output_dir)

    if motif_id_slugname_df is None:
        if plot_all:
            motif_id_slugname_df = get_motif_id_slugname_df(lr_results_df)
        else:
            motif_id_slugname_df = get_motif_id_slugname_df(lr_results_df).head(n_top)
    num_motif_ids = motif_id_slugname_df.shape[0]

    print(f'Plotting motif instances for {num_motif_ids} motifs')

    motif_ids = list(motif_id_slugname_df['motif_id'])
    scan_results_df = scan_results_df[scan_results_df['motif_id'].isin(motif_ids)]

    print('Formatting motif scan information')
    scan_results_and_lengths_df = (
        scan_results_df
        .merge(motif_length_df)
        .merge(peak_length_df)
    )
    del scan_results_df

    scan_results_and_lengths_df['instance_center'] = (
        scan_results_and_lengths_df['instance_position'] +
        (scan_results_and_lengths_df['motif_length'] / 2.0)
    )

    scan_results_and_lengths_df['instance_peak_center_distance'] = (
        scan_results_and_lengths_df['instance_center'] -
        (scan_results_and_lengths_df['peak_length'] / 2.0)
    )

    scan_results_and_lengths_df_by_motif_id = dict(list(scan_results_and_lengths_df.groupby('motif_id')))
    del scan_results_and_lengths_df
    slugname_by_motif_id = motif_id_slugname_df.copy().set_index('motif_id')['slugname'].to_dict()
    output_prefix_from_motif_id = lambda motif_id: f'{output_dir}/{slugname_by_motif_id[motif_id]}.depp'

    plot_for_motif = lambda motif_id: (
        plot_delta_enrichment_positionality(
            scan_results_and_lengths_df_by_motif_id[motif_id], 
            lr_input_df, 
            output_prefix_from_motif_id(motif_id),
            double_negative = double_negative,
            norm_scale = norm_scale,
            plot_dpi = plot_dpi, 
            figsize = figsize,
            plot_formats = plot_formats
        )
    )[2:]

    delta_enrichment_positionality_profiles_by_motif_id = {
        motif_id: 
        plot_for_motif(motif_id) 
        for motif_id 
        in progress_wrapper(motif_ids)
    }
    return delta_enrichment_positionality_profiles_by_motif_id

if __name__ == '__main__':
    main()
