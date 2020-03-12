import argparse
import os

import pandas as pd
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from warnings import warn
import datetime
import time
import json

from tqdm import tqdm

from slugify import slugify
from .motif_enrichment import dict_to_df
from scipy.stats import gaussian_kde

from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# progress_wrapper = tqdm

def main():
    parser = argparse.ArgumentParser(prog = 'modiplot', 
                                     description = (
                                         'Make a plot of the distribution '
                                         'of a motif '
                                         'within a set of '
                                         'scored sequences '
                                         '(a motif '
                                         'distribution plot, '
                                         'AKA "MoDiPlot") '
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
    alpha_method = parser.add_mutually_exclusive_group()
    
    parser.add_argument('output_dir', 
                        metavar = 'output_dir', 
                        type = str, 
                        help = (
                            'Read the contents of this meirlop '
                            'output dir and place plots here.'
                        ))
    
    parser.add_argument('--jobs', 
                        metavar = 'jobs', 
                        dest = 'jobs', 
                        type = int, 
                        default = 1, 
                        help = ( 
                            'The number of jobs for multithreading.'
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
    
    alpha_method.add_argument('--alphafactor', 
                              metavar = 'alpha_factor', 
                              dest = 'alpha_factor', 
                              type = float, 
                              default = 4.0, 
                              nargs = '?', 
                              const = None,
                              help = (
                                  'Factor multiplied against max motif '
                                  'count over position to determine '
                                  'automatically assigned point alpha '
                                  '(transparency) '
                                  'for plotting many motif instances. '
                                  'Mutually exclusive with --alphaoverride. '
                                  'Default = 4.0'
                            ))
    
    alpha_method.add_argument('--alphaoverride', 
                              metavar = 'alpha_override', 
                              dest = 'alpha_override',
                              type = float,
                              help = (
                                  'Override automatic alpha '
                                  'calculation with this constant. '
                                  '(See --alphafactor) '
                                  'Mutually exclusive with --alphafactor.'
                              ))
    
    parser.add_argument('--pointsize', 
                    metavar = 'pointsize', 
                    dest = 'pointsize', 
                    default = 2.0, 
                    help = (
                        'Size of points to plot.'
                    ))
    
    parser.add_argument('--separate', 
                        dest = 'separate', 
                        action = 'store_true', 
                        help = (
                            'Plot +/- motif orientations separately.'
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
    
    parser.add_argument('--fwdcolor', 
                        metavar = 'fwd_color', 
                        dest = 'fwd_color', 
                        type = str, 
                        default = 'red', 
                        help = (
                            'Color of points for '
                            'motifs in + orientation.'
                        ))

    parser.add_argument('--revcolor', 
                        metavar = 'rev_color', 
                        dest = 'rev_color', 
                        type = str, 
                        default = 'blue', 
                        help = (
                            'Color of points for '
                            'motifs in - orientation.'
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
                        help = (
                            'DPI of figures to output.'
                        ))
    
    parser.add_argument('--nopickle', 
                        dest = 'no_pickle', 
                        action = 'store_true', 
                        help = (
                            'Do not store motif distributions '
                            'in a pickle file. '
                            'They can take a while to write, '
                            'but might come in handy '
                            'in the future.'
                        ))
#     parser.add_argument('--nojson', 
#                     dest = 'no_json', 
#                     action = 'store_true', 
#                     help = (
#                         'Do not store motif distributions '
#                         'in a JSON file. '
#                         'They can take a while to write, '
#                         'but might come in handy '
#                         'in the future.'
#                     ))
    
    parser.set_defaults(func = run_modiplot)
    
def run_modiplot(args):
    
    if args.motif_slugs_file is not None:
        motif_id_slugname_df = pd.read_csv(
            args.motif_slugs_file, 
            sep = '\t'
        )
    else:
        motif_id_slugname_df = None
    
    (
        figs_by_motif_id_and_orientations_to_plot_parallel, 
        motif_distributions_by_motif
    ) = plot_motif_instances_from_output_dir(
        args.output_dir, 
        motif_id_slugname_df = motif_id_slugname_df, 
        n_top = args.n_top_motifs, 
        n_jobs = args.jobs, 
        progress_wrapper = tqdm, 
        plot_separate = args.separate, 
        plot_dpi = args.dpi, 
        figsize = (args.width, args.height), 
        plot_formats = args.formats, 
        pointsize = args.pointsize, 
        alpha_factor = args.alpha_factor, 
        alpha_override = args.alpha_override, 
        color_by_orientation = {
            '+': args.fwd_color, 
            '-': args.rev_color
        }
    )
    
    outpath_figs_dict = os.path.normpath(args.output_dir + 'figs_by_motif_id_and_orientations.p')
#     outpath_figs_json = os.path.normpath(args.output_dir + 'figs_by_motif_id_and_orientations.json')
    
    outpath_dists_dict = os.path.normpath(args.output_dir + 'motif_distributions_by_motif.p')
#     outpath_dists_json = os.path.normpath(args.output_dir + 'motif_distributions_by_motif.json')
    
    
    with open(outpath_figs_dict, 'wb') as outpath_figs_dict_file:
        pickle.dump(figs_by_motif_id_and_orientations_to_plot_parallel, outpath_figs_dict_file)

#     with open(outpath_figs_json, 'w') as outpath_figs_json_file:
#         json.dump(figs_by_motif_id_and_orientations_to_plot_parallel, outpath_figs_json_file)

    if args.no_pickle == False:
        print('Writing underlying figure data to files.')
        print('Could come in handy someday.')
        with open(outpath_dists_dict, 'wb') as outpath_dists_dict_file:
            pickle.dump(motif_distributions_by_motif, outpath_dists_dict_file)
#     if args.no_json == False:
#         with open(outpath_dists_json, 'w') as outpath_dists_json_file:
#             json.dump(motif_distributions_by_motif, outpath_dists_json_file)
    
    print('Done')
    

def load_meirlop_output_dfs(output_dir):
    outpath_lr_results = os.path.normpath(output_dir + '/lr_results.tsv')
    outpath_lr_input = os.path.normpath(output_dir + '/lr_input.tsv')
    outpath_peak_length = os.path.normpath(output_dir + '/peak_lengths.tsv')
    outpath_motif_length = os.path.normpath(output_dir + '/motif_lengths.tsv')
    outpath_scan_results = os.path.normpath(output_dir + '/scan_results.tsv')
    outpath_html_logos_json = os.path.normpath(output_dir + '/html_logos.json')
    
    lr_results_df = pd.read_csv(outpath_lr_results, sep = '\t')
    lr_input_df = pd.read_csv(outpath_lr_input, sep = '\t')
    peak_length_df = pd.read_csv(outpath_peak_length, sep = '\t')
    motif_length_df = pd.read_csv(outpath_motif_length, sep = '\t')
    
    try:
        scan_results_df = pd.read_csv(outpath_scan_results, sep = '\t')
    except:
        warn(f'The file {outpath_scan_results} could not be loaded. Try rerunning meirlop with the --scan flag enabled')
        results = (lr_results_df, lr_input_df, peak_length_df, motif_length_df)
        return results
    
    try:
        with open(outpath_html_logos_json) as outpath_html_logos_json_file:
            html_logos = json.loads(outpath_html_logos_json_file.read())
    except:
        warn(f'The file {outpath_html_logos_json} could not be loaded. Try rerunning meirlop with the --html flag enabled')
        results = (lr_results_df, lr_input_df, peak_length_df, motif_length_df, scan_results_df)
        return results
    
    results = (lr_results_df, lr_input_df, peak_length_df, motif_length_df, scan_results_df, html_logos)
    return results

def get_motif_id_slugname_df(lr_results_df):
    motif_ids_by_rank = list(lr_results_df['motif_id'])
    motif_id_to_slugname = {motif_id: slugify(f'rank {rank} {motif_id}', separator = '_') for rank, motif_id in enumerate(motif_ids_by_rank, 1)}
    motif_id_slugname_df = dict_to_df(motif_id_to_slugname, 'motif_id', 'slugname')
    return motif_id_slugname_df

def precompute_motif_dfs(
    lr_input_df, 
    scan_results_df, 
    motif_length_df, 
    peak_length_df, 
    progress_wrapper = tqdm
):
    max_peak_length = peak_length_df['peak_length'].max()
    motif_length_dict = motif_length_df.set_index('motif_id')['motif_length'].to_dict()
    peak_length_dict = peak_length_df.set_index('peak_id')['peak_length'].to_dict()
    scan_results_df['instance_position_center'] = (
        scan_results_df['instance_position'] - 
        (scan_results_df['peak_id'].map(peak_length_dict) / 2.0) + 
        (scan_results_df['motif_id'].map(motif_length_dict) / 2.0)
    )
    scan_results_df['instance_position_center_int'] = scan_results_df['instance_position_center'].astype(int)
    lr_input_df['peak_score_rank'] = lr_input_df['peak_score'].rank(method = 'first')
    lr_input_df['peak_score_rank_int'] = lr_input_df['peak_score_rank'].astype(int) - 1
    max_peak_score_rank = lr_input_df['peak_score_rank_int'].max()
    
    scan_results_df_gb_motif_id = scan_results_df.groupby('motif_id')
    peak_cols = ['peak_id', 'peak_score', 'peak_score_rank', 'peak_score_rank_int']
    peak_and_motif_df_by_motif_id = {
        motif_id: 
        (
            lr_input_df[peak_cols]
            .merge(
                scan_results_df_gb_motif_id
                .get_group(motif_id)
                .copy()
            )
        ) 
        for motif_id 
        in progress_wrapper(
            list(
                set(
                    scan_results_df['motif_id']
                )
            )
        )
    }
    sorted_peak_score_rank_df = lr_input_df[['peak_score', 'peak_score_rank_int']].sort_values(by = 'peak_score_rank_int')
    return peak_and_motif_df_by_motif_id, sorted_peak_score_rank_df, max_peak_length, max_peak_score_rank

def two_column_df_to_rolling_mean(df, min_index = None, max_index = None, window = 3, center = True):
    col1 = df.columns[0]
    col2 = df.columns[1]
    if min_index is None:
        min_index = int(np.floor(df[col1].min()))
    if max_index is None:
        max_index = int(np.ceil(df[col1].max()))
    rolling_mean_df = (
        df[[col1, col2]]
        .groupby(col1)
        .count()
        .reset_index()
        .merge(
        pd.DataFrame(
            {col1 : 
             list(range(min_index, max_index + 1))
            }), 
        how = 'outer')
        .fillna(0.0)
        .sort_values(by = col1)
        .set_index(col1)
        .rolling(window, center = center)
        .mean()
        .dropna()
        .reset_index()
    )
    rolling_mean_df[col1]
    return rolling_mean_df

def compute_motif_distributions(
    peak_and_motif_df, 
    max_peak_length, 
    max_peak_score_rank, 
    window = 3, 
    num_kde_points = 1000, 
    progress_wrapper = tqdm
): 
    peak_and_motif_df_by_orientation = {
        orientation: 
        peak_and_motif_df[
            peak_and_motif_df['motif_orientation'] == orientation
        ] 
        for orientation 
        in set(list(peak_and_motif_df['motif_orientation']))
    }
    
    rolling_mean_on_motif_position_by_orientation = {
        orientation: 
        two_column_df_to_rolling_mean(
            (df[['instance_position_center_int', 'motif_id']]
             .rename(columns = {'motif_id': 'motif_count'})), 
            min_index = -int(np.floor(max_peak_length / 2.0)), 
            max_index = int(np.ceil(max_peak_length / 2.0)), 
            window = window, 
            center = True
        ) 
        for orientation, df 
        in peak_and_motif_df_by_orientation.items()
    }
    
    yvals_by_orientation = {}
    xvals_by_orientation = {}
    for orientation, df in peak_and_motif_df_by_orientation.items():
        yvals_by_orientation[orientation] = np.linspace(0, max_peak_score_rank+1, num_kde_points)
        if df.shape[0] > 1:
            kernel = gaussian_kde(df['peak_score_rank_int'])
            xvals_by_orientation[orientation] = kernel(yvals_by_orientation[orientation])
        else:
            xvals_by_orientation[orientation] = yvals_by_orientation[orientation] * 0.0
    
    motif_rank_kde_by_orientation = {
        orientation: 
        pd.DataFrame({'peak_score_rank_int': yvals_by_orientation[orientation], 
                      'motif_density': xvals_by_orientation[orientation]
                     }) 
        for orientation 
        in list(peak_and_motif_df_by_orientation.keys())
    }
    
    return peak_and_motif_df_by_orientation, rolling_mean_on_motif_position_by_orientation, motif_rank_kde_by_orientation

def plot_motif_instances_single(
    peak_and_motif_df_by_orientation, 
    rolling_mean_on_motif_position_by_orientation, 
    motif_rank_kde_by_orientation, 
    sorted_peak_score_rank_df, 
    title = 'Plot of motif locations in ranked peaks', 
    filename = 'motif_instances_plot', 
    orientations_to_plot = ['+', '-'], 
    color_by_orientation = {'+': 'red', '-': 'blue'},
    figsize = (12.5, 10.0), 
    pointsize = 2.0, 
    alpha_factor = 4.0, 
    alpha_override = None, 
    plot_formats = ['svg', 'ps', 'png'],
    plot_tight = True,
    plot_dpi = 300, 
    close_fig = True
):
    if alpha_override == None:
        alpha = np.min([np.max([alpha_factor/df['motif_count'].max() for df in rolling_mean_on_motif_position_by_orientation.values()]), 1.0])
    else:
        alpha = alpha_override
    
    if plot_tight:
        plt.tight_layout()
    fig = plt.figure(figsize = figsize)
    gs = GridSpec(ncols=5,nrows=4, figure = fig)
    ax_score_rank = fig.add_subplot(gs[1:4,0])
    ax_rank = fig.add_subplot(gs[1:4,4])
    ax_motifs = fig.add_subplot(gs[1:4,1:4])
    ax_pos = fig.add_subplot(gs[0,1:4])
    ax_motifs.get_shared_x_axes().join(ax_motifs, ax_pos)
    ax_motifs.get_shared_y_axes().join(ax_motifs, ax_rank)
    ax_motifs.get_shared_y_axes().join(ax_motifs, ax_score_rank)
    ax_pos.set_ylabel('Motif Count')
    ax_rank.set_xlabel('Motif Density')
    ax_motifs.set_xlabel('Motif Position')
    ax_score_rank.set_ylabel('Peak Score Rank')
    ax_score_rank.set_xlabel('Peak Score')
    ax_motifs.set_yticklabels([])
    ax_rank.set_yticklabels([])
    
    fig.suptitle(title)
    ax_score_rank.plot(sorted_peak_score_rank_df['peak_score'], sorted_peak_score_rank_df['peak_score_rank_int'], color = 'black')
    for orientation, df in rolling_mean_on_motif_position_by_orientation.items():
        if orientation in orientations_to_plot:
            ax_pos.plot(df['instance_position_center_int'], df['motif_count'], '-', color = color_by_orientation[orientation])
    for orientation, df in motif_rank_kde_by_orientation.items():
        if orientation in orientations_to_plot:
            ax_rank.plot(df['motif_density'], df['peak_score_rank_int'], '-', color = color_by_orientation[orientation])
    for tick in ax_rank.get_xticklabels():
        tick.set_rotation(45)
    handles = []
    for orientation, df in peak_and_motif_df_by_orientation.items():
        if orientation in orientations_to_plot:
            ax_motifs.scatter(
                df['instance_position_center_int'], 
                df['peak_score_rank_int'], 
                color = color_by_orientation[orientation], 
                alpha = alpha, 
                s = pointsize)
            handle = mpatches.Patch(
                color = color_by_orientation[orientation], 
                label = orientation
            )
            handles.append(handle)
    ax_legends = fig.add_subplot(gs[0,4])
    ax_legends.axis('off')
    ax_legends.legend(
        handles = handles, 
        loc = 'lower left', 
        title = 'Orientation'
    )

    
    if plot_tight:
        bbox_inches = 'tight'
    else:
        bbox_inches = None
        
    if (len(plot_formats) > 0) and (filename != None):
        for fmt in plot_formats:
            fig.savefig(os.path.normpath(f'{filename}.{fmt}'), bbox_inches = bbox_inches, dpi = plot_dpi)
    
    if close_fig:
        plt.close(fig)
        fmt = plot_formats[0]
        return os.path.normpath(f'{filename}.{fmt}')
    
    return fig
    
    

orientations_to_filename_substr = lambda orientations: {
    ('+','-'): 'both', 
    ('+',): 'fwd', 
    ('-',): 'rev'
}[tuple(sorted(orientations))]

default_filename_func = lambda motif_id_slugname, orientations: f'{motif_id_slugname}_orientation_{orientations_to_filename_substr(orientations)}'

default_title_func = lambda motif_id, orientations: f'Plot of motif {motif_id} locations in ranked peaks, \n Orientation: '+'/'.join(orientations)

def plot_motif_instances_multiple(
    motif_distributions_by_motif, 
    sorted_peak_score_rank_df, 
    motif_id_slugname_df, 
    motif_ids = None, 
    figsize = (12.5, 10.0), 
    pointsize = 2.0, 
    alpha_factor = 4.0, 
    alpha_override = None,
    title_func = default_title_func,
    filename_func = default_filename_func, 
    plot_formats = ['svg', 'ps', 'png'],
    color_by_orientation = {'+': 'red', '-': 'blue'}, 
    progress_wrapper = tqdm, 
    plot_fwd = True, 
    plot_rev = True, 
    plot_separate = True,
    plot_tight = True,
    plot_dpi = 300, 
    n_jobs = 1, 
    close_fig = True
):
    if motif_ids == None:
        motif_ids = motif_id_slugname_df['motif_id']
    
    orientations_to_plot = []
    if plot_fwd: 
        orientations_to_plot.append('+')
    if plot_rev: 
        orientations_to_plot.append('-')
    
    
    motif_id_to_slugname = motif_id_slugname_df.set_index('motif_id')['slugname'].to_dict()
    
    orientations_to_plot = []
    if plot_fwd: 
        orientations_to_plot.append('+')
    if plot_rev: 
        orientations_to_plot.append('-')

    orientations_to_plot_sep = [orientations_to_plot]
    if plot_separate:
        orientations_to_plot_sep = (
            [orientations_to_plot] + 
            [[orientation] 
             for orientation 
             in orientations_to_plot])
    
    def wrap_plot(
        peak_and_motif_df_by_orientation, 
        rolling_mean_on_motif_position_by_orientation, 
        motif_rank_kde_by_orientation, 
        title, 
        filename, 
        orientations_to_plot 
    ):
        fig = plot_motif_instances_single(
            peak_and_motif_df_by_orientation = peak_and_motif_df_by_orientation, 
            rolling_mean_on_motif_position_by_orientation = rolling_mean_on_motif_position_by_orientation, 
            motif_rank_kde_by_orientation = motif_rank_kde_by_orientation, 
            sorted_peak_score_rank_df = sorted_peak_score_rank_df, 
            title = title, 
            filename = filename, 
            orientations_to_plot = orientations_to_plot, 
            color_by_orientation = color_by_orientation, 
            figsize = figsize, 
            pointsize = pointsize, 
            alpha_factor = alpha_factor, 
            alpha_override = alpha_override, 
            plot_formats = plot_formats,
            plot_tight = plot_tight,
            plot_dpi = plot_dpi, 
            close_fig = close_fig
        )
        return fig
    
    def get_wrap_plot_args(motif_id, orientations_to_plot):
        title = title_func(motif_id, orientations_to_plot)
        motif_id_slugname = motif_id_to_slugname[motif_id]
        filename = filename_func(motif_id_slugname, orientations_to_plot)
        (
            peak_and_motif_df_by_orientation, 
            rolling_mean_on_motif_position_by_orientation, 
            motif_rank_kde_by_orientation
        ) = motif_distributions_by_motif[motif_id]
        tup = (
            peak_and_motif_df_by_orientation, 
            rolling_mean_on_motif_position_by_orientation, 
            motif_rank_kde_by_orientation, 
            title, 
            filename, 
            orientations_to_plot
        )
        return tup
    
    wrap_plot_args_dict = {
        (motif_id, tuple(orientations_to_plot)): 
        get_wrap_plot_args(motif_id, orientations_to_plot) 
        for orientations_to_plot in orientations_to_plot_sep 
        for motif_id in motif_ids 
    }
    
    if n_jobs == 1:
        figs_by_motif_id_and_orientations_to_plot = {
            key: wrap_plot(*val) 
            for key, val 
            in progress_wrapper(wrap_plot_args_dict.items())}
    else:
        figs_by_motif_id_and_orientations_to_plot_tups = Parallel(
            n_jobs=n_jobs
        )(
            delayed(
                lambda tup: (tup[0], wrap_plot(*tup[1]))
            )((key, val)) 
            for key, val 
            in progress_wrapper(wrap_plot_args_dict.items())
        )
        figs_by_motif_id_and_orientations_to_plot = {
            tup[0]: tup[1] 
            for tup 
            in figs_by_motif_id_and_orientations_to_plot_tups
        }
        
    
    return figs_by_motif_id_and_orientations_to_plot

def plot_motif_instances_from_output_dir(
    output_dir, 
    motif_id_slugname_df = None, 
    n_top = 10, 
    motif_count_smooth_window = 3, 
    num_kde_points = 1000, 
    progress_wrapper = tqdm, 
    **kwargs
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
        motif_id_slugname_df = get_motif_id_slugname_df(lr_results_df).head(n_top)
    num_motif_ids = motif_id_slugname_df.shape[0]
    print(f'Plotting motif instances for {num_motif_ids} motifs')
    motif_ids = list(motif_id_slugname_df['motif_id'])
    scan_results_df = scan_results_df[scan_results_df['motif_id'].isin(motif_ids)]
    
    print('Formatting motif scan information')
    (
        peak_and_motif_df_by_motif_id, 
        sorted_peak_score_rank_df, 
        max_peak_length, 
        max_peak_score_rank
    ) = precompute_motif_dfs(
        lr_input_df, 
        scan_results_df, 
        motif_length_df, 
        peak_length_df, 
        progress_wrapper = progress_wrapper
    )
    
    print('Computing distributions of motifs')
    motif_distributions_by_motif = {
        motif_id: compute_motif_distributions(
            peak_and_motif_df, 
            max_peak_length, 
            max_peak_score_rank, 
            window = motif_count_smooth_window, 
            num_kde_points = num_kde_points)
        for motif_id, peak_and_motif_df 
        in progress_wrapper(peak_and_motif_df_by_motif_id.items())
    }
    
    print('Generating figures')
    figs_by_motif_id_and_orientations_to_plot_parallel = plot_motif_instances_multiple(
        motif_distributions_by_motif, 
        sorted_peak_score_rank_df, 
        motif_id_slugname_df, 
        filename_func = lambda motif_id_slugname, orientations: f'{output_dir}/'+default_filename_func(motif_id_slugname, orientations), 
        progress_wrapper = progress_wrapper, 
        **kwargs
    )
    return figs_by_motif_id_and_orientations_to_plot_parallel, motif_distributions_by_motif

if __name__ == '__main__':
    main()
    
