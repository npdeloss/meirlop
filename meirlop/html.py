from html import escape as escape_html
import base64
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

import io
import logomaker

import matplotlib.pyplot as plt

import matplotlib

def motif_matrix_to_df(motif_matrix, alphabet = 'ACGT'):
    return (
        pd.DataFrame(
            motif_matrix
        )
        .T
        .rename(
            columns = {
                k:v 
                for k,v 
                in enumerate(
                    list(
                        alphabet
                    )
                )
            }, 
            index = {
                i: 
                i+1 
                for i 
                in range(
                    motif_matrix.shape[1]
                )
            }
        )
    )

def plot_motif_matrix(motif_matrix, alphabet = 'ACGT', **kwargs):
    
    motif_logo = logomaker.Logo(
        logomaker.transform_matrix(
            motif_matrix_to_df(
                motif_matrix, 
                alphabet = alphabet
            ), 
            from_type = 'probability', 
            to_type = 'information'
        ), 
        **kwargs
    )

    # style using Logo methods
    motif_logo.style_spines(visible=False)
    motif_logo.style_spines(spines=['left', 'bottom'], visible=True)
    motif_logo.style_xticks(fmt='%d', anchor=0)

    # style using Axes methods
    motif_logo.ax.set_ylabel('bits', labelpad=-1)
    motif_logo.ax.xaxis.set_ticks_position('none')
    motif_logo.ax.xaxis.set_tick_params(pad=-1)
    motif_logo.ax.set_ylim(0.0, 2.0)
    
    return motif_logo

def get_motif_logo_svg(motif_logo, close_fig = True, **kwargs):
    matplotlib.use('svg')
    matplotlib.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['svg.fonttype'] = 'none'
    svg_data = io.StringIO()
    motif_logo.fig.savefig(svg_data, format = 'svg', **kwargs)
    if close_fig:
        plt.close(motif_logo.fig)
    return '\n'.join(svg_data.getvalue().split('\n')[3:])

def get_motif_logo_png(motif_logo, close_fig = True, **kwargs):
    matplotlib.use('agg')
    png_data = io.BytesIO()
    motif_logo.fig.savefig(png_data, format = 'png', **kwargs)
    if close_fig:
        plt.close(motif_logo.fig)
    png_data.seek(0)
    png_data_uri = (
        'data:image/png;base64,' + 
        (
            base64.b64encode(png_data.getvalue()).
            decode()
        )
    )
    png_img_str = f'<img src="{png_data_uri}">'
    return png_img_str

def get_svg_logo_for_motif_matrix(motif_matrix, savefig_kwargs = {'bbox_inches':'tight', 'transparent': True}, **kwargs):
    matplotlib.use('svg')
    matplotlib.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['svg.fonttype'] = 'none'
    return get_motif_logo_svg(plot_motif_matrix(motif_matrix, **kwargs), **savefig_kwargs)

def get_png_logo_for_motif_matrix(motif_matrix, savefig_kwargs = {'bbox_inches':'tight', 'transparent': True}, **kwargs):
    matplotlib.use('agg')
    return get_motif_logo_png(plot_motif_matrix(motif_matrix, **kwargs), **savefig_kwargs)

def get_html_for_lr_results_df(lr_results_df, 
                               motif_matrix_dict, 
                               name = '', 
                               n_jobs = 1, 
                               progress_wrapper = tqdm, 
                               cmdline = '', 
                               sortcol = 'coef', 
                               logo_format = 'svg'):
    df = (lr_results_df.copy()
          .sort_values(by = ['padj_sig', sortcol], ascending = False)
          .reset_index(drop = True))
    get_logo_for_motif_matrix = get_svg_logo_for_motif_matrix
    if logo_format is 'png':
        get_logo_for_motif_matrix = get_png_logo_for_motif_matrix
    get_motif_id_logo_tup = lambda motif_id: (
        motif_id, get_logo_for_motif_matrix(
            motif_matrix_dict[motif_id], 
            figsize = (
                0.3 * motif_matrix_dict[motif_id].shape[1], 
                1.5
            )
        )
    )
#     get_motif_id_logo_tup = lambda motif_id: (motif_id, get_html_logo_for_motif_matrix(motif_matrix_dict[motif_id]))
    html_logo_by_motif_id_tups = Parallel(n_jobs = n_jobs)(delayed(get_motif_id_logo_tup)(motif_id) 
                                                           for motif_id 
                                                           in progress_wrapper(list(motif_matrix_dict.keys())))
#     html_logo_by_motif_id_tups = [
#         get_motif_id_logo_tup(motif_id) 
#         for motif_id 
#         in progress_wrapper(list(motif_matrix_dict.keys()))]
    html_logo_by_motif_id = {motif_id: logo 
                             for motif_id, logo in html_logo_by_motif_id_tups}
    df['motif_logo'] = df['motif_id'].map(html_logo_by_motif_id)
    df = df[['motif_id',
             'motif_logo',
             'coef','abs_coef',
             'std_err',
             'ci_95_pct_lower','ci_95_pct_upper',
             'auc',
             'pval','padj','padj_sig', 'num_peaks', 'percent_peaks']]
    df_style = (df.style
                .bar(subset=['coef'], 
                     align='mid', 
                     color=['#e6bbad', '#add8e6'])
                .bar(subset=['padj_sig'], 
                     color = ['#ade6bb'], 
                     vmin = 0, 
                     vmax = 1)
                .bar(subset=['percent_peaks'], 
                     color = ['#e6d8ad'], 
                     vmin = 0.0, 
                     vmax = 100.0)
                .applymap(lambda x: (
                    'background-repeat: repeat-x; '
                    'background-size: 100% 50%; '
                    'background-position-y: 100%;'
                ), subset = [
                    'coef', 
                    'padj_sig', 
                    'percent_peaks'
                ]))
    old_width = pd.get_option('display.max_colwidth')
    pd.set_option('display.max_colwidth', -1)
    # df_html = df.to_html(escape = False)
    df_html = df_style.render(escape = False)
    pd.set_option('display.max_colwidth', old_width)
    cmdline_escaped = escape_html(cmdline)
    html = (f'''<!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>
    {name} Motif Enrichment Logistic Regression Results - MEIRLOP
    </title>
    <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.1/css/bootstrap.min.css"/>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.18/css/dataTables.bootstrap4.min.css"/>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/1.5.4/css/buttons.bootstrap4.min.css"/>
    
    <script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.1/js/bootstrap.min.js"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jszip/2.5.0/jszip.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.10.18/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.10.18/js/dataTables.bootstrap4.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/1.5.4/js/dataTables.buttons.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/1.5.4/js/buttons.bootstrap4.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/1.5.4/js/buttons.colVis.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/1.5.4/js/buttons.html5.min.js"></script>
    <script type="text/javascript">
    $(document).ready(function() {{
        $("table").addClass("table table-striped table-hover table-bordered table-sm table-responsive w-100 mw-100");
        $('table').DataTable({{
        'lengthMenu': [ [10, 25, 50, 100, -1], [10, 25, 50, 100, "All"] ], 
        'dom': ("<'row'<'col-sm-12 col-md-5'B><'col-sm-12 col-md-3'l><'col-sm-12 col-md-4'f>>" +
                "<'row'<'col-sm-12'tr>>" +
                "<'row'<'col-sm-12 col-md-5'i><'col-sm-12 col-md-7'p>>"), 
        'buttons': [ 'copy', 'excel', 'csv', 'colvis'], 
        'columnDefs': [{{'targets': [5, 6, 7, 8], 'visible': false}}]
        }});
    }} );
    </script>
    </head>
    <body>
    <div class="alert alert-info" role="alert">
        This table was generated with the following command line:
        <br/>
        <code>{cmdline_escaped}</code>
    </div>
    <div class="container-fluid">
    {df_html}
    </div>
    </body>
    </html> 
    ''')
    return html, html_logo_by_motif_id
