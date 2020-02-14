import weblogo as weblogolib
import base64
from html import escape as escape_html
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

def get_html_logo_for_motif_matrix(motif_matrix, image_size = 'medium', image_format = 'png'):
    pwm = motif_matrix.T
    data = weblogolib.LogoData.from_counts('ACGT', pwm)
    options = weblogolib.LogoOptions(fineprint=False,
                                     color_scheme=weblogolib.classic, 
                                     stack_width=weblogolib.std_sizes[image_size],
                                     logo_start=1, logo_end=pwm.shape[0])
    logo_format = weblogolib.LogoFormat(data, options)
    # print(image_size, image_format)
    if image_format is 'svg':
        # print('making svg logo')
        img_data_uri = ('data:image/svg+xml;base64,' 
                        + base64.b64encode(weblogolib
                                           .svg_formatter(data, logo_format))
                        .decode())
    else:
        # print('making png logo')
        img_data_uri = ('data:image/png;base64,' 
                        + base64.b64encode(weblogolib
                                           .png_formatter(data, logo_format))
                        .decode())
    
    return f'<img src="{img_data_uri}">'

def get_html_for_lr_results_df(lr_results_df, 
                               motif_matrix_dict, 
                               name = '', 
                               n_jobs = 1, 
                               progress_wrapper = tqdm, 
                               cmdline = '', 
                               sortcol = 'coef'):
    df = (lr_results_df.copy()
          .sort_values(by = ['padj_sig', sortcol], ascending = False)
          .reset_index(drop = True))
    get_motif_id_logo_tup = lambda motif_id: (motif_id, get_html_logo_for_motif_matrix(motif_matrix_dict[motif_id]))
    html_logo_by_motif_id_tups = Parallel(n_jobs = n_jobs)(delayed(get_motif_id_logo_tup)(motif_id) 
                                                           for motif_id 
                                                           in progress_wrapper(list(motif_matrix_dict.keys())))
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
                     vmax = 100.0))
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
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/responsive/2.2.2/css/responsive.bootstrap4.min.css"/>

    <script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.1/js/bootstrap.min.js"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jszip/2.5.0/jszip.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.10.18/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.10.18/js/dataTables.bootstrap4.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/1.5.4/js/dataTables.buttons.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/1.5.4/js/buttons.bootstrap4.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/1.5.4/js/buttons.colVis.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/1.5.4/js/buttons.html5.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/responsive/2.2.2/js/responsive.bootstrap4.min.js"></script>
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
    return html
