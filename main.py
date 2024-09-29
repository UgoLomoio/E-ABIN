"""
***************************************************************************************************************************************************************
*                                                                                                                                                             *
*   ADIN: A Python based GUI for anomaly detection and binary classification of gene expression microarray data using explainable machine learning and GAANs  *
*   Developed by Lomoio Ugo                                                                                                                                   *  
*   2024                                                                                                                                                      *
*                                                                                                                                                             *
***************************************************************************************************************************************************************
"""

import numpy as np
import pandas as pd 
import webbrowser
import dash
from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table import DataTable
import io 
import base64
#import multiprocessing

from sklearn.metrics import confusion_matrix
from torch.cuda import is_available
from torch.utils.data import dataloader
from adin import utils, ml, dl, preprocessing
import plotly.graph_objects as go
from dash import html, dcc
import torch
from collections import deque
import dash_cytoscape as cyto
import dash_cytoscape as cyto
from dash import Dash, html, Input, Output, State, callback, dcc
import json 
from torch_geometric.explain import Explainer, GNNExplainer
#from parallel_pandas import ParallelPandas

# Dynamically determine the number of CPU cores
#n_cpu = multiprocessing.cpu_count()
#initialize parallel-pandas
#ParallelPandas.initialize(n_cpu=n_cpu, split_factor=4, disable_pr_bar=True)

torch.cuda.empty_cache()


# Initialize the Dash app
html_plot_path = "http://127.0.0.1:8050/"
app = dash.Dash(__name__, suppress_callback_exceptions=False, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Anomaly Detection in Individualized Networks"

# Global variables to hold the data
log_messages = deque(maxlen=1)  
gene_data = None
expr = None 
targets = None
patients = None
fig_pca = None
fig_tsne = None
df_result = None
data = None
gene_filename = None
limited_expr = None  
saved_figures = {}
gene_fileg = None 
models = {"Temp": None}
method_g = None
fig_box = None
fig_roc_test = None
fig_roc = None
X_train = None
X_test = None
y_train = None
y_test = None 
mydataloader = None 
progress = 0 
genes = None 
nodes = []
edges = []
explainer = None 
node_mapping = None 
map_edgeattr = None
map_final = {0: "Normal", 1: "Anomalous"}


analysis_options = {
    'ML': ['KNN', 'LR', 'DT', 'SVM', 'RF'], #['LDA', 'NB', 'KNN', 'LR', 'DT', 'SVM', 'RF']
    'GAAN': ['GAAN']
}

reset_button = dbc.Button(
    "Reset", 
    id="reset-button", 
    color="danger",  # This sets the button color to red
    style={
        'position': 'absolute', 
        'top': '1%',  
        'right': '50%',  
        'width': '8%', 
        'height': '5%', 
        'textAlign': 'center'
    },
    n_clicks=0
)

navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Upload", href="/upload", id='upload-nav', n_clicks=0)),
            dbc.NavItem(dbc.NavLink("Embeddings", href="/embeddings", id='embeddings-nav', n_clicks=0)),
            dbc.NavItem(dbc.NavLink("Analysis", href="/analysis", id='analysis-nav', n_clicks=0)),
            dbc.NavItem(dbc.NavLink("Explainability", href="/explain", id='explain-nav', n_clicks=0))
        ],
        brand="ADIN",
        brand_href="/",
        color="primary",
        dark=True,
        style={
                    'width': '100%',
                    'height': '62px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'left': '-5px',
                    'top': '-5px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '0px'
        },       
)


# Define the custom modebar button configuration
config = {
    'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'figure',
    'height': 800,
    'width': 800,
    'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
    },
    'displayModeBar': True,
    'displaylogo': False  # Optionally remove Plotly logo from modebar
}



footer = html.Footer([
            
            html.Div([
                #html.P("Anonymized Footer for peer-review"), 

                html.Div([
                    html.Img(src='/assets/unicz.png', style = {'right': '0%', 'width': '10%', 'height': '10%'}),
                ], style = {
                    'position': 'absolute',
                    'background-color': '#EBEBEB',
                    'display': 'inline-block'}
                ),

                html.Div([
                    html.Img(src='/assets/unical.png', style = {'left': '0%', 'width': '10%', 'height': '10%'}),  
                    ], style = {
                        'position': 'absolute',
                        'background-color': '#EBEBEB',
                        'display': 'inline-block'
                    }
                ),
                html.Div([
                    html.P(["Pietro Hiram Guzzi", html.Sup(1), ", Lomoio Ugo", html.Sup("1, 2"), ", Tommaso Mazza", html.Sup(3), " and Pierangelo Veltri", html.Sup(4), html.Br(), "(1) Magna Graecia University of Catanzaro, (2) Relatech SpA", html.Br(), "(3) IRCCS Casa Sollievo della Sofferenza, (4) University of Calabria Rende", html.Br()]),
                    ], style= {
                        'position': 'absolute',
                        'right': '10%',
                        'left': '10%',
                        'width': '80%',
                        'height': '10%',
                        'textAlign': 'center',
                        'background-color': '#EBEBEB',
                        'display': 'inline-block'
                    }
                ),
            ], style = {'width': '100%'})
        ])


download_button_style = {   'position': 'absolute', 'bottom': '6%', 'left': '47%', 'width': '10%',
                            'background-color': '#34B212', 'height': '8%', 'text-align': 'center', 'display':'none'}

upload_layout = html.Div([
    
    html.Div([    
        html.H4("Upload Gene Expression File", style={'position': 'absolute', 'left': '1%'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            multiple=False,
            style={
                'width': '70%', 'height': '10%', 'lineHeight': '200%',
                'borderWidth': '1px', 'left': '21%', 'right': '5%',
                'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '1%'
            },
        ),
        # Add the radio button here
        html.Div([
            html.Div("Is a preprocessed gene expression file", style={'display': 'inline-block', 'padding-left': '10px', 'padding-right': '10px'}),
            dcc.RadioItems(
                id='preprocessed-option',
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ],
                value='no',
                style={'display': 'inline-block'}
            )
        ], style={'margin': '10px 0'}),
    ]),
    
    html.Div([
        dcc.Interval(id='interval-component', interval=500, n_intervals=0),
        dcc.Interval(id='autorefresh-component', interval=1000, n_intervals=0), 
        dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
        html.P(id="log-display", style={'width': '20%', "whiteSpace":"pre-wrap", "padding-top":"1.4%", 'textAlign': 'center', 'background-color': '#FFFFFF', "overflow":"auto"}),
        dbc.Progress(id="progress", animated=True, striped=True, style={'position': 'absolute', 'width': '70%', 'left': '22%', 'right': '5%', 'top': '25%'}),
        html.Div(id="output-data-upload", style={'position': 'absolute', 'width': '100%', 'top': '30%'}),  # This is where the uploaded data will be shown
            
        html.Iframe(src=app.get_asset_url('summary.html'), style={"height": "500px", "overflow": "hidden"}),
   
        dbc.Button("Upload Files", id="upload-button", style={
                'position': 'absolute', 'top': '10%', 'right': '0%', 'width': '8%',
                'background-color': '#34B212', 'height': '6%', 'text-align': 'center'
        }, n_clicks=0),
        dbc.Button("Download Preprocessed File", id="download-button", style=download_button_style, n_clicks=0),
        dcc.Download(id="download-csv"),
    ]),
    #footer
])


modal = dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Graph Viewer")),
                        dbc.ModalBody(dcc.Graph(id="modal-figure")),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-modal", n_clicks=0)
                        ),
                    ],
                    id="modal",
                    is_open=False,
                    fullscreen=True
                )

embedding_layout = html.Div([
    
    html.Div([
        dbc.Button("Compute Embedding", id="embeddings-button", href='/embeddings', style={
                'position': 'absolute', 'top': '10%', 'right': '45%', 'width': '10%',
                'background-color': '#34B212', 'height': '5%', 'text-align': 'center'
        }, n_clicks=0),
        dcc.Loading(
            id="loading-embedding-output",
            children=[
                html.Div(id="output-data-embedding"),  # This is where the embedding data will be shown
            ],
            fullscreen=False,  # Set to True if you want the loading spinner to cover the entire screen
            overlay_style={"visibility":"visible"},
            type="circle"
        )
    ]),
    #footer
])

analysis_layout = html.Div([
    html.Div([
        html.H2("Select the Analysis Method:", id='text-analysis', style={'position': 'absolute', 'top': '8%', 'left': '0%', 'width': '100%', 'textAlign': 'center'}),
        dcc.Dropdown(
            id='analysis-method',
            options=[
                {'label': 'ML Binary Classification', 'value': 'ML'},
                {'label': 'GAAN Anomaly Detection', 'value': 'GAAN'}
            ],
            value='ML', 
            style={'position': 'absolute', 'top': '13%', 'left': '16%', 'width': '70%', 'textAlign': 'center'}
        ),
        dbc.Button("Analyze", id='analyze-button', style={
            'position': 'absolute', 'top': '13%', 'right': '3%', 'width': '8%', 
            'height': '4%', 'textAlign': 'center', 'background-color': '#34B212'
        }, n_clicks=0),
        dcc.Loading(
            id="loading-analysis-output",
            children=[
               html.Div(id='analysis-output', style={'position': 'absolute', 'top': '33%', 'left': '0%', 'width': '100%'})
            ],
            fullscreen=False,  # Set to True if you want the loading spinner to cover the entire screen
            overlay_style={"visibility":"visible"},
            type="circle"
        )
    ]),
    #footer
])

explainability_layout = html.Div([
        
        html.Div([
            
            html.H2("Select Model for Explainability:", style={'position': 'absolute', 'top': '8%', 'left': '0%', 'width': '100%', 'textAlign': 'center'}),
            
            dcc.Dropdown(
                id='model-dropdown',
                style={'position': 'absolute', 'top': '18%', 'left': '15%', 'width': '70%', 'textAlign': 'center'}
            ),
            
            dbc.Button("Explain", id='explain-button', style={
                'position': 'absolute', 'top': '13%', 'right': '3%', 'width': '8%', 
                'height': '4%', 'textAlign': 'center', 'background-color': '#34B212'
            }, n_clicks=0),
            
            dcc.Dropdown(
                id='analysis-type',
                options=[
                    {'label': 'ML Binary Classification', 'value': 'ML'},
                    {'label': 'GAAN Anomaly Detection', 'value': 'GAAN'}
                ],
                value='ML', 
                style={'position': 'absolute', 'top': '13%', 'left': '15%', 'width': '70%', 'textAlign': 'center'}
            )
        ]), 

        dcc.Loading(
            id="loading-explain-output",
            children=[
                    html.Div(id='explain-output', style={'position': 'absolute', 'top': '24%', 'left': '0%', 'width': '100%'}),
            ],
            fullscreen=False,  # Set to True if you want the loading spinner to cover the entire scree
            overlay_style={"visibility":"visible"},
            type="circle"
        )
        #footer
])


homepage_layout = html.Div([
    navbar,
    reset_button,
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content'),#, children = footer),  # This is where page content will be dynamically inserted
    modal
])


# Main app layout
app.layout = homepage_layout

def add_log_message(message):
    global log_messages
    log_messages.append(message)


def upload_preprocessed_file(gene_file):
    
    global gene_data
    global expr 
    global targets 
    global limited_expr
    global progress
    global patients 

    progress = 0
    if gene_file is not None:
        
        print("Uploading Already Preprocessed Gene Expression File")
        add_log_message("Uploading Gene File")
        content_type, content_string = gene_file.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Assume that the user uploaded a CSV file
            gene_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        except Exception as e:
            return html.Div([
                html.P('There was an error processing this gene file.',
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center'})
            ])
        
        progress = 20

        print("Reading gene expression file")
        add_log_message("Reading gene expression file")
        try:
            expr = gene_data.loc[:, gene_data.columns != 'Target']
            patients = list(gene_data.index)
            targets = gene_data.loc[:, gene_data.columns == 'Target']
        except Exception as e:
            return html.Div([
                html.P("'Target' column for patient binary diagnosis not found.",
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center'})
            ])
        
        progress = 40
        targets_uq = np.unique(targets['Target'].values)
        if len(targets_uq) > 2:
            raise Exception("Target column must have only 2 unique values to perform Anomaly Detection and Binary Classification. Found {} unique targets.".format(len(targets_uq)))

        df = expr.join(targets)

        progress = 90
        if "Target" in expr.columns:
            expr = expr.drop("Target", axis=1)
        
        limited_expr = expr.iloc[:, :50]
        limited_expr = limited_expr.join(targets)
        limited_expr = limited_expr.T
        limited_expr = limited_expr.reset_index()
        
        columns = [{'name': col, 'id': col} for col in limited_expr.columns]
        data = limited_expr.to_dict(orient='records')
            
        progress = 100 

        return html.Div([
                html.Br(),
                html.Br(),
                html.Br(),
                html.H5("Gene Data", style = {'textAlign': 'center'}),
                html.Hr(),
                DataTable(
                    data=data,
                    columns=columns,
                    page_size=10, 
                    style_table={'overflowY': 'auto', 'overflowX': 'scroll', 'height': '20%'},  # Scrollable table height
                )
        ])
    
def upload_file(gene_file):
    
    global gene_data
    global expr 
    global targets 
    global limited_expr
    global progress 
    global patients 

    progress = 0
    if gene_file is not None:
        
        print("Uploading Gene File")
        add_log_message("Uploading Gene File")
        content_type, content_string = gene_file.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Assume that the user uploaded a CSV file
            gene_data = utils.read_gene_expression(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            return html.Div([
                html.P('There was an error processing this gene file.',
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center'})
            ])
        progress = 10

        print("Detecting Geo accession code")
        add_log_message("Detecting Geo accession code")
        geo_code = utils.find_geoaccession_code(io.StringIO(decoded.decode('utf-8')))
        if geo_code is None:
            raise Exception ("Cannot detect GEO accession code from file. File must be corrupted.")
        print("Detecting platforms")
        add_log_message("Detecting platforms")
        gse, platforms = utils.get_platforms(geo_code)       
        progress = 30
 
        print("Reading gene expression file")
        add_log_message("Reading gene expression file")
        expr = utils.get_expressions(gene_data)
        patients = list(expr.index)
        #print("Patients:", patients)
        progress = 35

        print("Downloading annotation file")  
        
        found_platform, annotation_df = utils.get_annotation_df(gse, expr, platforms)
        if found_platform is None:
            raise Exception ("Cannot detect any valid platforms from file. File must be corrupted.")
        targets = utils.get_targets(gene_data)
        progress = 50 
        
        #print("Expr:", expr)
        print("Targets:", targets)
        df = expr.join(targets)
    
        print("Preprocessing")
        add_log_message("Renaming columns & Replacing none values.")
        expr = preprocessing.preprocess(df, annotation_df, need_rename=True) 
        progress = 90
        #print("Preprocessed:", expr)
        

        if "Target" in expr.columns:
            expr = expr.drop("Target", axis=1)
        
        targets_uq = np.unique(targets['Target'].values)
        if len(targets_uq) > 2:
            raise Exception("Target column must have only 2 unique values to perform Anomaly Detection and Binary Classification. Found {} unique targets.".format(len(targets_uq)))
       
        
        limited_expr = expr.iloc[:10, :50]
        limited_expr = limited_expr.join(targets)
        limited_expr = limited_expr.T
        limited_expr = limited_expr.reset_index()
        
        columns = [{'name': col, 'id': col} for col in limited_expr.columns]
        data = limited_expr.to_dict(orient='records')
        progress = 100

        return html.Div([
                html.Br(),
                html.Br(),
                html.Br(),
                html.H5("Gene Data", style = {'textAlign': 'center'}),
                html.Hr(),
                DataTable(
                    data=data,
                    columns=columns,
                    page_size=10, 
                    style_table={'overflowY': 'auto', 'overflowX': 'scroll', 'height': '20%'},  # Scrollable table height
                ),
                #footer
        ])
        #FIX PREPROCESSING
    
def update_explainability_ml(model_name):
    
    global expr 
    global models 
    global X_train
    global X_test 
    global config 
    global patients
    global targets 
    global genes 
    global saved_figures 
    global y_train, y_test 

    if expr is not None and model_name != "Temp":
        if model_name is not None and model_name in list(models.keys()):
            model = models[model_name]
            genes = expr.columns 
            X = np.concatenate((X_train, X_test), axis = 0)
            
            if model_name in ["KNN", "SVM"]:
                return html.Div([
                            html.P(
                                "SHAP explanations are not available for SVM and KNN models due to high memory demands.",
                                style={'color': 'red', 'fontWeight': 'bold'}
                            )
                ], style = {"position": "fixed", "top": "40%", 'width': "100%", 'textAlign': 'center'})
            elif model_name == "LR":
                fig = ml.explain_model(model, model_name, X, genes, X_train=X_train) 
                fig_sum = ml.shap_summary_plotly(model, model_name, X, targets, genes, patients, X_train = X_train)
            else: 
                fig = ml.explain_model(model, model_name, X, genes)
                fig_sum = ml.shap_summary_plotly(model, model_name, X, targets, genes, patients)
            
            saved_figures["shap-exp"] = fig
            saved_figures["shap-sum"] = fig_sum

            style_force = {}

           
            #fig_sum_mat = ml.shap_summary(model, model_name, X, genes, X_train = X_train) #test with shap library results
            #fig_force_mat = ml.shap_force(model, model_name, X_test, y_test, genes, index = 0, X_train = X_train)
            patient = patients[0]
            ys = targets.values
            y = ys[0].item()
            plot_title = "Shap Force plot for model {} and patient {} with class {}".format(model_name, patient, map_final[y])
            
            return html.Div([
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                
                html.H4(
                    "Model {} SHAP Explanations".format(model_name),
                    style={
                        'top': '23%',  
                        'width': '100%',
                        'left': '0%',
                        'textAlign': 'center'
                    }
                ),

                html.Div([

                    html.Div([
                        dcc.Graph(
                            id = "shap-exp",
                            figure=fig,
                            config = config,
                            style = {'display': 'inline-block'}
                        ),
                        html.Br(),
                        dbc.Button("Popout Figure", id = "modal-shapexp", n_clicks=0),
                    ], style = {'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(
                            id = "shap-sum",
                            figure=fig_sum,
                            config = config,
                            style = {'display': 'inline-block'}
                        ),
                        html.Br(),
                        dbc.Button("Popout Figure", id = "modal-shapsum",  n_clicks=0)
                
                    ], style = {'display': 'inline-block'})

                ], style = {'width': '100%'}),
                
                html.Br(),

                html.Div([
                    dcc.Dropdown(
                        id='dropdown',
                        clearable=False,
                        options=[
                            {'label': name, 'value': i}
                            for i, name in enumerate(patients)
                        ]
                    ),
                    html.Div([
                         ml.get_plot("force-plot", {}, model, model_name, X, ys, genes, index = 0, top_n = 10, X_train = X_train, class_id = y, title=plot_title)
                    ], id = 'force-plot-div', style = {"width": "100%"})
                ])
            ])
    
    return html.Div()

def update_explainability_dl(model_name):

    global default_stylesheet
    global nodes 
    global edges 
    global explainer 
    
    global patients 
    global edge_list 
    global edge_weights 
    global models 
    global mydataloader 
    
    if model_name is not None:
        if model_name in list(models.keys()):
            
            model = models[model_name]
            y = mydataloader.y
            edge_weights = mydataloader.edge_attr
            preds = model.predict(mydataloader)
            nodes, edges = dl.compute_elements_cyto(patients, edge_list, edge_weights.cpu(), y.cpu(), preds.cpu())
            div = html.Div([
                
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),

                    dcc.Dropdown(
                        id='dropdown-update-layout',
                        value='grid',
                        clearable=False,
                        options=[
                            {'label': name.capitalize(), 'value': name}
                            for name in ['grid', 'random', 'circle', 'cose', 'concentric', 'breadthfirst']
                        ]
                    ),

                    html.Div([
                        html.Button('Add Node', id='btn-add-node', n_clicks_timestamp=0),
                        html.Button('Remove Node', id='btn-remove-node', n_clicks_timestamp=0)
                    ]),

                    html.Div([
                        html.P("Network:"),
                        cyto.Cytoscape(
                            id='network',
                            elements=edges+nodes,
                            layout={'name': 'breadthfirst'},
                            style={'width': '100%', 'height': '400px'},
                            stylesheet=default_stylesheet,
                            responsive=True
                        ),
                        html.Div([
                            html.P("Clicked Node/Edge informations: "),
                            html.Pre(id='cytoscape-tapNodeData-json', children = "Click one node get more information", style=styles['pre']),
                            html.Pre(id='cytoscape-tapEdgeData-json', children = "Click one edge to visualize its information", style=styles['pre']),
                        ]),
                    ]),
                    
                    dcc.Graph(
                            id='exp-node', 
                            style = {'width': '50%', 'height': '400px', 'display':'inline-block'}
                    ),
                    html.Br(),
                    dbc.Button("Popout Figure", id = "modal-expnode", n_clicks=0),

                    html.P("Subgraph:"),
                    cyto.Cytoscape(
                        id='subnetwork',
                        elements=edges+nodes,
                        layout={'name': 'breadthfirst'},
                        style={'width': '100%', 'height': '400px'},
                        stylesheet=default_stylesheet,
                        responsive=True
                    ),
                    #footer
            ], style = {'position': "absolute", "width": "100%"})
            return div
        
        else:
            return None
    
    else:
        return None

def update_embeddings(update=False):
    
    global expr, targets, fig_pca, fig_tsne, limited_expr, config
    global saved_figures

    if expr is not None:

        if update:
            
            df = expr.join(targets)
            fig_pca, fig_tsne = utils.compress_expr(expr.values, targets.values)
            if limited_expr is None:
                limited_expr = expr.iloc[:, :50]
                limited_expr = limited_expr.join(targets)
                limited_expr = limited_expr.T
                limited_expr = limited_expr.reset_index()

            columns = [{'name': col, 'id': col} for col in limited_expr.columns]
            data = limited_expr.to_dict(orient='records')
            div = html.Div([    
                        html.Div([
                            html.Div([
                                dcc.Graph(id='pca-plot', figure = fig_pca, config = config, style={'top': '8%', 'left': '1%', 'height': '40%', 'width': '40%', 'display': 'inline-block'}),
                                html.Br(),
                                dbc.Button("Popout Figure", id = "modal-pca", n_clicks=0),
                            ], style = {'display': 'inline-block'}),
                                
                            html.Div([
                                dcc.Graph(id='tsne-plot', figure = fig_tsne, config = config, style={'top': '8%', 'right': '1%', 'height': '40%', 'width': '40%', 'display': 'inline-block'}),
                                html.Br(),
                                dbc.Button("Popout Figure", id = "modal-tsne", n_clicks=0)
                            ], style = {'display': 'inline-block'})
                        ], style = {'width': '100%'}),
                        
                        html.Br(),
                        html.Br(),
                        html.Div([
                                html.H5("Gene Data", style={'left': '50%', 'textAlign': 'center'}),
                                DataTable(
                                    id = 'data-table',
                                    data = data,
                                    columns = columns,
                                    page_size=5, 
                                    style_table={'overflowY': 'auto', 'overflowX': 'scroll', 'height': '20%'},  # Scrollable table height
                                )],       
                        style={'width': '100%', 'height': '20%'}
                        ),
                        #footer
            ])
            saved_figures["pca-plot"] = fig_pca
            saved_figures["tsne-plot"] = fig_tsne
            return div 

        
        else:
            
            if limited_expr is not None and fig_pca is not None and fig_tsne is not None:
                
                columns = [{'name': col, 'id': col} for col in limited_expr.columns]
                data = limited_expr.to_dict(orient='records')
            
                div = html.Div([    
                        html.Div([
                            html.Div([
                                dcc.Graph(id='pca-plot', figure = fig_pca, config = config, style={'top': '8%', 'left': '1%', 'height': '40%', 'width': '40%', 'display': 'inline-block'}),
                                html.Br(),
                                dbc.Button("Popout Figure", id = "modal-pca", n_clicks=0),
                            ], style = {'display': 'inline-block'}),
                                
                            html.Div([
                                dcc.Graph(id='tsne-plot', figure = fig_tsne, config = config, style={'top': '8%', 'right': '1%', 'height': '40%', 'width': '40%', 'display': 'inline-block'}),
                                html.Br(),
                                dbc.Button("Popout Figure", id = "modal-tsne",  n_clicks=0)
                            ], style = {'display': 'inline-block'})
                        ], style = {'width': '100%'}),
                        html.Br(),
                        html.Br(),
                        html.Div([
                                html.H5("Gene Data", style={'position': 'absolute', 'top': '45%', 'left': '50%', 'textAlign': 'center'}),
                                DataTable(
                                    id = 'data-table',
                                    data = data,
                                    columns = columns,
                                    page_size=5, 
                                    style_table = {'overflowY': 'auto', 'overflowX': 'scroll', 'height': '20%'},
                                )],       
                        style={'top': '55%', 'width': '100%', 'height': '20%'}
                        ),
                        #footer
                    ])
                
                return div 
            
            else:
                return embedding_layout
            
    else:
        return embedding_layout

def update_analysis_output():
    
    global df_result
    global expr 
    global targets 
    global edge_index
    global models 
    global method_g 
    global fig_roc
    global fig_roc_test
    global fig_box 
    global X_train
    global X_test 
    global y_train 
    global y_test 
    global mydataloader
    global config 
    global edge_list 
    global node_mapping
    global map_edgeattr
    global explainer 
    global saved_figures 

    if method_g == 'GAAN':
        toprint = "Anomaly Detection"
    else:
        toprint = "Binary Classification"
    
        
    if method_g == 'ML':
            
        if fig_box is None or fig_roc is None or fig_roc_test is None:
               
            df = pd.concat([expr, targets], axis=1)
            X_train, X_test, y_train, y_test = ml.train_test_split(df)
        
            models_tuple, fig_roc, fig_box = ml.baselineComparison(X_train, y_train)
            temp = {}
            for model_name, model in models_tuple:
                temp[model_name] = model
            models = temp 
        
            df_result = ml.create_results_df(models, X_test, y_test)
            columns = [{'name': col, 'id': col} for col in df_result.columns]
            data = df_result.to_dict(orient='records')
            fig_roc_test, _, _ = ml.models_roc_curves(models_tuple, X_test, y_test)
            
            best_modelname = df_result["Model Name"][0]
            best_model = models[best_modelname]
            preds = best_model.predict(X_test)
            classes = {0: "Normal", 1: "Anomalous"}
            cm = confusion_matrix(y_test, preds)
            fig_confm = dl.plot_cm(classes, cm)

            saved_figures["roc-test"] = fig_roc_test
            saved_figures["boxplot"] = fig_box
            saved_figures["roc"] = fig_roc
            saved_figures["confm"] = fig_confm

            return html.Div([
                            
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),

                            # DataTable for displaying the results
                            html.Div([
                                    DataTable(
                                        data=data,
                                        columns=columns,
                                        page_size=5,  # Reduce rows displayed per page
                                        style_table= {'overflowY': 'auto', 'overflowX': 'auto', 'height': '20%'},  # Scrollable table height
                                    ),
                            ]), # Adjust width and margin for alignment
                            
                            html.Div([
                                html.Div([
                                    dcc.Graph(
                                        id='box-plot',
                                        figure=fig_box,  # Box plot figure
                                        config = config,
                                    ),
                                    html.Br(),
                                    dbc.Button("Popout Figure", id = "modal-boxplot", n_clicks=0)
                                ], style = {'display': 'inline-block'}),
                                html.Div([
                                    # First ROC Curve
                                    dcc.Graph(
                                        id='roc-curve',
                                        figure=fig_roc,
                                        config = config,
                                          # Adjust height as needed
                                    ),
                                    html.Br(),
                                    dbc.Button("Popout Figure", id = "modal-roc", n_clicks=0)
                                ], style={'display': 'inline-block'})
                            ]),

                            html.Div([
                                html.Div([
                                    # Second ROC Curve
                                    dcc.Graph(
                                        id='roc-curve-test',
                                        figure=fig_roc_test,
                                        config = config,
                                    ),
                                    html.Br(),
                                    dbc.Button("Popout Figure", id = "modal-roctest", n_clicks=0),
                                ], style={'display': 'inline-block'}),
                                html.Div([
                                    dcc.Graph(
                                        id='conf-matrix',
                                        figure=fig_confm,
                                        config = config,
                                    ),
                                    html.Br(),
                                    dbc.Button("Popout Figure", id = "modal-confm",  n_clicks=0)
                                ], style = {'display': 'inline-block'}),
                            ])
                            #footer
                ], style = {'width': '100%'})
        
    elif method_g == 'GAAN':
        
        edges_i, edge_list, edge_weights, map_edgeattr = utils.get_edges_by_sim(expr)

        # Parse the edges and create node mapping
        source_nodes, target_nodes, node_mapping = utils.parse_edges(edges_i)
  
        # Create edge_index tensor
        edge_index = utils.create_edge_index(source_nodes, target_nodes)
        x = expr.values
        y = targets.values
        mydataloader = dl.create_torch_geo_data(x, y, edge_index, edge_weights)
        node_mapping_rev = {value: key for key, value in node_mapping.items()}
        dataloader_train, dataloader_test = dl.train_test_split_and_mask(mydataloader, map_edgeattr, node_mapping_rev)
        model = dl.train_gaan(dataloader_train)
        models["GAAN"] = model
        if "Temp" in models.keys():
            del models["Temp"]
        df_result = dl.create_results_df(model, dataloader_test)
        
        preds = model.predict(dataloader_test).cpu()
        
        y_test = dataloader_test.y.cpu()
        classes = {0: "Normal", 1: "Anomalous"}
        cm = confusion_matrix(y_test, preds)
        fig_cm = dl.plot_cm(classes, cm)
        fig_roc_test = ml.plot_roc_curve_(y_test, preds)
        saved_figures["confm"] = fig_cm 
        saved_figures["roc-test"] = fig_roc_test

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=0),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        
        columns = [{'name': col, 'id': col} for col in df_result.columns]
        data = df_result.to_dict(orient='records')
        return  html.Div([
                              html.Br(),
                              html.Br(),
                              html.Br(),
                              html.Br(),
                              html.Br(),
                              html.Br(),

                              DataTable(
                                    data=data,
                                    columns=columns,
                                    page_size=5, 
                                    style_table={'overflowY': 'auto', 'overflowX': 'auto'},  # Scrollable table height
                              ),
                              html.Div([
                                  html.Div([
                                      dcc.Graph(
                                            id='roc-curve-test',
                                            figure=fig_roc_test,
                                            config = config,
                                            style={'width': '40%', 'height': '20%', 'right': '5%'}  # Adjust height as needed
                                      ),
                                      html.Br(),
                                      dbc.Button("Popout Figure", id = "modal-roctest", n_clicks=0),
                                  ], style = {'display': 'inline-block'}),
                                  html.Div([
                                      # First ROC Curve
                                      dcc.Graph(
                                        id='conf-matrix',
                                        figure=fig_cm,
                                        config = config,
                                        style={'height': '20%', 'width': '40%', 'left': '5%'}  # Adjust height as needed
                                      ),
                                      html.Br(),
                                      dbc.Button("Popout Figure", id = "modal-confm", n_clicks=0)
                                  ], style = {'display': 'inline-block'})

                              ]),
   
                              #footer
            ], style={'top': '18%', 'width': '100%'}
        ) 

# Callback to update page content based on URL
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"), 
    Input("reset-button", "n_clicks")
)
def update_layout(pathname, n_clicks):
    
    global gene_filename
    global limited_expr
    global df_result
    global fig_pca
    global model_name 
    global method_g 
    global gene_data 
    global expr 
    global targets 
    global fig_tsne 
    global data 
    global saved_figures 
    global gene_fileg 
    global models   
    global method_g
    global analysis_options 
    global log_messages
    global X_train
    global X_test
    global y_train
    global y_test 
    global mydataloader
    global progress 
    global download_button_style
    global nodes 
    global edges 
    global explainer 
    global map_edgeattr
    global node_mapping
    global genes 

    # Debugging print statements
    
    print(f"Callback triggered with pathname: {pathname}")
    ctx = callback_context

    # Determine which input triggered the callback
    if not ctx.triggered:
        return "Waiting for input..."
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
    if triggered_id == 'url':
        if pathname == "/embeddings":
            if limited_expr is not None:
                return update_embeddings(False)
            else:
                return embedding_layout
    
        elif pathname == '/analysis':
            return analysis_layout
        
        elif pathname == '/explain':
            return explainability_layout
    
        elif pathname == '/upload':
            if gene_filename is not None:
            
                if limited_expr is not None:
                    columns = [{'name': col, 'id': col} for col in limited_expr.columns]
                    data = limited_expr.to_dict(orient='records')
                    return html.Div([
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.H5("Gene Data", style = {'textAlign': 'center'}),
                            html.Hr(),
                            DataTable(
                                data=data,
                                columns=columns,
                                page_size=10,
                                style_table={'overflowY': 'auto', 'overflowX': 'scroll', 'height': '20%'},  # Scrollable table height
                            ), 
                            #footer
                    ])
            
                else: 
                    return upload_layout
            else:
                return upload_layout
            
    elif triggered_id == 'reset-button':
        # Handle reset button logic
        if n_clicks > 0:
            #reset all global parameters
            # Global variables to hold the data
            gene_data = None
            expr = None 
            targets = None
            fig_pca = None
            fig_tsne = None
            df_result = None
            data = None
            gene_filename = None
            limited_expr = None 
            saved_figures = {}
            gene_fileg = None 
            models = {"Temp": None}
            method_g = None
            log_messages = deque(maxlen=1)  
            fig_box = None
            fig_roc_test = None
            genes = None
            fig_roc = None
            fig_box = None
            X_train = None
            X_test = None
            y_train = None
            y_test = None 
            mydataloader = None
            progress = 0
            nodes = []
            edges = []
            explainer = None 
            node_mapping = None 
            map_edgeattr = None
            analysis_options = {
                'ML': ['KNN', 'LR', 'DT', 'SVM', 'RF'], #['LDA', 'NB', 'KNN', 'LR', 'DT', 'SVM', 'RF']
                'GAAN': ['GAAN']
            }
            
            download_button_style = {   'position': 'absolute', 'bottom': '6%', 'left': '47%', 'width': '10%',
                            'background-color': '#34B212', 'height': '8%', 'text-align': 'center', 'display':'none'}
            if pathname == "/upload":
                return upload_layout
            elif pathname == "/embeddings":
                return embedding_layout
            elif pathname == '/analysis':
                return analysis_layout
            elif pathname == '/explain':
                return explainability_layout
    
    return dash.no_update


@app.callback(
    [Output("progress", "value"), Output("progress", "label"), Output("progress-interval", "disabled")],
    [Input("progress-interval", "n_intervals")],
    prevent_initial_call = True
)
def update_progress(n):
    
    global progress 
    
    print(progress)
    if progress < 100:
        return progress, f"{progress} %", False
    else:
        return progress, f"{progress} %", True
    
    
# Callback to update page content based on upload file button clicks
@app.callback(
    Output("output-data-upload", "children"),
    Output("log-display", "style"),
    Output("autorefresh-component", "disabled"),  # Enable or disable auto-refresh
    Output("download-button", "style"),
    Input("upload-button", "n_clicks"),
    Input("upload-data", "contents"),
    Input("preprocessed-option", "value"),
    prevent_initial_call=True  # Ensure callback doesn't run on load
)
def update_upload_onclick(upload_nclicks, gene_file, preprocessed):
    
    global limited_expr 
    global gene_fileg 
    global download_button_style

    # Debugging print statements
    print(f"Button clicks - Upload: {upload_nclicks}")

    # Prioritize button clicks over pathname
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
        download_button_style['display'] = 'none'
        return None, {'display': 'block'}, dash.no_update, download_button_style
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'upload-button':
        
        download_button_style['display'] = 'block'
        if gene_fileg is not None:
            if preprocessed == 'yes':
                return upload_preprocessed_file(gene_fileg), {'display': 'none'}, True, download_button_style
            else:    
                return upload_file(gene_fileg), {'display': 'none'}, True, download_button_style
            
        if gene_file is not None: 
            gene_fileg = gene_file 
            if preprocessed == 'yes':
                return upload_preprocessed_file(gene_file), {'display': 'none'}, True, download_button_style
            else:
                return upload_file(gene_file), {'display': 'none'}, True, download_button_style
            
        
    if limited_expr is not None:
        download_button_style['display'] = 'block'
        columns = [{'name': col, 'id': col} for col in limited_expr.columns]
        data = limited_expr.to_dict(orient='records')
        div = html.Div([
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.H5("Gene Data", style = {'textAlign': 'center'}),
                    html.Hr(),
                    DataTable(
                        data=data,
                        columns=columns,
                        page_size=5,
                        style_table={'overflowY': 'auto', 'overflowX': 'scroll', 'height': '20%'}, # Scrollable table height
                    ),
                    #footer
        ])
        return div, {'display': 'none'}, True, download_button_style
    
    # Default return if no specific button click (could be useful if other conditions were added)
    return None, {'display': 'block'}, dash.no_update, download_button_style # Show log display if no button click

# Callback to update page content based on upload file button clicks
@app.callback(
    Output('modal-figure', 'figure', allow_duplicate=True),
    Output("modal", "is_open", allow_duplicate=True),
    Input('modal-expnode', 'n_clicks'),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True  # Ensure callback doesn't run on load
)
def toogle_modal_exp_dl(exp_n, closemodal_n):
    
    global saved_figures 

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Default figure to prevent any issues if nothing matches
    fig = dash.no_update
    is_open = False 
    if button_id == 'close-modal':
        return dash.no_update, False
    elif button_id == 'modal-expnode':
        if "expnode" in saved_figures.keys():
            if exp_n != 0:
                fig = saved_figures['expnode']
                is_open = True
        else:
            is_open = False
    else:
        is_open = False
    return fig, is_open

# Callback to update page content based on upload file button clicks
@app.callback(
    Output('modal-figure', 'figure', allow_duplicate=True),
    Output("modal", "is_open", allow_duplicate=True),
    Input('modal-shapexp', 'n_clicks'),
    Input('modal-shapsum', 'n_clicks'),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True  # Ensure callback doesn't run on load
)
def toogle_modal_exp_ml(shapexp_n, shapsum_n, closemodal_n):
    
    global saved_figures 

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Default figure to prevent any issues if nothing matches
    fig = dash.no_update
    is_open = False 

    if button_id == 'close-modal':
        return dash.no_update, False
    elif button_id == 'modal-shapexp':
        if shapexp_n != 0:
            fig = saved_figures['shap-exp']
            is_open = True 
    elif button_id == 'modal-shapsum':
        if shapsum_n != 0:
            fig = saved_figures['shap-sum']
            is_open = True 
    else:
        is_open = False
    return fig, is_open

# Callback to update page content based on upload file button clicks
@app.callback(
    Output('modal-figure', 'figure', allow_duplicate= True),
    Output("modal", "is_open", allow_duplicate=True),
    Input('modal-pca', 'n_clicks'),
    Input('modal-tsne', 'n_clicks'),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True  # Ensure callback doesn't run on load
)
def toogle_modal_emb(pca_n, tsne_n, closemodal_n):
    
    global saved_figures 

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
   
    # Default figure to prevent any issues if nothing matches
    fig = dash.no_update

    is_open = False 

    if button_id == 'close-modal':
        return dash.no_update, False
    elif button_id == 'modal-pca':
        if pca_n != 0:
            fig = saved_figures['pca-plot']
            is_open = True
    elif button_id == 'modal-tsne':
        if tsne_n != 0:
            fig = saved_figures['tsne-plot']
            is_open = True 
    else:
        is_open = False
    return fig, is_open


# Callback to update page content based on upload file button clicks
@app.callback(
    Output('modal-figure', 'figure', allow_duplicate=True),
    Output("modal", "is_open", allow_duplicate=True),
    Input('modal-roctest', 'n_clicks'),
    Input('modal-confm', 'n_clicks'),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True  # Ensure callback doesn't run on load
)
def toogle_modal_analysis_dl(roctest_n, confm_n, closemodal_n):
    
    global saved_figures 

    print("open modal callback called")
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Default figure to prevent any issues if nothing matches
    fig = dash.no_update
    is_open = False 

    if button_id == 'close-modal':
        return dash.no_update, False
    elif button_id == 'modal-roctest':
        if roctest_n != 0:
            fig = saved_figures['roc-test']
            is_open = True 
    elif button_id == 'modal-confm':
         if confm_n != 0:
            fig = saved_figures['confm']
            is_open = True 
    else:
        is_open = False
    return fig, is_open

# Callback to update page content based on upload file button clicks
@app.callback(
    Output('modal-figure', 'figure', allow_duplicate=True),
    Output("modal", "is_open", allow_duplicate=True),
    Input('modal-boxplot', 'n_clicks'),
    Input('modal-roc', 'n_clicks'),
    Input('modal-roctest', 'n_clicks'),
    Input('modal-confm', 'n_clicks'),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True  # Ensure callback doesn't run on load
)
def toogle_modal_analysis_ml(boxp_n, roc_n, roctest_n, confm_n, closemodal_n):
    
    global saved_figures 

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Default figure to prevent any issues if nothing matches
    fig = dash.no_update
    is_open = False 

    if button_id == 'close-modal':
        return dash.no_update, False
    elif button_id == 'modal-boxplot':
        if boxp_n != 0:
            fig = saved_figures['boxplot']
            is_open = True 
    elif button_id == 'modal-roc':
        if roc_n != 0:
            fig = saved_figures['roc']
            is_open = True 
    elif button_id == 'modal-roctest':
        if roctest_n != 0:
            fig = saved_figures['roc-test']
            is_open = True 
    elif button_id == 'modal-confm':
        if confm_n != 0:
            fig = saved_figures['confm']
            is_open = True 
    else:
        is_open = False
    return fig, is_open



# Callback to update page content based on analyze button clicks
@app.callback(
    Output('model-dropdown', 'options'),
    Input("analysis-type", "value"),
)
def update_dropdown_models(method):
    
    global models 
    
    # Debugging print statements
    print(f"Changing Model Dropdown options")
    
    if "Temp" not in models.keys():
        options =  [{'label': i, 'value': i} for i in analysis_options[method]]
        return options
    else:
        return [{'label': None, 'value': None}]

# Callback to update the file name
@app.callback(
    Output('upload-data', 'children'),
    Input('upload-data', 'filename'), 
    Input('upload-data', 'contents')
)
def update_gene_filename(filename, file):
    global gene_filename
    global gene_fileg
    
  
    if filename is not None:
        if file is not None:
            gene_filename=filename
            gene_fileg=file
            return f"Selected file: {filename}"
        else:    
            return "Drag and Drop or 'Select Files'"
    else:
        if gene_filename is not None:
           return f"Selected file: {gene_filename}"
        else:    
            return "Drag and Drop or 'Select Files'"
        
@app.callback(
    Output("log-display", "children"),
    Output("interval-component", "disabled"), 
    Input("interval-component", "n_intervals"), 
    prevent_initial_call = True
)
def update_log_display(n_intervals):
    global log_messages
    global progress
    
    log_texts = " \n".join(log_messages)
    
    if progress < 100:
        return log_texts, False
    else: 
        return log_texts, True 
    
# Callback to update page content based on compute embeddings button clicks
@app.callback(
    Output("output-data-embedding", "children"),
    Input("embeddings-button", "n_clicks"),
)
def update_embeddings_onclick(embeddings_nclicks):
    global expr
    # Debugging print statements
    print(f"Button clicks Embeddings: {embeddings_nclicks}")

    # Prioritize button clicks over pathname
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'embeddings-button':
        if expr is not None:
            div = update_embeddings(True)
        else:
            div = html.Div([
                html.Br(),
                html.Br(),
                html.P("Upload a gene expression file first.",
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center', 'fontWeight': 'bold'})
            ])
        return div 

    return dash.no_update

# Callback to download the preproccessed file button clicks
@app.callback(
    Output("download-csv", "data"),
    Input("download-button", "n_clicks"),
)
def download_onclick(download_nclicks):
    
    global expr 

    # Debugging print statements
    print(f"Button clicks - Download: {download_nclicks}")

    # Prioritize button clicks over pathname
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'download-button':
        if expr is not None:
            return dcc.send_data_frame(expr.to_csv, "preprocessed_data.csv")


# Callback to update page content based on analyze button clicks
@app.callback(
    Output("analysis-output", "children"),
    Input("analyze-button", "n_clicks"),
    Input("analysis-method", "value"),
)
def update_analyze_onclick(analyze_nclicks, method):
    
    global method_g
    
    method_g = method 
    
    # Debugging print statements
    print(f"Button clicks Analyze: {analyze_nclicks}")

    # Prioritize button clicks over pathname
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'analyze-button':
        if expr is not None:
            div = update_analysis_output()
            return div
        else:
            div = html.Div([
                html.Br(),
                html.Br(),
                html.Br(),
                html.P("Upload a gene expression file first.",
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center', 'fontWeight': 'bold'})
            ])
            return div 
    return dash.no_update

# Callback to update page content based on analyze button clicks
@app.callback(
    Output("explain-output", "children"),
    Input("explain-button", "n_clicks"),
    Input("model-dropdown", "value")
)
def update_explain_onclick(explain_nclicks, model_name_in):

    global X_test, y_test, expr 
    global model_name

    model_name = model_name_in 
    # Debugging print statements
    print(f"Button clicks Explain: {explain_nclicks}")

    # Prioritize button clicks over pathname
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'explain-button':
        if expr is not None:
            if model_name in analysis_options['ML']:
                div = update_explainability_ml(model_name)
            else:
                div = update_explainability_dl(model_name)
            return div
        else:
            div = html.Div([
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.P("Upload a gene expression file first.",
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center', 'fontWeight': 'bold'})
            ])
        return div 
    return dash.no_update

default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'content': 'data(label)'
         }
    },
    {
        'selector': 'edge',
        'style': {
            'line-color': '#A3C4BC'
        }
    },
      # Class selectors
    {
        'selector': '.Anomalous',
            'style': {
                'background-color': 'red',
            }
        },
        {
            'selector': '.Control',
                'style': {
                    'background-color': 'green',
            }
        }
]

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll',
        'width': '50%',
        'height': '20%',
        'display': 'inline-block'
    }
}

@callback(Output('network', 'layout'),
              Input('dropdown-update-layout', 'value'))
def update_layout(layout):
    return {
        'name': layout,
        'animate': True
    }

@callback(Output('network', 'elements'),
         Input('btn-add-node', 'n_clicks_timestamp'),
         Input('btn-remove-node', 'n_clicks_timestamp'),
         State('network', 'elements'),
         prevent_initial_call = True)
def update_elements(btn_add, btn_remove, elements):
    current_nodes, deleted_nodes = get_current_and_deleted_nodes(elements)
    # If the add button was clicked most recently and there are nodes to add
    if int(btn_add) > int(btn_remove) and len(deleted_nodes):

        # We pop one node from deleted nodes and append it to nodes list.
        current_nodes.append(deleted_nodes.pop())
        # Get valid edges -- both source and target nodes are in the current graph
        cy_edges = get_current_valid_edges(current_nodes, edges)
        return cy_edges + current_nodes

    # If the remove button was clicked most recently and there are nodes to remove
    elif int(btn_remove) > int(btn_add) and len(current_nodes):
            current_nodes.pop()
            cy_edges = get_current_valid_edges(current_nodes, edges)
            return cy_edges + current_nodes

    # Neither have been clicked yet (or fallback condition)
    return elements

def get_current_valid_edges(current_nodes, all_edges):
    """Returns edges that are present in Cytoscape:
    its source and target nodes are still present in the graph.
    """
    valid_edges = []
    node_ids = {n['data']['id'] for n in current_nodes}

    for e in all_edges:
        if e['data']['source'] in node_ids and e['data']['target'] in node_ids:
            valid_edges.append(e)
    return valid_edges

def get_current_and_deleted_nodes(elements):
    """Returns nodes that are present in Cytoscape and the deleted nodes
    """
    current_nodes = []
    deleted_nodes = []

    # get current graph nodes
    for ele in elements:
        # if the element is a node
        if 'source' not in ele['data']:
            current_nodes.append(ele)

    # get deleted nodes
    node_ids = {n['data']['id'] for n in current_nodes}
    for n in nodes:
        if n['data']['id'] not in node_ids:
            deleted_nodes.append(n)

    return current_nodes, deleted_nodes

@callback(Output('cytoscape-tapEdgeData-json', 'children'),
          Input('network', 'tapEdgeData'),
          prevent_initial_call = True)
def displayTapEdgeData(data):
    if data:
        return "You recently clicked/tapped the edge between " + \
               data['source'].upper() + " and " + data['target'].upper() + " with weight: " + str(data['weight'])
    else:
        return "Click one edge to visualize its information"

@callback(Output('exp-node', 'figure'),
          Input('network', 'tapNodeData'),
          prevent_initial_call = True
          )
def updateExpOnTapNode(node_data):

    global genes 
    global node_mapping 
    global mydataloader 
    global explainer 
    global expr 
    global saved_figures

    if node_data:

        node = node_data["id"]
        id = node_mapping[node]
        genes = expr.columns
        fig = dl.plotly_featureimportance_from_gnnexplainer(explainer, mydataloader, id, genes)
        saved_figures['expnode'] = fig
        return fig

@callback(
          Output('subnetwork', 'elements'),
          Input('network', 'tapNodeData'),
          prevent_initial_call = True
          )
def updateSubgraphOnTapNode(node_data):

    global edges 
    global nodes 
    global genes 
    global node_mapping 
    
    if node_data:

        elements = dl.get_subgraph(node_data, nodes, edges)
        return elements

@callback(Output('cytoscape-tapNodeData-json', 'children'),
          Input('network', 'tapNodeData'),
          prevent_initial_call = True
          )
def displayTextOnTapNode(node_data):
    if node_data:
        text = json.dumps(node_data, indent=2)
    else:
        text = json.dumps('Click a node to get more information')
    return text


@app.callback(
    Output("force-plot-div", "children"),
    Input("dropdown", "value"),
    prevent_initial_call = True
)
def update_forceplot(patient_idx):

    global models 
    global model_name 
    global X_test, y_test 
    global X_train, y_train 
    global genes
    global targets 

    model = models[model_name]

    patient = patients[patient_idx]
    ys = targets.values
    X = np.concatenate((X_train, X_test), axis=0)
    y = ys[patient_idx][0]
    plot_title = "Shap Force plot for model {} and patient {} with class {}".format(model_name, patient, map_final[y])
    
    return ml.get_plot("force-plot", {}, model, model_name, X, ys, genes, index = patient_idx, top_n = 10, X_train = X_train, class_id = y, title=plot_title)

if __name__ == '__main__':
        
    print("Running ADIN")
    webbrowser.open(html_plot_path)    
    app.run_server(debug=False)
    




#EXPLAINABILITY E DL: add only GraphSVX shap 
#fix preprocessing
#FIX THE FOOTER