"""
***************************************************************************************************************************************************************
*                                                                                                                                                             *
*   E-ABIN: an Explainable module for Anomaly detection in BIological Networks                                                                                *
*   Developed by Lomoio Ugo                                                                                                                                   *  
*   2024                                                                                                                                                      *
*                                                                                                                                                             *
***************************************************************************************************************************************************************
"""

import networkx as nx   
import numpy as np
import os  

import platform

if platform.system() == "linux":
    import cudf.pandas
    cudf.pandas.install()

import pandas as pd 
import webbrowser
import dash
from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table import DataTable
import io 
import base64
import random 
import sys 
#import multiprocessing

from sklearn.metrics import confusion_matrix
from adin import utils, ml, dl, preprocessing, gaan_config, ml_config, gcn, gaan, captum_explainations 
from adin.gae import GAE_Explainable
from dash import html, dcc
import torch
from collections import deque
import dash_cytoscape as cyto
from dash import Dash, html, Input, Output, State, callback, dcc
import json 
from torch_geometric.explain import Explainer, GNNExplainer
import traceback
import joblib 
from adin.utils import set_seed

#from parallel_pandas import ParallelPandas
# Dynamically determine the number of CPU cores
#n_cpu = multiprocessing.cpu_count()
#initialize parallel-pandas
#ParallelPandas.initialize(n_cpu=n_cpu, split_factor=4, disable_pr_bar=True)

print("Detected Operative System {}".format(platform.system()))

torch.cuda.empty_cache()

uploads_dir = os.path.join(os.getcwd(), "use_case", "uploads")
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Initialize the Dash app
html_plot_path = "http://127.0.0.1:8050/"
app = dash.Dash(__name__, suppress_callback_exceptions=False, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Explainable Anomaly detection in BIological Networks"

# Global variables to hold the data
log_messages = deque(maxlen=1)  
gene_data = None
edge_index = None
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
map_final = {0: "Normal", 1: "Anomalous"}
gaan_params = None 
ml_params = None 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
isns = None 
explainers = {}
genes_isn = None 
interaction_df = None 
contamination = None 
non_deterministic_seed = random.randrange(sys.maxsize)
current_seed = non_deterministic_seed
is_deterministic = True
drop_ratio = 0.00
grid_search = True
python_version = sys.version.split()[0]
python_version_major = int(python_version.split(".")[0])
python_version_minor = int(python_version.split(".")[1])


if python_version_major <= 2:
    raise Exception("Python 2 is not supported. Please use Python 3 or higher.")

if python_version_major == 3 and python_version_minor >= 13:
    print("Warning: Python version {}.{} is not officially supported. Removing ISNs features for compatibility.".format(python_version_major, python_version_minor))
    analysis_options = {
        'ML': ['KNN', 'LR', 'DT', 'SVM', 'RF'], #['LDA', 'NB', 'KNN', 'LR', 'DT', 'SVM', 'RF']
        'GAAN (node)': ['GAAN_node', 'GAE', 'GCN']
    }
else:
    print("Python version {}.{} is supported.".format(python_version_major, python_version_minor))
    # Keep all features available
    analysis_options = {
        'ML': ['KNN', 'LR', 'DT', 'SVM', 'RF'], #['LDA', 'NB', 'KNN', 'LR', 'DT', 'SVM', 'RF']
        'GAAN (node)': ['GAAN_node', 'GCN', 'GAE'],
        'GAAN (ISNs)': ['GAAN_isn']
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

gpu = 0 if torch.cuda.is_available() else -1
n_gpus = torch.cuda.device_count() - 1

def set_deterministic(seed):
    global current_seed
    global is_deterministic

    #FOR REPRODUCIBILITY:
    print(f"SETTING DETERMINISTIC MODE FOR REPRODUCIBILITY, SEED: {seed}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    torch.use_deterministic_algorithms(True)
    set_seed(seed)
    current_seed = seed 
    is_deterministic = True

def unset_deterministic():
    global non_deterministic_seed
    global current_seed 
    global is_deterministic

    print(f"NOT DETERMINISTIC MODE IS USED. NO REPRODUCIBILITY GARANTEED")
    torch.use_deterministic_algorithms(False)
    set_seed(non_deterministic_seed)
    current_seed = non_deterministic_seed
    is_deterministic = False

set_deterministic(0)


gaan_params_layout = html.Div([

    # Header
    html.H2("GAAN Model Parameters"),

    # Input for noise_dim (int)
    html.Label("Noise Dimension (noise_dim):"),
    dcc.Input(id='noise_dim', type='number', value=64, min=1, step=1),
    html.Br(),

    # Input for hid_dim (int)
    html.Label("Hidden Dimension (hid_dim):"),
    dcc.Input(id='hid_dim', type='number', value=128, min=1, step=1),
    html.Br(),

    # Input for num_layers (int)
    html.Label("Number of Layers (num_layers):"),
    dcc.Input(id='num_layers', type='number', value=2, min=2, step=1),
    html.Br(),

    # Input for dropout (float)
    html.Label("Dropout Rate (dropout):"),
    dcc.Input(id='dropout', type='number', value=0.3, min=0.0, max=1.0, step=0.01),
    html.Br(),

    # Input for contamination (float)
    html.Label("Contamination (contamination):"),
    dcc.Input(id='contamination', type='number', value=contamination, min=0.0, max=0.5, step=0.01),
    html.Br(),

    # Input for learning rate (lr)
    html.Label("Learning Rate (lr):"),
    dcc.Input(id='lr', type='number', value=0.00005, min= 0.0000001, max = 0.5, step=0.00001),
    html.Br(),

    # Input for epoch (int)
    html.Label("Number of Epochs (epoch):"),
    dcc.Input(id='epoch', type='number', value=200, min=1, step=1),
    html.Br(),

    # Input for GPU index (gpu)
    html.Label("GPU Index (gpu):"),
    dcc.Input(id='gpu', type='number', value=gpu, min=-1, max=n_gpus, step=1),
    html.Br(),

    # Input for batch_size (int)
    html.Label("Batch Size (batch_size):"),
    dcc.Input(id='batch_size', type='number', value=1, min=1, step=1),
    html.Br(),

    # Slider for verbosity (verbose)
    html.Label("Verbosity Mode (verbose):"),
    dcc.Slider(
        id='verbose',
        min=0,
        max=3,
        step=1,
        value=1,
        marks={i: str(i) for i in range(4)}
    ),
    html.Br(),

    html.Label("Threshold Similarity:"),
    dcc.Input(id='th', type='number', value=0.93, min=0.5, max=1.0, step=0.01),
    html.Br(),

    html.Label("Deterministic Mode:"),
    dcc.Dropdown(
                id="deterministic-dl",
                options=[
                    {"label": "yes", "value": True},
                    {"label": "no", "value": False},
                ],
                value=True,
    ),
    html.Br(),

    html.Label("Deterministic Seed:"),
    dcc.Input(id='seed-dl', type='number', value=0, min=0, max=2**64-1,step=1),
    html.Br()

])

# Define collapsible layout for each algorithm's parameters
def get_lr_params():
    return dbc.Collapse(
        dbc.CardBody([
            html.H5("Linear Regression Parameters", className="card-title"),
            html.Label("Solver:"),
            dcc.Dropdown(
                id="lr-solver",
                options=[
                    {"label": "lbfgs", "value": "lbfgs"},
                    {"label": "liblinear", "value": "liblinear"},
                    {"label": "newton-cg", "value": "newtow-cg"},
                    {"label": "newton-cholesky", "value": "newton-cholesky"},
                    {"label": "sag", "value": "sag"},
                    {"label": "saga", "value": "saga"},
                ],
                value="lbfgs",
            ),
            html.Label("Penality:"),
            dcc.Dropdown(
                id="lr-penality",
                options=[
                    {"label": "None", "value": None},
                    {"label": "l1", "value": "l1"},
                    {"label": "l2", "value": "l2"},
                    {"label": "elasticnet", "value": "elasticnet"},
                ],
                value="l2",
            ),
            html.Label("Max iter:"),
            dcc.Input(type="number", id="lr-max-iter", value=100, step=1),
        ]),
        id="collapse-lr",
        is_open=True,
    )

def get_svm_params():
    return dbc.Collapse(
        dbc.CardBody([
            html.H5("SVM Parameters", className="card-title"),
            html.Label("Kernel:"),
            dcc.Dropdown(
                id="svm-kernel",
                options=[
                    {"label": "Linear", "value": "linear"},
                    {"label": "Polynomial", "value": "poly"},
                    {"label": "RBF", "value": "rbf"},
                    {"label": "Sigmoid", "value": "sigmoid"},
                ],
                value="linear",
            ),
            html.Label("C (Regularization Parameter):"),
            dcc.Input(type="number", id="svm-c", value=1, step=0.1),
        ]),
        id="collapse-svm",
        is_open=True,
    )

def get_knn_params():
    return dbc.Collapse(
        dbc.CardBody([
            html.H5("K-Nearest Neighbors Parameters", className="card-title"),
            html.Label("Number of Neighbors (K):"),
            dcc.Input(type="number", id="knn-k", value=5, min=1, step=1),
            html.Label("Metric:"),
            dcc.Dropdown(
                id="knn-metric",
                options=[
                    {"label": "Euclidean", "value": "euclidean"},
                    {"label": "Manhattan", "value": "manhattan"},
                    {"label": "Minkowski", "value": "minkowski"},
                ],
                value="euclidean",
            ),
        ]),
        id="collapse-knn",
        is_open=True,
    )

def get_rf_params():
    return dbc.Collapse(
        dbc.CardBody([
            html.H5("Random Forest Parameters", className="card-title"),
            html.Label("Number of Estimators:"),
            dcc.Input(type="number", id="rf-n-estimators", value=100, min=1, step=1),
            html.Label("Max Depth:"),
            dcc.Input(type="number", id="rf-max-depth", value = -1, min=-1, step=1),
        ]),
        id="collapse-rf",
        is_open=True,
    )

def get_dt_params():
    return dbc.Collapse(
        dbc.CardBody([
            html.H5("Decision Tree Parameters", className="card-title"),
            html.Label("Max Depth:"),
            dcc.Input(type="number", id="dt-max-depth", value=-1, min=-1, step=1),
            html.Label("Min Samples Split:"),
            dcc.Input(type="number", id="dt-min-samples-split", value=2, min=2, step=1),
        ]),
        id="collapse-dt",
        is_open=True,
    )

def get_split_params():
    return dbc.Collapse(
        dbc.CardBody([
            html.H5("Train-Test split parameters", className="card-title"),
            html.Label("Train Split:"),
            dcc.Input(type="number", id="split-train", min=0.2, max=0.95, value = 0.7),
            html.Label("Cross Validation Splits:"),
            dcc.Input(type="number", id="cv-split", value=5, min=2, max = 10, step=1),
        ]),
        id="collapse-split",
        is_open=True,
    )

# Assemble all parameter collapsibles in the modal body
ml_params_layout = html.Div([
    get_split_params(),
    get_lr_params(),
    get_svm_params(),
    get_knn_params(),
    get_rf_params(),
    get_dt_params(),
    
    html.Label("Deterministic Mode:"),
    dcc.Dropdown(
                id="deterministic-ml",
                options=[
                    {"label": "yes", "value": True},
                    {"label": "no", "value": False},
                ],
                value=True,
    ),
    html.Br(),

    html.Label("Deterministic Seed:"),
    dcc.Input(id='seed-ml', type='number', value=0, min=0, max=2**64-1,step=1),
    html.Br(),

    html.Label("Grid Search:"),
    dcc.RadioItems(
        id='grid-search',
        options=[
            {'label': 'Activate Grid Search', 'value': True},
            {'label': 'Deactivate Grid Search', 'value': False},
        ],
        value=True,  # Default value
        inline=True,
        style={'margin-top': '10px'}
    )
])



navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Upload", href="/upload", id='upload-nav', n_clicks=0, style={"font-weight": "bold", 'color':'orange'})),
            dbc.NavItem(dbc.NavLink("Embeddings", href="/embeddings", id='embeddings-nav', n_clicks=0, style={"font-weight": "bold", 'color':'orange'})),
            dbc.NavItem(dbc.NavLink("Analysis", href="/analysis", id='analysis-nav', n_clicks=0, style={"font-weight": "bold", 'color':'orange'})),
            dbc.NavItem(dbc.NavLink("Explainability", href="/explain", id='explain-nav', n_clicks=0, style={"font-weight": "bold", 'color':'orange'}))
        ],
        brand="E-ABIN",
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
                    'margin': '0px',
                    'font-weight': 'bold'
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
                    html.Img(src='/assets/unicz.png', style = {'width': '100%', 'height': '100px'}),
                ], style = {
                    'width': '12%',  # Adjust the width of the image container
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'background-color': '#EBEBEB',
                    }
                ),
                html.Div([
                    html.P(["Lomoio Ugo", html.Sup("1, 2"), ", Tommaso Mazza", html.Sup(3), ", Pierangelo Veltri", html.Sup(4), ", and Pietro Hiram Guzzi", html.Sup(1), html.Br(), "(1) Magna Graecia University of Catanzaro, (2) Relatech SpA", html.Br(), "(3) IRCCS Casa Sollievo della Sofferenza, (4) University of Calabria Rende", html.Br()]),
                    ], style= {
                        'width': '70%',  # Adjust the width of the image container
                        'vertical-align': 'middle',
                        'textAlign': 'center',
                        'background-color': '#EBEBEB',
                        'display': 'inline-block'
                    }
                ),
                html.Div([
                    html.Img(src='/assets/unical.png', style = {'width': '100%', 'height': '100px'}),  
                    ], style = {
                        'width': '18%',  # Adjust the width of the image container
                        'display': 'inline-block',
                        'vertical-align': 'top',
                        'background-color': '#EBEBEB',
                    }
                ),
            ], style = {'width': '100%',  # Ensure the row occupies full width
                        'height': '100px',
                        'textAlign': 'center',
                        'background-color': '#EBEBEB'})
        ])


download_button_style = {'position': 'absolute', 'top': '30%', 'right': '5%', 'width': '10%', 
                         'background-color': '#34B212', 'height': '8%', 'text-align': 'center', 'display':'none'}

upload_layout = html.Div([
    
    html.Div([    
        html.H4("Load Dataset", style={'position': 'absolute', 'left': '1%'}),
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
            ),            
            html.Br(),
            html.Div("NaN drop ratio:", style={'display': 'inline-block', 'padding-left': '10px', 'padding-right': '10px'}),
            dcc.Input(id='drop-ratio', type='number', value=1.0, min=0.0, max=1.0, step=0.01),
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


submit_style = {'position': 'absolute', 'bottom': '10%', 'right': '45%', 'width': '10%','background-color': '#34B212', 'height': '5%', 'text-align': 'center'}

modal_hyper = dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Config Params")),
                        dbc.ModalBody(gaan_params_layout, id="modal-body-content"),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-modal-params", n_clicks=0),
                        ),
                        # Submit button
                     
                        html.Button("Submit", id='submit-params-button-gaan', n_clicks=0, style=submit_style, hidden=False),
                        html.Button("Submit", id='submit-params-button-ml', n_clicks=0, style=submit_style, hidden=True),

                        # Placeholder for displaying the input values
                        html.Div(id='output-container')    
                    ],
                    id="modal-hyper",
                    is_open=False,
                    fullscreen=True,
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
])

analysis_layout = html.Div([
    html.Div([
        html.H2("Select the Analysis Method:", id='text-analysis', style={'position': 'absolute', 'top': '8%', 'left': '0%', 'width': '100%', 'textAlign': 'center'}),
        dcc.Dropdown(
            id='analysis-method',
            options=[
                {'label': 'ML Binary Classification', 'value': 'ML'},
                {'label': 'Node Anomaly Detection', 'value': 'GAAN (node)'},
                {'label': 'Graph Anomaly Detection', 'value': 'GAAN (ISNs)'} if python_version_minor < 13 else None
            ],
            value='ML', 
            style={'position': 'absolute', 'top': '13%', 'left': '16%', 'width': '70%', 'textAlign': 'center'}
        ),
        html.Div([
            dbc.Button("Analyze", id='analyze-button', style={
                'display': 'inline-block', 'textAlign': 'center', 'background-color': '#34B212'
            }, n_clicks=0),
            dbc.Button("Config Params", id='modal-params-button', disabled=False, style={
                'display': 'inline-block','textAlign': 'center', 'background-color': '#34B212'
            }, n_clicks=0),
        ], style={'position':'absolute', 'left': '5%', 'display': 'inline-block'}),
        
        dbc.Button("Save Models", id="save-models-button", color="primary", n_clicks=0, style={'position': 'absolute', 'top': '13%', 'right': '8%', 'width': '10%', 'textAlign': 'center'}),
        html.Div(id="save-models-output", style={'position': 'absolute', 'top': '16%', 'right': '8%', "color": "green"}),

        modal_hyper,
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
                    {'label': 'Node Anomaly Detection', 'value': 'GAAN (node)'},
                    {'label': 'Graph Anomaly Detection', 'value': 'GAAN (ISNs)'} if python_version_minor < 13 else None
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


def upload_preprocessed_file(gene_file, filename):
    
    global gene_data
    global expr 
    global targets 
    global limited_expr
    global progress
    global patients 
    global dataset_path

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
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center'}),
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
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center'}),
            ])
        
        progress = 40
        targets_uq = np.unique(targets['Target'].values)
        if len(targets_uq) > 2:
            raise Exception("Target column must have only 2 unique values to perform Anomaly Detection and Binary Classification. Found {} unique targets.".format(len(targets_uq)))


        df = expr.join(targets)
        if gene_filename not in os.listdir(uploads_dir):
            os.makedirs(os.path.join(uploads_dir, gene_filename.split('.')[0]), exist_ok=True)
        if not os.path.exists(os.path.join(uploads_dir, gene_filename)):
            df.to_csv(os.path.join(uploads_dir, gene_filename), index=True)

        progress = 90
        if "Target" in expr.columns:
            expr = expr.drop("Target", axis=1)
        
        limited_expr = expr.iloc[:, :50]
        limited_expr = limited_expr.join(targets)
        limited_expr = limited_expr.T
        limited_expr = limited_expr.reset_index()
        limited_expr = limited_expr.rename(columns = {'index': 'Gene'})
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
                footer 
        ])
    
def upload_file(gene_file, filename):
    
    global gene_data
    global expr 
    global targets 
    global limited_expr
    global progress 
    global patients 
    global drop_ratio

    progress = 0
    if gene_file is not None:
        
        print("Uploading Gene File")
        add_log_message("Uploading Gene File")
        content_type, content_string = gene_file.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Assume that the user uploaded a CSV file
            skiprows, gene_data = utils.read_gene_expression(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            traceback_str = traceback.format_exc()
    
            return html.Div([
                html.P('There was an error processing this gene file. {} \n {}'.format(e, traceback_str),
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center'}),
            ])
        progress = 10

        if skiprows == 0:
            return html.Div([
                html.P("There was an error processing this gene file. Can't find automatically a skiprow value to read the file",
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center'}), 
            ])

        print("Detecting Geo accession code")
        add_log_message("Detecting Geo accession code")
        geo_code = utils.find_geoaccession_code(io.StringIO(decoded.decode('utf-8')), skiprows)
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
        
        # Convert all columns to numeric, invalid entries become NaN
        expr = expr.apply(pd.to_numeric, errors='coerce')
        
        #print("Warning: PREPROCESSING IN DEBUG MODE, ONLY 50 GENES ARE USED")
        #expr = expr.iloc[:, :50]

        #print("Expr:", expr)
        print("Targets:", targets)
        df = expr.join(targets)
    
        print("Preprocessing")
        add_log_message("Renaming columns & Replacing none values.")
        #print(annotation_df.shape, annotation_df.columns)
       
        expr = preprocessing.preprocess(df, annotation_df, rate_drop=drop_ratio, need_rename=True) 
        progress = 90
        #print("Preprocessed:", expr)

        if gene_filename not in os.listdir(uploads_dir):
            os.makedirs(os.path.join(uploads_dir, gene_filename.split('.')[0]), exist_ok=True)
        if not os.path.exists(os.path.join(uploads_dir, gene_filename)):
            df.to_csv(os.path.join(uploads_dir, gene_filename), index=True)

        if "Target" in expr.columns:
            expr = expr.drop("Target", axis=1)
        
        targets_uq = np.unique(targets['Target'].values)
        if len(targets_uq) > 2:
            raise Exception ("Target column must have only 2 unique values to perform Anomaly Detection and Binary Classification. Found {} unique targets.".format(len(targets_uq)))
       
        
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
                footer
        ])

def create_div_captum(model_name):

    global models 
    global mydataloader
    global edge_list 

    if model_name in models.keys():
        if model_name not in ["GCN"]:
            div = html.Div([html.P("Captum explainations are not available for {}.".format(model_name), style={'color': 'red', 'fontWeight': 'bold'})])
            return div
        else:
            G = nx.from_edgelist(edge_list)       
            model = models[model_name]
            img1, img2 = captum_explainations.explain_with_captum(model, model_name, mydataloader, G)
            div = html.Div([
                html.Div([
                    html.P("Integrated Gradients", style={'text-align': 'center', 'font-weight': 'bold'}),
                    html.Img(src=img1, style={'width': '90%', 'display': 'block', 'margin': '0 auto'}),
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.P("Saliency Maps", style={'text-align': 'center', 'font-weight': 'bold'}),
                    html.Img(src=img2, style={'width': '90%', 'display': 'block', 'margin': '0 auto'}),
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
            ], style={'text-align': 'center'})
            return div

def create_div_exp(captum_div, model_name):

    global models 
    global mydataloader 
    global patients 
    global edge_list
    global nodes 
    global edges 
    if model_name in [""]:
        return html.Div(html.P(
            "Explanations are not available for {} models.".format(model_name),
            style={'color': 'red', 'fontWeight': 'bold'}
        ))
    else:
        model = models[model_name]
        y = mydataloader.y
        if model_name == "GCN":
            preds = gcn.test(model, mydataloader)
        else:
            preds = model(mydataloader.x, mydataloader.edge_index)

        nodes, edges = dl.compute_elements_cyto(patients, edge_list, None, y.cpu(), preds.cpu(), isn = False)
        print(f"Number of Nodes: {len(nodes)}, Number of Edges: {len(edges)}")
        return  html.Div([
                
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
                            dbc.Button("Enlarge Figure", id = "modal-expnode", n_clicks=0),

                            captum_div, 

                            html.P("Subgraph:"),
                            cyto.Cytoscape(
                                id='subnetwork',
                                elements=edges+nodes,
                                layout={'name': 'breadthfirst'},
                                style={'width': '100%', 'height': '400px'},
                                stylesheet=default_stylesheet,
                                responsive=True
                            ),
                            footer
                    ], style = {'position': "absolute", "width": "100%"})

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
                            ),
                            footer
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
                        dbc.Button("Enlarge Figure", id = "modal-shapexp", n_clicks=0),
                    ], style = {'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(
                            id = "shap-sum",
                            figure=fig_sum,
                            config = config,
                            style = {'display': 'inline-block'}
                        ),
                        html.Br(),
                        dbc.Button("Enlarge Figure", id = "modal-shapsum",  n_clicks=0)
                
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
                ]),
                footer
            ])
    
    return html.Div()

def create_div_exp_isn(isn_index, captum_div, model_name):

    global models
    global isns 
    global mydataloader
    global edge_list 
    global nodes 
    global edges 
    global explainers 
    global genes_isn
    global node_mapping 
    global expr 
    global saved_figures
   
    isn = isns[isn_index]
    model = models[model_name]
    y = mydataloader.y
    preds = model.predict(mydataloader)

    genes_isn = []
    for edge in edge_list:
        node1, node2 = edge.split("_")
        if node1 not in genes_isn:
            genes_isn.append(node1)
        if node2 not in genes_isn:
            genes_isn.append(node2)
    print(len(genes_isn))

    target = y[isn_index].item()
    y = torch.tensor([target for i in range(len(genes_isn))])
    pred = preds[isn_index].item()
    preds = torch.tensor([pred for i in range(len(genes_isn))])
    nodes, edges = dl.compute_elements_cyto(genes_isn, edge_list, None, y, preds, isn = True)
    print(f"Number of Nodes: {len(nodes)}, Number of Edges: {len(edges)}")
    #explainer = explainers[model_name]
    #fig = dl.plotly_featureimportance_from_gnnexplainer(explainer, mydataloader, isn_index, genes_isn)
    #saved_figures['expnode'] = fig

    return html.Div([
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                # Dropdown to select graph
                dcc.Dropdown(
                    id='graph-selection-dropdown',
                    options=[{'label': idx, 'value': idx} for idx, graph in enumerate(isns)],
                    value=isn_index,  # Set the current graph as the selected value
                    clearable=False
                ),

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
                        elements=edges + nodes,
                        layout={'name': 'breadthfirst'},
                        style={'width': '100%', 'height': '400px'},
                        stylesheet=default_stylesheet,
                        responsive=True
                    ),
                    html.Div([
                        html.P("Clicked Node/Edge informations: "),
                        html.Pre(id='cytoscape-tapNodeData-json', children="Click one node to get more information", style=styles['pre']),
                        html.Pre(id='cytoscape-tapEdgeData-json', children="Click one edge to visualize its information", style=styles['pre']),
                    ]),
                ]),

                html.P("Subgraph:"),
                cyto.Cytoscape(
                    id='subnetwork',
                    elements=edges + nodes,
                    layout={'name': 'breadthfirst'},
                    style={'width': '100%', 'height': '400px'},
                    stylesheet=default_stylesheet,
                    responsive=True
                ),
                footer
            ], style={'position': "absolute", "width": "100%"})

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
    global isns 
    
    if model_name is not None:
        if model_name in list(models.keys()):
            if "isn" in model_name:
                captum_div = create_div_captum(model_name)
                return create_div_exp_isn(0, captum_div, model_name)
            else:
                captum_div = create_div_captum(model_name)
                return create_div_exp(captum_div, model_name)             
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
                                dbc.Button("Enlarge Figure", id = "modal-pca", n_clicks=0),
                            ], style = {'display': 'inline-block'}),
                                
                            html.Div([
                                dcc.Graph(id='tsne-plot', figure = fig_tsne, config = config, style={'top': '8%', 'right': '1%', 'height': '40%', 'width': '40%', 'display': 'inline-block'}),
                                html.Br(),
                                dbc.Button("Enlarge Figure", id = "modal-tsne", n_clicks=0)
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
                        footer
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
                                dbc.Button("Enlarge Figure", id = "modal-pca", n_clicks=0),
                            ], style = {'display': 'inline-block'}),
                                
                            html.Div([
                                dcc.Graph(id='tsne-plot', figure = fig_tsne, config = config, style={'top': '8%', 'right': '1%', 'height': '40%', 'width': '40%', 'display': 'inline-block'}),
                                html.Br(),
                                dbc.Button("Enlarge Figure", id = "modal-tsne",  n_clicks=0)
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
                        footer
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
    global explainers 
    global saved_figures 
    global ml_params 
    global gaan_params 
    global isns 
    global interaction_df
    global contamination 
    global is_deterministic
    global current_seed
    global grid_search

    if is_deterministic:
        set_deterministic(current_seed)
    else:
        unset_deterministic(current_seed)

    if method_g == 'ML':
        
        #if fig_box is None or fig_roc is None or fig_roc_test is None:
               
            df = pd.concat([expr, targets], axis=1)
            if ml_params is not None:
                train_size = ml_params.train_split
                test_size = 1.0 - train_size
            else:
                train_size = 0.7
                test_size = 0.3

            if ml_params is None:
                ml_params = ml_config.ML_config("lbfgs", "l2", 100, "linear", 1.0, 5, "euclidean", 100, -1, -1, 2, train_size, 5)

            dataset_path = os.path.join(uploads_dir, gene_filename.split('.')[0])
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path, exist_ok=True)
            X_train, X_test, y_train, y_test = ml.train_test_split(df, dataset_path, test_size = test_size)
            models_tuple, fig_roc, fig_box = ml.baselineComparison(X_train, y_train, params = ml_params, grid_search=grid_search)
            temp = {}
            for model_name, model in models_tuple:
                temp[model_name] = model

            if "Temp" not in models.keys():
                for model_name, model in models.items():
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
            fig_confm = dl.plot_cm(classes, cm, best_modelname)

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
                                    dbc.Button("Enlarge Figure", id = "modal-boxplot", n_clicks=0)
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
                                    dbc.Button("Enlarge Figure", id = "modal-roc", n_clicks=0)
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
                                    dbc.Button("Enlarge Figure", id = "modal-roctest", n_clicks=0),
                                ], style={'display': 'inline-block'}),
                                html.Div([
                                    dcc.Graph(
                                        id='conf-matrix',
                                        figure=fig_confm,
                                        config = config,
                                    ),
                                    html.Br(),
                                    dbc.Button("Enlarge Figure", id = "modal-confm",  n_clicks=0)
                                ], style = {'display': 'inline-block'}),
                            ]),
                            footer
                ], style = {'width': '100%'})
        
    elif method_g == 'GAAN (node)':
        
        print("Creating Graph from gene expression array")

        if gaan_params is not None:
            th = gaan_params.th
        else:
            th = 0.93

        if edge_index is None:

            edges_i, edge_list, _ = utils.get_edges_by_sim(expr, th=th)

            # Parse the edges and create node mapping
            source_nodes, target_nodes, node_mapping = utils.parse_edges(edges_i)
        
            print("Creating Dataloader")
            # Create edge_index tensor
        
            edge_index = utils.create_edge_index(source_nodes, target_nodes)

        x = expr.values
        y = targets.values
        mydataloader = dl.create_torch_geo_data(x, y, edge_index)

        print("Train - Test split")
        dataloader_train, dataloader_test = dl.train_test_split_and_mask(mydataloader, train_size = 0.7)
        in_dim = dataloader_train.x.shape[1]

        uqs, counts = np.unique(y, return_counts = True)
        dict_counts = {}
        for uq, count in zip(uqs, counts):
            dict_counts[uq.item()] = count.item()
        contamination = (dict_counts[1]/(dict_counts[0] + dict_counts[1]))*0.5
        #print(contamination)
        if gaan_params is not None:
            if gaan_params.contamination is None:
                gaan_params.contamination = contamination
            gaan_params.isn = False 
        else:
            gpu = 0 if torch.cuda.is_available() else -1
            gaan_params = gaan_config.GAAN_config(noise_dim=64, hid_dim=128, num_layers=2, dropout=0.3, contamination=contamination, lr = 0.00005, epoch = 200, gpu = gpu, batch_size=1, verbose = 1, isn = False, th = 0.93)
        
        attrs = vars(gaan_params)
        print(', '.join("%s: %s" % item for item in attrs.items()))

        print("Create GAAN model")
        model = dl.create_model(in_dim, gaan_params, isn = False)
        model = dl.train_gaan(model, dataloader_train)
        models["GAAN_node"] = model
        if "Temp" in models.keys():
            del models["Temp"]

        preds = model.predict(dataloader_test).cpu()
        y_test = dataloader_test.y.cpu().squeeze(-1)
        classes = {0: "Normal", 1: "Anomalous"}
        cm = confusion_matrix(y_test, preds)
        fig_cm = dl.plot_cm(classes, cm, "GAAN_node")
        fig_roc_test = ml.plot_roc_curve_(y_test, preds)
        saved_figures["confm"] = fig_cm 
        saved_figures["roc-test"] = fig_roc_test 
        explainer =  Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=0, lr = 0.001),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='node',
                    return_type='probs'
                ),
        )
        models["GAAN_node"] = model
        explainers["GAAN_node"] = explainer

        gpu = 0 if torch.cuda.is_available() else -1
       
        model_gae = GAE_Explainable(in_dim, device = device, hid_dim=gaan_params.hid_dim, num_layers=gaan_params.num_layers, dropout=gaan_params.dropout, contamination=gaan_params.contamination, lr=gaan_params.lr, epoch=gaan_params.epoch, verbose=gaan_params.verbose) #batch_size=gaan_params.batch_size
        model_gae.fit(dataloader_train)
        models["GAE"] = model_gae
        preds = model.predict(dataloader_test).cpu()
        cm = confusion_matrix(y_test, preds)
        fig_cm = dl.plot_cm(classes, cm, "GAE")
        fig_roc_test = ml.plot_roc_curve_(y_test, preds)
        saved_figures["confm-gae"] = fig_cm 
        saved_figures["roc-test-gae"] = fig_roc_test 
        explainer = Explainer(
                model=model_gae,
                algorithm=GNNExplainer(epochs=0, lr = 0.001),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='node',
                    return_type='raw'
                ),
        )
        models["GAE"] = model_gae
        explainers["GAE"] = explainer
       

        lr = gaan_params.lr
        hidden_dims = [128]
        inchannels = mydataloader.x.shape[1]
        model_gcn = gcn.GCN(inchannels, hidden_dims=hidden_dims).to(device)
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Define optimizer.
        for epoch in range(2000):
            if epoch%100 == 0:
                print("Training GCN, epoch {}".format(epoch))
            loss = gcn.train(model_gcn, optimizer, criterion, dataloader_train)
        models["GCN"] = model_gcn
        preds = gcn.test(model_gcn, dataloader_test, optimizer = optimizer).cpu().detach().numpy()
        cm = confusion_matrix(y_test, preds)
        fig_cm = dl.plot_cm(classes, cm, "GCN")
        fig_roc_test = ml.plot_roc_curve_(y_test, preds)
        saved_figures["confm-gcn"] = fig_cm 
        saved_figures["roc-test-gcn"] = fig_roc_test
        explainer = Explainer(
                model=model_gcn,
                algorithm=GNNExplainer(epochs=100, lr = 0.001),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='node',
                    return_type='raw'
                ),
        )
        explainers["GCN"] = explainer   

        df_result = dl.create_results_df(models, dataloader_test)
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
                                            id='roc-curve-test-gcn',
                                            figure=saved_figures["roc-test-gcn"],
                                            config = config,
                                            style={'width': '40%', 'height': '20%', 'right': '5%'}  # Adjust height as needed
                                      ),
                                      html.Br(),
                                      dbc.Button("Enlarge Figure", id = "modal-roctest-gcn", n_clicks=0),
                                  ], style = {'display': 'inline-block'}),
                                  html.Div([
                                      # First ROC Curve
                                      dcc.Graph(
                                        id='conf-matrix-gcn',
                                        figure=saved_figures["confm-gcn"],
                                        config = config,
                                        style={'height': '20%', 'width': '40%', 'left': '5%'}  # Adjust height as needed
                                      ),
                                      html.Br(),
                                      dbc.Button("Enlarge Figure", id = "modal-confm-gcn", n_clicks=0)
                                  ], style = {'display': 'inline-block'})

                              ]),

                              html.Div([
                                  html.Div([
                                      dcc.Graph(
                                            id='roc-curve-test-gae',
                                            figure=saved_figures["roc-test-gae"],
                                            config = config,
                                            style={'width': '40%', 'height': '20%', 'right': '5%'}  # Adjust height as needed
                                      ),
                                      html.Br(),
                                      dbc.Button("Enlarge Figure", id = "modal-roctest-gae", n_clicks=0),
                                  ], style = {'display': 'inline-block'}),
                                  html.Div([
                                      # First ROC Curve
                                      dcc.Graph(
                                        id='conf-matrix-gae',
                                        figure=saved_figures["confm-gae"],
                                        config = config,
                                        style={'height': '20%', 'width': '40%', 'left': '5%'}  # Adjust height as needed
                                      ),
                                      html.Br(),
                                      dbc.Button("Enlarge Figure", id = "modal-confm-gae", n_clicks=0)
                                  ], style = {'display': 'inline-block'})

                              ]),

                              html.Div([
                                  html.Div([
                                      dcc.Graph(
                                            id='roc-curve-test',
                                            figure=saved_figures["roc-test"],
                                            config = config,
                                            style={'width': '40%', 'height': '20%', 'right': '5%'}  # Adjust height as needed
                                      ),
                                      html.Br(),
                                      dbc.Button("Enlarge Figure", id = "modal-roctest", n_clicks=0),
                                  ], style = {'display': 'inline-block'}),
                                  html.Div([
                                      # First ROC Curve
                                      dcc.Graph(
                                        id='conf-matrix',
                                        figure=saved_figures["confm"],
                                        config = config,
                                        style={'height': '20%', 'width': '40%', 'left': '5%'}  # Adjust height as needed
                                      ),
                                      html.Br(),
                                      dbc.Button("Enlarge Figure", id = "modal-confm", n_clicks=0)
                                  ], style = {'display': 'inline-block'})

                              ]),
   
                              footer
            ], style={'top': '18%', 'width': '100%'})
    
    elif method_g == "GAAN (ISNs)": 

        print("Creating ISNs from gene expression arrays")
        
        if isns is None:
            isns, interaction_df = utils.create_sparse_isns(expr)
            print(f"Interaction dataframe: {interaction_df}")
            isns = torch.stack(isns).to(device)
            isns = isns.T
            print(isns.shape)
        
        edge_list = [row["feature_1"]+"_"+row["feature_2"] for index, row in interaction_df.iterrows()]
        
        # Parse the edges and create node mapping
        source_nodes, target_nodes, node_mapping = utils.parse_edges(edge_list)

        # Create edge_index tensor
        edge_index = utils.create_edge_index(source_nodes, target_nodes)
    
        x = torch.tensor(isns).to(device)
        y = torch.tensor(targets.values).to(device).flatten()
        edge_index = edge_index.to(device)
        print(x.shape, y.shape, edge_index.shape)
        uqs, counts = torch.unique(y, return_counts = True)
        dict_counts = {}
        for uq, count in zip(uqs, counts):
            dict_counts[uq.item()] = count.item()
        contamination = (dict_counts[1]/(dict_counts[0] + dict_counts[1]))*0.5

        if gaan_params is not None:
            if gaan_params.contamination is None:
                gaan_params.contamination = contamination
            gaan_params.isn = True 
        else:
            gpu = 0 if torch.cuda.is_available() else -1
            gaan_params = gaan_config.GAAN_config(noise_dim=64, hid_dim=128, num_layers=2, dropout=0.3, contamination=contamination, lr = 0.00005, epoch = 200, gpu = gpu, batch_size=1, verbose = 1, isn = True, th = 0.93)

        mydataloader = dl.create_torch_geo_data(x, y, edge_index)
        
        print("Train - Test split")
        dataloader_train, dataloader_test = dl.train_test_split_and_mask(mydataloader, train_size = 0.7, isn = True)
        in_dim = dataloader_train.x.shape[1]
        
        print("Create GAAN model")
        model = dl.create_model(in_dim, gaan_params, isn = True)
        model = dl.train_gaan(model, dataloader_train)
        models["GAAN_isn"] = model
        if "Temp" in models.keys():
            del models["Temp"]
        df_result = dl.create_results_df({"GAAN_isn": model}, dataloader_test)
        
        preds = model.predict(dataloader_test).cpu()
        explainer = None
        explainers["GAAN_isn"] = explainer   
        y_test = dataloader_test.y.cpu()
        classes = {0: "Normal", 1: "Anomalous"}
        cm = confusion_matrix(y_test, preds)
        fig_cm = dl.plot_cm(classes, cm, "GAAN_isn")
        fig_roc_test = ml.plot_roc_curve_(y_test, preds)
        saved_figures["confm"] = fig_cm 
        saved_figures["roc-test"] = fig_roc_test
        
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
                                      dbc.Button("Enlarge Figure", id = "modal-roctest", n_clicks=0),
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
                                      dbc.Button("Enlarge Figure", id = "modal-confm", n_clicks=0)
                                  ], style = {'display': 'inline-block'})

                              ]),
   
                              footer
            ], style={'top': '18%', 'width': '100%'}
        ) 
    
    else:
        return dash.no_update 

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
    global edge_index
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
    global explainers
    global isns 
    global node_mapping
    global genes 
    global gaan_params
    global ml_params 
    global genes_isn
    global interaction_df
    global contamination
    global current_seed
    global is_deterministic
    global grid_search

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
                            dbc.Button("Download Preprocessed File", id="download-button", style=download_button_style, n_clicks=0),
                            dcc.Download(id="download-csv"),
                            footer
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
            interaction_df = None
            gene_data = None
            expr = None 
            edge_index = None
            gaan_params = None
            ml_params = None 
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
            explainers = {} 
            node_mapping = None 
            isns = None 
            contamination = None
            analysis_options = {
                'ML': ['KNN', 'LR', 'DT', 'SVM', 'RF'], #['LDA', 'NB', 'KNN', 'LR', 'DT', 'SVM', 'RF']
                'GAAN (node)': ['GAAN_node', 'GCN', 'GAE'],
                'GAAN (ISNs)': ['GAAN_isn']
            }
            genes_isn = None 
            drop_ratio = 0.00
            grid_search = True
            is_deterministic = True

            
            download_button_style = {'position': 'absolute', 'top': '30%', 'right': '5%', 'width': '10%',
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
    Input("drop-ratio", "value"),
    Input("upload-data", "filename"),
    prevent_initial_call=True  # Ensure callback doesn't run on load
)
def update_upload_onclick(upload_nclicks, gene_file, preprocessed, d_ratio, filename):

    global limited_expr
    global gene_fileg 
    global download_button_style
    global drop_ratio
    global dataset_path

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
        
        drop_ratio = d_ratio

        download_button_style['display'] = 'block'
        if gene_fileg is not None:
            if preprocessed == 'yes':
                return upload_preprocessed_file(gene_fileg, filename), {'display': 'none'}, True, download_button_style
            else:    
                return upload_file(gene_fileg, filename), {'display': 'none'}, True, download_button_style

        if gene_file is not None: 
            gene_fileg = gene_file 
            if preprocessed == 'yes':
                return upload_preprocessed_file(gene_file, filename), {'display': 'none'}, True, download_button_style
            else:
                return upload_file(gene_file, filename), {'display': 'none'}, True, download_button_style


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
                    dbc.Button("Download Preprocessed File", id="download-button", style=download_button_style, n_clicks=0),
                    dcc.Download(id="download-csv"),
                    footer
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

    if is_open:
        fig.layout.height = 700 
        fig.layout.width = 1200

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

    if is_open:
        fig.layout.height = 700 
        fig.layout.width = 1200

    return fig, is_open


# Callback to update page content based on upload file button clicks
@app.callback(
    Output("modal-hyper", "is_open"),
    Output("output-container", "children", allow_duplicate = True),
    Input('modal-params-button', 'n_clicks'),
    Input("close-modal-params", "n_clicks"),
    prevent_initial_call=True  # Ensure callback doesn't run on load
)
def toogle_modal_params(open_n, close_n):

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
   
    is_open = False 

    if button_id == 'close-modal-params':
        return False, ""
    elif button_id == 'modal-params-button':
        is_open = True
    else:
        is_open = False

    return is_open, dash.no_update

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

    if is_open:
        fig.layout.height = 700 
        fig.layout.width = 1200

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

    if is_open:
        fig.layout.height = 700 
        fig.layout.width = 1200

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

    if is_open:
        fig.layout.height = 700 
        fig.layout.width = 1200

    return fig, is_open

# Callback to update page content based on analyze button clicks
@app.callback(
    Output('model-dropdown', 'options'),
    Input("analysis-type", "value")
)
def update_dropdown_models(method):
    
    global models 
    
    # Debugging print statements
    print(f"Changing Model Dropdown options")
    
    if "Temp" not in models.keys():
        if method is not None and method in analysis_options.keys():
            options =  [{'label': i, 'value': i} for i in analysis_options[method]]
            return options
        else:
            return [{'label': None, 'value': None}]
    else:
        return [{'label': None, 'value': None}]


# Callback to update page content based on analyze button clicks
@app.callback(
    Output('modal-body-content', 'children'),
    Output('submit-params-button-gaan', 'hidden'),
    Output('submit-params-button-ml', 'hidden'),
    Input('analysis-method', 'value')
)
def update_visibility_modal(method):
    
    # Debugging print statements
    print(f"Changing modal body and button visibility")

    if "GAAN" in method:
        return gaan_params_layout, False, True 
    else:
        return ml_params_layout, True, False 
  

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
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center', 'fontWeight': 'bold'}),
                footer
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
    global targets 

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
            #print(targets)
            df = pd.concat([expr, targets], axis=1)
            return dcc.send_data_frame(df.to_csv, "preprocessed_data.csv")
        else:
            print("Can't download preprocessed dataset.")

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
                style = {'color': 'red', 'width': '100%', 'textAlign': 'center', 'fontWeight': 'bold'}),
                footer
            ])
            return div 
    return dash.no_update

# Callback to update page content based on analyze button clicks
@app.callback(
    Output("explain-output", "children", allow_duplicate=True),
    Input("explain-button", "n_clicks"),
    Input("model-dropdown", "value"),
    prevent_initial_call = True
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
    
    trained = False
    if button_id == 'explain-button':
        if expr is not None:
            if model_name in analysis_options['ML']:
                if model_name in models.keys():
                    div = update_explainability_ml(model_name)
                    trained = True
            elif model_name in analysis_options['GAAN (node)']:
                if model_name in models.keys():
                    div = update_explainability_dl(model_name)
                    trained = True
            if "GAAN (ISNs)" in analysis_options.keys():
                if model_name in analysis_options['GAAN (ISNs)']:
                    if model_name in models.keys():
                        div = update_explainability_dl(model_name)
                        trained = True
            if not trained and model_name is not None:
                div = html.Div([
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.P(f"Train the model '{model_name}' and try again.", 
                           style = {'color': 'red', 'width': '100%', 'textAlign': 'center', 'fontWeight': 'bold'}), 
                    footer 
                ])
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
                       style = {'color': 'red', 'width': '100%', 'textAlign': 'center', 'fontWeight': 'bold'}), 
                footer 
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

@app.callback(Output('network', 'layout'),
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

@app.callback(Output('cytoscape-tapEdgeData-json', 'children'),
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
    global explainers 
    global expr 
    global saved_figures
    global model_name

    if "isn" not in model_name:
        if node_data:
            node = node_data["id"]
            id = node_mapping[node]
            genes = expr.columns
            explainer = explainers[model_name]
            fig = dl.plotly_featureimportance_from_gnnexplainer(explainer, mydataloader, id, genes)
            saved_figures['expnode'] = fig
            return fig

@app.callback(
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

@app.callback(Output('cytoscape-tapNodeData-json', 'children'),
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

# Callback to handle the form submission and display selected values
@app.callback(
    Output('output-container', 'children', allow_duplicate=True),
    [Input('submit-params-button-gaan', 'n_clicks'),
    Input('noise_dim', 'value'),
    Input('hid_dim', 'value'),
    Input('num_layers', 'value'),
    Input('dropout', 'value'),
    Input('contamination', 'value'),
    Input('lr', 'value'),
    Input('epoch', 'value'),
    Input('gpu', 'value'),
    Input('batch_size', 'value'),
    Input('verbose', 'value'),
    Input('th', 'value'),
    Input('deterministic-dl', 'value'),
    Input('seed-dl', 'value')
    ],
    prevent_initial_call=True
)
def update_gaan_params(n_clicks, noise_dim, hid_dim, num_layers, dropout, contamination, lr, epoch, gpu, batch_size, verbose, th, deterministic, seed):
    
    global gaan_params 

    print("Saving GAAN params")

    if n_clicks > 0:
        
        gaan_params = gaan_config.GAAN_config(noise_dim=noise_dim, hid_dim=hid_dim, num_layers=num_layers, dropout=dropout, contamination=contamination, lr = lr, epoch = epoch, gpu = gpu, batch_size=batch_size, verbose = verbose, th = th)
        if deterministic:
            set_deterministic(seed)

        return html.Div([
            html.H4("Submitted Parameters:"),
            html.P(f"Noise Dimension: {noise_dim}"),
            html.P(f"Hidden Dimension: {hid_dim}"),
            html.P(f"Number of Layers: {num_layers}"),
            html.P(f"Dropout Rate: {dropout}"),
            html.P(f"Contamination: {contamination}"),
            html.P(f"Learning Rate: {lr}"),
            html.P(f"Number of Epochs: {epoch}"),
            html.P(f"GPU Index: {gpu}"),
            html.P(f"Batch Size: {batch_size}"),
            html.P(f"Verbosity Mode: {verbose}"),
            html.P(f"Threshold: {th}"),
            html.P(f"Deterministic: {deterministic}"),
            html.P(f"Seed: {seed}")
        ])
    
    return dash.no_update


@app.callback(
    Output("output-container", "children", allow_duplicate=True),  # A div to display the saved data for confirmation
    [Input("submit-params-button-ml", "n_clicks"),
     Input("lr-solver", "value"),
     Input("lr-penalty", "value"),
     Input("lr-max-iter", "value"),
     Input("svm-kernel", "value"),
     Input("svm-c", "value"),
     Input("knn-k", "value"),
     Input("knn-metric", "value"),
     Input("rf-n-estimators", "value"),
     Input("rf-max-depth", "value"),
     Input("dt-max-depth", "value"),
     Input("dt-min-samples-split", "value"),
     Input("split-train", "value"),
     Input("cv-split", "value"),
     Input("deterministic-ml", "value"),
     Input("seed-ml", "value"),
     Input("grid-search", "value")
    ],
    prevent_initial_call = True  
)
def update_params_ml(n_clicks, lr_solver, lr_penalty, lr_max_iter, svm_kernel, svm_c, knn_k, knn_metric, rf_n_estimators, rf_max_depth, dt_max_depth, dt_min_samples_split, train_split, cv_n_splits, deterministic, seed, gsearch):

    global ml_params
    global grid_search

    print("update ML params")

    if n_clicks > 0:

        grid_search = gsearch
        ml_params = ml_config.ML_config(lr_solver, lr_penalty, lr_max_iter, svm_kernel, svm_c, knn_k, knn_metric, rf_n_estimators, rf_max_depth, dt_max_depth, dt_min_samples_split, train_split, cv_n_splits)
        
        if deterministic:
            set_deterministic(seed)

        # Return confirmation message (or save output) 
        return html.Div([
            html.H4("Submitted Parameters:"),
            html.P(f"Logistic Regression solver: {lr_solver}"),
            html.P(f"Logistic Regression max_iter: {lr_max_iter}"),
            html.P(f"SVM kernel: {svm_kernel}"),
            html.P(f"SVM C: {svm_c}"),
            html.P(f"KNN number of neighbours 'k': {knn_k}"),
            html.P(f"KNN distance metric: {knn_metric}"),
            html.P(f"Random Forest number of estimators: {rf_n_estimators}"),
            html.P(f"Random Forest trees max_depth: {rf_max_depth}"),
            html.P(f"Decision tree max_depth: {dt_max_depth}"),
            html.P(f"Decision tree min_samples_split: {dt_min_samples_split}"),
            html.P(f"Train split: {train_split}"),
            html.P(f"Cross validation n_splits: {cv_n_splits}"),
            html.P(f"Deterministic: {deterministic}"),
            html.P(f"Seed: {seed}"),    
            html.P(f"Grid Search: {gsearch}")
        ])

    return dash.no_update

@app.callback(
    Output("save-models-output", "children", allow_duplicate = True),
    Input("save-models-button", "n_clicks"),
    prevent_initial_call=True  # Only trigger after button is clicked
)
def save_models(n_clicks):
    if n_clicks > 0:
        #save_paths = []  # List to store paths of saved models for confirmation messages

        sep = os.sep 

        for model_name, model_obj in models.items():
            # Set file paths for each model
            if "GAAN" in model_name:
                # Save PyTorch model state dictionary
                file_path = f"saved_models{sep}{model_name}_model.pth"
                torch.save(model_obj.state_dict(), file_path)
            else:
                # Save non-PyTorch model (e.g., scikit-learn) with joblib
                file_path = f"saved_models{sep}{model_name}_model.joblib"
                joblib.dump(model_obj, file_path)

            # Append the saved file path to the list
            #save_paths.append(file_path)

        # Return a confirmation message with saved file paths
        return f"Models saved successfully"
    
    return ""

@app.callback(
    Output('explain-output', 'children', allow_duplicate=True),
    Input('graph-selection-dropdown', 'value'),
    Input("model-dropdown", "value"),
    prevent_initial_call = True
)
def update_graph_div(selected_graph_index, model_name):
    captum_div = create_div_captum(model_name)
    return create_div_exp_isn(selected_graph_index, captum_div, model_name)

if __name__ == '__main__':
        
    print("Running E-ABIN")
    webbrowser.open(html_plot_path)    
    app.run(debug=False)