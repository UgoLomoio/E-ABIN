from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score, f1_score, recall_score, precision_score
import torch 
import numpy as np 
#import matplotlib.pyplot as plt 
import pandas as pd 
import gzip
import shutil
import os
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
#import seaborn as sns
import itertools
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import plotly 
from plotly import graph_objects as go
import GEOparse
from torch.nn.functional import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_geoaccession_code(lines):
    for i, line in enumerate(lines):  # Enumerate to count the lines
        if i >= 28:  # Stop after reading 28 lines
            break
        if "!Series_geo_accession" in line:
            geo_accession = line.split("\t")[1].strip().replace('"', "")
    return geo_accession

def get_platforms(geo_accession):
    gse = GEOparse.get_GEO(geo=geo_accession, destdir="./")
    # Get the platform IDs associated with this series
    platforms = gse.gpls.keys()
    return gse, platforms

def get_annotation_df(gse, values_df, platforms):
    
    found_platform = None
    annotation_df = None

    for platform in platforms:

      print(platform)
      # Get the platform object
      gpl = gse.gpls[platform]

      # Convert the platform object into a DataFrame
      platform_df = gpl.table
      columns_values = values_df.columns.to_list()
      columns = platform_df["ID"].to_list()
  
      if columns_values == columns:
        found_platform = platform
        annotation_df = platform_df
        break 
    
    return found_platform, annotation_df

def get_edges_by_sim(expr):

    exp_values = expr.values
    map_edgeattr = {}
    edges_1 = []
    edge_list = []
    edge_weights = []
    th = 0.93
    for i, elem1 in enumerate(exp_values):
        if not torch.is_tensor(elem1):
            elem1 = torch.tensor(elem1)
        for j, elem2 in enumerate(exp_values):
            if not torch.is_tensor(elem2):
                elem2 = torch.tensor(elem2)
            sim = cosine_similarity(elem1, elem2, dim = 0).item()
            #sim = corr[i, j]
            if sim > th:
                if f"{expr.T.columns[i]}_{expr.T.columns[j]}" not in edges_1:
                    edges_1.append(f"{expr.T.columns[i]}_{expr.T.columns[j]}")
                    edge_list.append((expr.T.columns[i], expr.T.columns[j]))
                    edge_weights.append(sim)
                    map_edgeattr[f"{expr.T.columns[i]}_{expr.T.columns[j]}"] = sim
    return edges_1, edge_list, edge_weights, map_edgeattr


def unzip_data(filepath):
    
    unzipped = False
    file_extension = filepath[-4:]
    if file_extension == ".csv" or file_extension == ".txt" or file_extension == ".bgx":
        unzipped = True
        return filepath 
    elif file_extension == ".zip" or file_extension[-3:] == ".gz": #.zip not tested
        if file_extension[-3:] == ".gz":
            file_extension = file_extension[-3:]
            file = file_extension[:-3]
        else:
            file = file_extension[:-4]
        unzipped = False 
        #unzip data 
        print("unzipping file ", filepath)
        with gzip.open(filepath, 'rb') as f_in:
            with open(file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return file
    
    else:
        raise Exception("Unknown {} file extension in input.".format(file_extension))

def get_targets(dataframe):
    
    map_ys = {}
    ys = []
    patients = []
    
    for index, row in enumerate(dataframe.iterrows()):

        patient = row[1]["ID_REF"]
        label = row[0]
        if "healthy" in label.lower():
            y = "control"
        elif "control" in label.lower():
            y = "control"
        elif "normal" in label.lower():
            y = "control"
        else: 
            y = "cancer"
        ys.append(y)
        patients.append(patient)
        map_ys[patient] = y
        
    ys = [0 if y == "control" else 1 for y in ys]
    clinic = pd.DataFrame(data = np.array(ys), index = patients, columns = ["Target"])
    clinic.index.name = "sample"
    return clinic

def get_expressions(dataframe):
    
    values_id = 36
    values_columns = dataframe.columns[values_id:-1]
    values = np.array(dataframe[values_columns].values)
    
    clinic = get_targets(dataframe)
    patients = list(clinic.index)
    
    patient_values = {}
    for i, patient in enumerate(patients):
        patient_values[patient] = values[i, :]
    
    values_df = pd.DataFrame(values, index = patients, columns = values_columns)
    return values_df

def read_human_metilation(filepath):
    df = pd.read_csv(filepath, index_col = 0, skiprows=7)
    return df

def read_gene_expression(filepath):
    df = pd.read_csv(filepath, index_col = 0, skiprows=28, sep = "\t").T
    return df 

def dense_isn(
    data,
    device = "cpu"
):
    """
    Network computation based on the Lioness algorithm
    """
    num_samples = torch.tensor(data.shape[0], dtype=torch.float32)
    orig = torch.tensor(data.values).to(device)
    orig_transpose = torch.transpose(orig, 0, 1)
    dot_prod = torch.matmul(orig_transpose, orig)
    mean_vect = torch.sum(orig, dim=0)
    std_vect = torch.sum(torch.pow(orig, 2), dim=0)
    glob_net = num_samples * torch.corrcoef(orig_transpose)

    @torch.jit.script
    def edge(num, mean_v, std_v, dot, glob, row):
        mean = mean_v - row
        d_q = torch.sqrt((num - 1) * (std_v - torch.pow(row, 2)) - torch.pow(mean, 2))
        nom = (num - 1) * (dot - (torch.reshape(row, (row.shape[0], 1)) * row)) - (
            torch.reshape(mean, (row.shape[0], 1)) * mean
        )
        return torch.flatten(
            glob - ((num - 1) * (nom / (torch.reshape(d_q, (d_q.shape[0], 1)) * d_q)))
        )

    for i in range(data.shape[0]):
        yield edge(num_samples, mean_vect, std_vect, dot_prod, glob_net, orig[i])
        

def preprocess(data, label_column: str):
    """Seperate data into cases and controls and add necessary columns."""
    data.index.name = None

    controls = (
        data[data[label_column] == 0].drop([label_column], axis=1).T.reset_index()
    )
    cases = data[data[label_column] == 1].drop([label_column], axis=1).T.reset_index()
    controls[["N1", "N2"]] = controls["index"].str.split("_", expand=True)
    controls = controls.drop(["index"], axis=1)

    cases[["N1", "N2"]] = cases["index"].str.split("_", expand=True)
    cases = cases.drop(["index"], axis=1)

    return controls, cases


def graphs(data):
    """Compute graphs from ISNs."""
    data_list = {}
    for i in range(data.shape[1] - 2):
        temp = data.iloc[:, [data.shape[1] - 2, data.shape[1] - 1, i]].copy()
        key = temp.columns[2]
        temp.columns = ["E1", "E2", "Weight"]
        data_list[key] = torch.to_numpy_array(
            torch.from_pandas_edgelist(temp, "E1", "E2", "Weight"), weight="Weight"
        )

    return data_list


def calculate_filtration_curve(data_list, thr_values):
    """Apply the stat (number of graph edges) to the computed ISNs."""
    curve = np.zeros((len(data_list), len(thr_values)))
    for a, (individual, Adj) in enumerate(data_list.items()):
        for i, thr_value in enumerate(thr_values):
            curve[a, i] = torch.from_numpy_array(
                np.where((Adj > thr_value) & (Adj != 0), 1, 0), create_using=Graph
            ).number_of_edges()

    return curve


def plot_filtration_curve(df, label_column="label", output=None):
    """Plot labeled data."""
    thr_values = np.arange(-4, 4, 0.03)

    controls_data, cases_data = preprocess(df, label_column)
    controls_FC = calculate_filtration_curve(graphs(controls_data), thr_values)
    cases_FC = calculate_filtration_curve(graphs(cases_data), thr_values)

    plt.figure(figsize=(12, 10))
    labels = {"Cases": "red", "Controls": "blue"}
    plt.errorbar(
        thr_values,
        np.mean(controls_FC, axis=0),
        yerr=np.std(controls_FC, axis=0),
        elinewidth=0.5,
    )
    plt.errorbar(
        thr_values,
        np.mean(cases_FC, axis=0),
        yerr=np.std(cases_FC, axis=0),
        elinewidth=0.5,
        color="red",
    )
    plt.xlabel("Threshold Values", fontsize=20)
    plt.ylabel("Graph Statistic: N Edges", fontsize=20)
    handles = [
        plt.Line2D([], [], color=labels[label], marker="o", linestyle="-")
        for label in labels
    ]
    plt.legend(handles, labels.keys(), loc="upper right", fontsize=15)

    #if output is None:
        #plt.show()
    #else:
    #    plt.savefig(output)


def find_filter_edges(expr, sig):
    """Find the indexes of the columns we want to keep."""
    columns = np.asarray(
        [
            f"{a}_{b}"
            for (a, b) in zip(
                np.repeat(expr.columns, expr.shape[1]),
                np.tile(expr.columns, expr.shape[1]),
            )
        ]
    )
    sorter = np.argsort(columns)
    return sorter[np.searchsorted(columns, sig, sorter=sorter)]

def interactions(edges):

  df = pd.DataFrame([], columns = ["gene1", "gene2"])
  for i, edge in enumerate(edges):
    gene1, gene2 = edge.split("_")
    df.loc[i] = [gene1, gene2]
  return df

def unmapped_info(df):
    rows = df.shape[1]
    location = [2 * i for i in range(rows)]
    chromosome = sorted([(i % 23) + 1 for i in range(rows)])
    return pd.DataFrame(
        {"chr": chromosome[:rows], "location": location}, index=df.columns
    )

def ensure_tensor(data, dtype):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=dtype)
    elif data.dtype != dtype:
        data = data.to(dtype)
    return data

def edges_by_genecorr(expr, corr, corr_th = 0.6):
    edges = []
    for i in range(corr.shape[0]):
      for j in range(i + 1, corr.shape[1]):
        if corr[i][j] > 0.6:
          if f"{expr.columns[i]}_{expr.columns[j]}" not in edges:
            edges.append(f"{expr.columns[i]}_{expr.columns[j]}")
    return edges

def compute_weighted_edges(expr: pd.DataFrame, metric='euclidean'):
    """
    Compute weighted edges between patients based on their gene expression distance.

    Parameters:
    - expr (pd.DataFrame): Gene expression data with patients as rows and genes as columns.
    - metric (str): The distance metric to use (default is 'euclidean').

    Returns:
    - edges (pd.DataFrame): DataFrame with columns 'patient1', 'patient2', and 'distance'.
    """
    # Ensure the input is a DataFrame
    if not isinstance(expr, pd.DataFrame):
        raise ValueError("Input expr must be a pandas DataFrame")

    # Compute the pairwise distances
    distance_matrix = pdist(expr.values, metric=metric)
    distance_matrix = squareform(distance_matrix)

    # Create a DataFrame to hold the edges
    num_patients = expr.shape[0]
    edges = []

    for i in range(num_patients):
        for j in range(i + 1, num_patients):
            edges.append({
                'patient1': expr.index[i],
                'patient2': expr.index[j],
                'distance': distance_matrix[i, j]
            })

    edges_df = pd.DataFrame(edges)
    
    return edges_df

def compute_edges_with_threshold(expr: pd.DataFrame, metric='euclidean', threshold=1.0):
    """
    Compute unweighted edges between patients based on their gene expression data,
    with a distance threshold to decide if an edge should exist.

    Parameters:
    - expr (pd.DataFrame): Gene expression data with patients as rows and genes as columns.
    - metric (str): The distance metric to use (default is 'euclidean').
    - threshold (float): The distance threshold below which an edge is created.

    Returns:
    - edges (pd.DataFrame): DataFrame with columns 'patient1', 'patient2'.
    """
    # Ensure the input is a DataFrame
    if not isinstance(expr, pd.DataFrame):
        raise ValueError("Input expr must be a pandas DataFrame")

    edges_df = compute_weighted_edges(expr)
    edges = []
    for patient1, patient2, distance in edges_df.iterrows():
        if distance <= threshold:
            edges.append({
                'patient1': patient1,
                'patient2': patient2,
            })
    edges_df = pd.DataFrame(edges)
    return edges_df

def parse_edges(edge_list):
    source_nodes = []
    target_nodes = []
    node_mapping = {}
    next_index = 0

    for edge in edge_list:
        source, target = edge.split('_')

        if source not in node_mapping:
            node_mapping[source] = next_index
            next_index += 1
        if target not in node_mapping:
            node_mapping[target] = next_index
            next_index += 1

        source_nodes.append(node_mapping[source])
        target_nodes.append(node_mapping[target])

    return source_nodes, target_nodes, node_mapping

def create_edge_index(source_nodes, target_nodes):
    edge_list = np.array([source_nodes, target_nodes], dtype=np.int64)
    edge_index = torch.LongTensor(edge_list)
    return edge_index

def compute_pvalues(filepath, top_k = 50):
    data = pd.read_csv(filepath, index_col=0)
    columns = data.columns[:-1]

    p_values = []
    columns = []
    correlation = []

    for i, gene in enumerate(data.columns[:-1]):
        if i % 2000 == 0:
          print("\r", i+1, "/", len(data.columns[:-1]), end = "")

        x = data.loc[:, gene].values
        y = data.loc[:, "Target"].values
        model = sm.OLS(x, sm.add_constant(y)).fit()
        if np.any(np.isnan(x)):  # Verifica se ci sono NaN nei dati
            continue

        p_values.append(model.pvalues[1])  # p-value for the group variable
        correlation.append(model.params[0])
        columns.append(gene)
        del x
        del y
        del gene
        del i
        del model
        
    adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
    results_df = pd.DataFrame({
        'gene': columns,
        'coef': correlation,
        'p_value': p_values,
        'adj_p_value': adjusted_p_values
    }).set_index('gene')
    top_genes = results_df.loc[results_df['adj_p_value'] < 0.05].head(top_k)
    return top_genes


def compute_edge_correlation(data, top_nodes, top_k = 100):

    tensors = torch.stack([torch.tensor(data[col].values, dtype=torch.float32) for col in data.columns])
    tensors = tensors.cpu()
    # Compute the correlation matrix
    corr_matrix = torch.corrcoef(tensors)
    
    gene_index = {gene: i for i, gene in enumerate(data.columns[:-1])}  # Create a gene index mapping

    # Convert string indices to tensor
    top_node_indices = torch.tensor([gene_index[gene] for gene in top_nodes.index])
    final_corr_matrix = corr_matrix[top_node_indices, :]
    final_corr_matrix = final_corr_matrix[:, top_node_indices]
    
    correlations = []
    for i, gene in enumerate(top_nodes.index):

      top_values, top_indices = torch.topk(final_corr_matrix[i, :], k = 20)
      top_indices = top_indices[top_values != 1.0]
      top_values = top_values[top_values != 1.0]

      for j, gene2 in enumerate(top_nodes.index[top_indices]):
        corr = final_corr_matrix[i, j].item()
        correlations.append((gene, gene2, abs(corr)))

      del top_values
      del top_indices
    
    top_edges = pd.DataFrame(correlations, columns=['Gene1', 'Gene2', 'Correlation'], index = None).sort_values(by = 'Correlation', ascending = False)
    top_edges = top_edges.head(top_k)
    return top_edges 

def plot_cm(classes, cm):
    """
    input:
    classes: dict of class names
    cm: Confusion matrix (2D array)
    output: A Plotly figure (confusion matrix heatmap)
    """
    classes = list(classes.keys())#or values
    cm = np.array(cm)
    fig = go.Figure()

    # Create heatmap trace
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Blues',
            colorbar=dict(title='Count'),
            zmid=cm.max() / 2  # Center the color scale
        )
    )

    # Add text annotations
    annotations = [
        go.layout.Annotation(
            text=str(cm[i, j]),
            x=j,
            y=i,
            showarrow=False,
            font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black"),
            align="center"
        )
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
    ]

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Target',
        xaxis=dict(
            tickvals=np.arange(len(classes)),
            ticktext=classes,
            title='Predicted'
        ),
        yaxis=dict(
            tickvals=np.arange(len(classes)),
            ticktext=classes,
            title='Target',
            autorange='reversed'  # Reverse y-axis to have the matrix displayed correctly
        ),
        annotations=annotations
    )

    fig.update_layout(
        autosize=False,
        width=300,
        height=300
    )

    return fig

def validate_model(y_trues, y_preds):

    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    specificity = recall_score(y_trues, y_preds, pos_label=0)
    sensitivity = recall_score(y_trues, y_preds)
    precision =  precision_score(y_trues, y_preds)
    aucs = roc_auc_score(y_trues, y_preds)
    auc = np.mean(aucs)


    cr = classification_report(y_trues, y_preds)
    cm = confusion_matrix(y_trues, y_preds)
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "cm": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc_score": auc,
        "precision": precision,
        "report": cr,
        "predictions": y_preds,
    }

    classes = {"Healthy Control": 0, "Anomalous": 1}
    fig_cm = plot_cm(classes, cm)

    return fig_cm, metrics, "Accuracy: {} \n F1 score: {} \n Sensitivity: {} \n Specificity: {} \n ROC AUC score: {} \n Confusion Matrix: \n {} \n Classification Report: \n {} \n".format(acc, f1, sensitivity, specificity, auc, cm, cr)


def compress_expr(expr, ys):
   
    labels = {0: "Healthy Control", 1: "Cancer"}

    uqs, counts = np.unique(ys, return_counts=True)

    methods = ["PCA", "t-SNE"]
    for method in methods:
        
        fig = go.Figure()
        if method == "PCA":
            pca = PCA(n_components=2)
            expr = pca.fit_transform(expr)
        else:
            if min(counts) < 10:
                perplexity = 2
            else:
                perplexity = 10
            tsne = TSNE(n_components=2, perplexity = 2)
            expr = tsne.fit_transform(expr)
      
        unique_ys = np.unique(ys)
        colors = ["purple", "orange"]
        for y in unique_ys:
            mask = ys == y
            mask = mask.flatten()
            fig.add_trace(go.Scatter(x=expr[mask, 0], y=expr[mask, 1], mode="markers", marker=dict(color=colors[y], size=8), name=labels[y]))

        fig.update_layout(title = "Gene Expression 2D {}".format(method))
    
        if method == "PCA":
          fig_pca = fig
        else:
          fig_tsne = fig 
        
        fig.update_layout(width = 700, height = 400)

    return fig_pca, fig_tsne
  