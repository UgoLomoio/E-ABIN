from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from .utils import plot_cm, validate_model
import torch 
from torch_geometric.data import Data
from pygod.metric import eval_roc_auc, eval_f1, eval_average_precision, eval_recall_at_k, eval_precision_at_k
import gc 
import pandas as pd 
import numpy as np 
import plotly.express as px 
from .gaan import GAAN_Explainable

scores = {"ROC AUC": eval_roc_auc, "F1": eval_f1, "Average Precision": eval_average_precision, "Recall@k": eval_recall_at_k, "Precision@k": eval_precision_at_k}

epoch = 100
verbose = 1
dropout = 0.2
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
gpu = -1 if device == "cpu" else 0 #-1 cpu
num_layers = 7
learning_rate = 0.005
batch_size = 2
h_dim = 64
noise_dim = 64#16
contamination = 0.005
verbose = 1 
detectors = {}

def train_and_test_mlp(X_train, X_test, y_train, y_test, max_iter = 50):
    
    # Initialize and train the MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=max_iter, learning_rate="adaptive",  random_state=42, verbose = True)
    mlp.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = mlp.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)
    

def train_test_split_and_mask(data, map_edgeattr, node_mapping_rev, train_size = 0.2):
    """
    Input:
        data: torch_geometric.Data object
        train_size: float between 0.0 and 1.0, OPTIONAL, default 0.2
    """
    #input 
    
    # Assuming `data.y` contains the labels
    num_nodes = data.num_nodes
    if train_size < 0.1 or train_size >= 1:
       raise Exception("Train size {} must be greater (or equal) then 0.1 and lower then 1".format(train_size))

    # Create a boolean mask for training nodes
    indices = torch.arange(num_nodes).to(device)
    #train_indices, test_indices = train_test_split(indices, train_size=train_size, stratify=data.y.cpu())
    y = data.y
    uqs, counts = torch.unique(y, return_counts = True)
    start = int((2/3)*counts[0])
    train_indices = torch.arange(0, start).to(device)
    test_indices = torch.arange(start, num_nodes).to(device)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    data.edge_index = data.edge_index.to(device)

    data.train_mask = train_mask
    data.test_mask = test_mask

    train_edgeindex = filter_edge_index(data.edge_index, data.train_mask)
    test_edgeindex = filter_edge_index(data.edge_index, data.test_mask)
    train_edge_attr = filter_edge_attr(map_edgeattr, node_mapping_rev, train_edgeindex)    
    test_edge_attr = filter_edge_attr(map_edgeattr, node_mapping_rev, test_edgeindex, start)
    #print(data.edge_attr[data.train_mask])
    train_dataloader = create_torch_geo_data(data.x[data.train_mask], data.y[data.train_mask], train_edgeindex, train_edge_attr)
    test_dataloader = create_torch_geo_data(data.x[data.test_mask], data.y[data.test_mask], test_edgeindex, test_edge_attr)

    return train_dataloader, test_dataloader

def filter_edge_attr(map_edgeattr, node_mapping_rev, filtered_edgeindex, start = 0):

    filtered_edge_attr = [] 
    
    for i in range(len(filtered_edgeindex[0])):
        start = int(start)
        node1 = node_mapping_rev[filtered_edgeindex[0, i].item()+start] 
        node2 = node_mapping_rev[filtered_edgeindex[1, i].item()+start]
        
        edge_attr = map_edgeattr[f"{node1}_{node2}"]    
        filtered_edge_attr.append(edge_attr)

    filtered_edge_attr = torch.tensor(filtered_edge_attr).to(device)
    return filtered_edge_attr

def create_torch_geo_data(x, y, edge_index, edge_weights):
    
    x = ensure_tensor(x, dtype=torch.float).to(device)
    edge_index = ensure_tensor(edge_index, dtype=torch.long).to(device)
    edge_weights = ensure_tensor(edge_weights, dtype=torch.float).to(device)
    y = ensure_tensor(y, dtype=torch.long).to(device)
    return Data(x=x, edge_index = edge_index, edge_attr=edge_weights, y = y)


def train_gaan(train_data):
    
    global detectors 
    
    in_dim = train_data.x.shape[1]
    detector = GAAN_Explainable(in_dim, noise_dim=noise_dim, hid_dim=h_dim, num_layers=num_layers, batch_size = 2, dropout=dropout, device=device, backbone=None, contamination=contamination, lr=learning_rate, epoch=epoch, verbose=verbose)
    gc.collect()
    torch.cuda.empty_cache()
    print(train_data)
    detector.fit(train_data)
    detectors["GAAN"] = detector
    return detector 

def create_results_df(detector, test_data):
    columns = ["Model name"]
    [columns.append(elem) for elem in list(scores.keys())]
    df = pd.DataFrame(columns = columns)
 
    y_test = test_data.y
    pred, score, prob, conf = detector.predict(test_data,
                                                return_pred=True,
                                                return_score=True,
                                                return_prob=True,
                                                return_conf=True)

    k = 5
    temp = ["GAAN"]
    for score_name, score_fun in scores.items():
        if "@" in score_name:
          score_val = score_fun(y_test.cpu(), pred.cpu()).item()
        else:
          score_val = score_fun(y_test.cpu(), pred.cpu()).item()
        temp.append(score_val)

    df.loc[0] = temp
    return df


def compute_elements_cyto(patients, edge_list, edge_weights, y, preds):
   
    map_final = {0: "Control", 1: "Anomalous"}

    nodes = []
    for i, patient in enumerate(patients):
        target = map_final[y[i].item()]
        pred = map_final[preds[i].item()]  
        nodes.append({'data': {'id': patient, 'label': patient, 'prediction': pred, 'classes': target}, 'classes': target})

    edges = []
    for i, edge in enumerate(edge_list):
        patient1, patient2 = edge
        weight = edge_weights[i]
        if patient1 != patient2:
            edges.append({'data': {'source': patient1, 'target': patient2, 'weight': weight.item()}})

    return nodes, edges
    
def get_subgraph(node_data, nodes, edges):
    
    node = node_data["id"]
    nodes_adj = [{"data": node_data}]
    edges_adj = []
    nodes_idxs = [node["data"]["id"] for node in nodes]
    nodes_idxs = np.array(nodes_idxs)
        
    for i, edge in enumerate(edges):
        #print(i+1, "/", len(edges), end = "\r")
        node1 = edge["data"]["source"]
        node2 = edge["data"]["target"]
        if node1 == node or node2 == node:
            edges_adj.append(edge)

    for i, edge in enumerate(edges_adj):
        #print(i+1, "/", len(edges_adj), end = "\r")
        node1 = edge["data"]["source"]
        node2 = edge["data"]["target"]
        if node1 != node:
            node1_idx = np.argwhere(nodes_idxs == node1)[0][0]
            node1_data = nodes[node1_idx]
            nodes_adj.append(node1_data)
        elif node2 != node:
            node2_idx = np.argwhere(nodes_idxs == node2)[0][0]
            node2_data = nodes[node2_idx]
            nodes_adj.append(node2_data)

    elements = nodes_adj + edges_adj 
    return elements

def plotly_featureimportance_from_gnnexplainer(explainer, data, node_id, genes, top_k = 10):

    explanation = explainer(data.x, data.edge_index, index = node_id)
    
    node_mask = explanation.get('node_mask')
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available "
                                 f"in '{explanation.__class__.__name__}' "
                                 f"(got {explanation.available_explanations})")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
                raise ValueError(f"Cannot compute feature importance for "
                                 f"object-level 'node_mask' "
                                 f"(got shape {node_mask.size()})")
    
    importances = node_mask.sum(dim=0).cpu().detach().numpy()

    # Create a DataFrame for gene importances
    feature_importances = pd.DataFrame({
        'Gene': genes,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Select the top 10 most important genes
    top_genes = feature_importances.head(top_k)

    # Create an interactive bar plot using Plotly Express
    fig = px.bar(top_genes, x='Importance', y='Gene', 
                 orientation='h',  # Horizontal bar plot
                 title='Top 10 Most Important Genes',
                 labels={'Importance': 'Importance Score', 'Gene': 'Gene'},
                 color='Importance',
                 color_continuous_scale='viridis')

    # Update layout for better presentation
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Gene",
        width=800,
        height=400,
        template='plotly_white'
    )

    return fig

# Filter the edge_index and ensure all edges reference valid nodes in the mask
def filter_edge_index(edge_index, mask):
    # Get the indices of valid nodes according to the mask
    idx_map = torch.nonzero(mask, as_tuple=True)[0].to(device)
    
    # Create a mapping from old indices to new indices
    node_idx_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(idx_map)}

    # Filter edge_index to include only valid edges
    mask_edge_index = (mask[edge_index[0]] & mask[edge_index[1]]).nonzero(as_tuple=False).to(device).squeeze()
    
    if mask_edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long).to(device)  # No valid edges

    # Filter and remap edge_index
    filtered_edge_index = edge_index[:, mask_edge_index]
    filtered_edge_index = torch.tensor([[node_idx_map.get(int(i), -1) for i in edge] for edge in filtered_edge_index.t()], dtype=torch.long).to(device).t()

    # Remove edges with -1 index due to remapping issues
    valid_edges = filtered_edge_index.min(dim=0).values >= 0
    filtered_edge_index = filtered_edge_index[:, valid_edges]

    return filtered_edge_index


def ensure_tensor(data, dtype):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=dtype)
    elif data.dtype != dtype:
        data = data.to(dtype)
    return data

