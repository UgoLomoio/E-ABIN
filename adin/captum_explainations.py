from captum.attr import Saliency, IntegratedGradients
import torch 
import numpy as np 
import random
from collections import defaultdict
import networkx as nx 
import matplotlib.pyplot as plt 
from matplotlib import cm
import io 
import base64

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = None 
model_name = None 

def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


def draw_network(g, y, edge_mask=None):
    import matplotlib

    g = g.copy().to_undirected()

    pos = nx.spring_layout(g)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]

    min_val, max_val = 0.3, 1.0
    n = 10
    orig_cmap = plt.cm.Reds
    colors = orig_cmap(np.linspace(min_val, max_val, n))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cmap", colors)

    nx.draw(g, pos=pos, width=widths, font_size = 14,
            edge_color=edge_color, node_color = y, cmap = cmap, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.show()

def model_forward(edge_mask, data):
    out, h = model.forward(data.x, data.edge_index, edge_mask)
    return out


def explain(method, data, target=0):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data,),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask, target=target,
                                  additional_forward_args=(data,))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


# Function to convert a Matplotlib figure to a base64 string
def matplotlib_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode()
    buf.close()
    return "data:image/png;base64,{}".format(base64_img)

def explain_with_captum(model_in, modelname_in, data, G, target = 1):

    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib as mpl
    global model 
    global model_name

    model = model_in
    model_name = modelname_in
    if model_name == "GCN":
        # Define the colormap
        colors = [(0, 1, 0), (1, 0, 0)]  # Green to Red
        cmap = LinearSegmentedColormap.from_list("GreenRed", colors)
        # Define the white-to-black colormap
        colors = [(1, 1, 1), (0, 0, 0)]  # White to Black
        white_black_cmap = LinearSegmentedColormap.from_list("WhiteBlack", colors)

        imgs = []

        for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:

            edge_mask = explain(method, data, target=1)

            edge_mask_dict = aggregate_edge_directions(edge_mask, data)

            max_weight = max(edge_mask_dict.values())
            normalized_weights = {edge: weight / max_weight for edge, weight in edge_mask_dict.items()}


            fig, ax= plt.subplots(figsize=(10, 8))  
            plt.title(title)
 
            pos = nx.spring_layout(G)  
           
            node_mapping = {node: i for i, node in enumerate(G.nodes())}
            node_mapping_rev = {i: node for i, node in node_mapping.items()}
    
           
            G = nx.relabel_nodes(G, node_mapping)
    
   
            pos = {node_mapping[node]: coord for node, coord in pos.items()}
    
         
            
            normalized_weights = {
                edge: weight if edge[0] != edge[1] else 0
                for edge, weight in normalized_weights.items()
            }
            edges, weights = zip(*normalized_weights.items())
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edges,
                                   width=[20 * w for w in weights],
                                   edge_color=weights,
                                   edge_cmap=white_black_cmap)

            node_sizes = [200 + 50 * G.degree(n) / max(dict(G.degree()).values()) for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos,
                                   node_size=node_sizes,
                                   node_color=data.y.cpu().detach().numpy(),
                                   cmap=cmap)

            y = data.y.cpu().detach().numpy()
            uq_y = np.unique(y)
            value_index = {v: i for i, v in enumerate(uq_y)}
            patches = [mpl.patches.Patch(color=color, label=label) for label, color in {"healthy": "green", "anomalous": "red"}.items()]
            plt.legend(handles=patches)

            nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
    
            plt.colorbar(plt.cm.ScalarMappable(cmap=white_black_cmap), label="Normalized Edge Weights", shrink=0.8, ax=ax)

            plt.tight_layout() 
            img = matplotlib_to_base64(fig)
            imgs.append(img)
        return imgs[0], imgs[1]
    else:
        fig = plt.figure(figsize=(1, 1))
        empty_img = matplotlib_to_base64(fig)
        return empty_img, empty_img