import torch
from torch import nn
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, n_genes, n_classes = 2, hidden_dims = [512, 128]):

        super().__init__()
        torch.manual_seed(1234)

        if hidden_dims[-1] != n_classes:
          hidden_dims.append(n_classes)

        n_layers = len(hidden_dims)

        convs = []
        input_dim = n_genes

        for i in range(n_layers):
          h_dim = hidden_dims[i]
          conv = GCNConv(input_dim, h_dim)
          convs.append(conv)
          input_dim = h_dim

        self.convs = nn.Sequential(*convs)

    def forward(self, x, edge_index, edge_weight = None):

        h = x.clone()
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                h = h.relu()
        out = h
        return out, h

    def __call__(self, *args, **kwargs):

        x = args[0]
        edge_index = args[1]
        #print(x.shape, edge_index.shape)
        out, _ = self.forward(x, edge_index)
        out = out.view(-1, 1)  # Ensure the target is in the right shape [N, 1]
        #print(out.shape)
        return out

def test(model, data, optimizer = None):

    model.eval()
    if optimizer is not None:
        optimizer.zero_grad()  # Clear gradients.
    out, h = model.forward(data.x, data.edge_index)  # Perform a single forward pass.
    predicted_classes = torch.argmax(out, axis=1).float() # [0.6, 0.2, 0.7, 0.1] -> 2
    predicted_classes.requires_grad = True
    return predicted_classes

def train(model, optimizer, criterion, data):

    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out, h = model.forward(data.x, data.edge_index)  # Perform a single forward pass.

    predictions = out# Shape: [14, 2]
    target_classes = data.y.view(-1) .long() # Shape: [14], integer class labels
    loss = criterion(predictions, target_classes)

    loss.backward()  # Compute gradients
    optimizer.step()  # Update model parameters

    return loss