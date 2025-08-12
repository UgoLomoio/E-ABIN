import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DeepGONet_pl(pl.LightningModule):
    def __init__(
        self, 
        n_input,
        n_classes,
        n_hidden,
        connection_matrix=None,
        keep_prob=0.4,
        use_bn=False,
        lr_method='adam',
        lr=0.001,
        type_training='LGO',
        alpha=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        
        super().__init__()
        self.save_hyperparameters()
        self.connection_matrix = connection_matrix
        self.type_training = type_training
        self.layers = nn.ModuleList()
        self.use_bn = use_bn
        self.bns = nn.ModuleList() if use_bn else None
        self.n_classes = n_classes
        sizes = [n_input] + n_hidden
        for i in range(len(n_hidden)):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1])).to(device)
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(sizes[i+1])).to(device)
        self.layers = self.layers.to(device)
        self.bns = self.bns.to(device) if self.use_bn else None
        self.out = nn.Linear(n_hidden[-1], n_classes).to(device)
        self.keep_prob = keep_prob
        self.lr_method = lr_method
        self.alpha = alpha
        self.lr = lr
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_bn:
                x = self.bns[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=1-self.keep_prob, training=self.training)
        return self.out(x)

    def _l1_loss(self, x):
        return torch.sum(torch.abs(x))

    def _l2_loss(self, x):
        return torch.sum(x ** 2)

    def compute_regularization(self):
        additional_loss = torch.tensor(0., device=next(self.parameters()).device)
        type_training = self.type_training.upper()
        if type_training == 'LGO' and self.connection_matrix is not None:
            for idx, layer in enumerate(self.layers):
                cm = self.connection_matrix[idx]
                additional_loss += self._l2_loss(layer.weight * (1 - cm))
        elif type_training == 'L2':
            for layer in self.layers:
                additional_loss += self._l2_loss(layer.weight)
        elif type_training == 'L1':
            for layer in self.layers:
                additional_loss += self._l1_loss(layer.weight)
        return additional_loss


    def training_step(self, batch, batch_idx):

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self(x)

        if self.n_classes >= 2:
            loss_fn = nn.CrossEntropyLoss()
            if y.dim() > 1 and y.size(1) == self.n_classes:
                y_labels = torch.argmax(y, dim=1)
            else:
                y_labels = y
            ce_loss = loss_fn(logits, y_labels)
            acc = (torch.argmax(logits, dim=1) == y_labels).float().mean()
        else:
            logits = logits.squeeze(1)
            loss_fn = nn.BCEWithLogitsLoss()
            ce_loss = loss_fn(logits, y.float())
            preds = (torch.sigmoid(logits) > 0.5).long()
            acc = (preds == y.long()).float().mean()

        if self.type_training in ['LGO', 'L1', 'L2']:
            additional_loss = self.compute_regularization()
            loss = ce_loss + self.alpha * additional_loss
        else:
            loss = ce_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self(x)

        if self.n_classes >= 2:
            loss_fn = nn.CrossEntropyLoss()
            if y.dim() > 1 and y.size(1) == self.n_classes:
                y_labels = torch.argmax(y, dim=1)
            else:
                y_labels = y
            loss = loss_fn(logits, y_labels)
            acc = (torch.argmax(logits, dim=1) == y_labels).float().mean()
        else:
            logits = logits.squeeze(1)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, y.float())
            preds = (torch.sigmoid(logits) > 0.5).long()
            acc = (preds == y.long()).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        if self.lr_method == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.lr_method == 'momentum':
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.09, nesterov=True)
        elif self.lr_method == 'adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=self.lr)
        elif self.lr_method == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        return torch.optim.Adam(self.parameters(), lr=self.lr)