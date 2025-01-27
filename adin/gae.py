# -*- coding: utf-8 -*-
""" Graph Autoencoder
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

#Wrapper for GNNExplainer usage

import torch
import warnings
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.nn.models import GCN  
from torch_geometric.data import Data
from torch import nn 
import math
from torch_geometric.utils import to_dense_adj
from pygod.nn.decoder import DotProductDecoder
from torch_geometric.loader import NeighborLoader
from pygod.metric import eval_roc_auc, eval_f1, eval_average_precision, eval_recall_at_k, eval_precision_at_k
import time 
from pygod.utils import to_graph_score

from inspect import signature
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import binom
from scipy.special import erf

from torch_geometric.nn import GIN
from torch_geometric import compile


def logger(epoch=0,
           loss=0,
           score=None,
           target=None,
           time=None,
           verbose=0,
           train=True,
           deep=True):
    """
    Logger for detector.

    Parameters
    ----------
    epoch : int, optional
        The current epoch.
    loss : float, optional
        The current epoch loss value.
    score : torch.Tensor, optional
        The current outlier scores.
    target : torch.Tensor, optional
        The ground truth labels.
    time : float, optional
        The current epoch time.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    train : bool, optional
        Whether the logger is used for training.
    deep : bool, optional
        Whether the logger is used for deep detector.
    """
    if verbose > 0:
        if deep:
            if train:
                print("Epoch {:04d}: ".format(epoch), end='')
            else:
                print("Test: ", end='')

            if isinstance(loss, tuple):
                print("Loss I {:.4f} | Loss O {:.4f} | "
                      .format(loss[0], loss[1]), end='')
            else:
                print("Loss {:.4f} | ".format(loss), end='')

        if verbose > 1:
            if target is not None:
                auc = eval_roc_auc(target, score)
                print("AUC {:.4f}".format(auc), end='')

            if verbose > 2:
                if target is not None:
                    pos_size = target.nonzero().size(0)
                    rec = eval_recall_at_k(target, score, pos_size)
                    pre = eval_precision_at_k(target, score, pos_size)
                    ap = eval_average_precision(target, score)

                    contamination = sum(target) / len(target)
                    threshold = np.percentile(score,
                                              100 * (1 - contamination))
                    pred = (score > threshold).long()
                    f1 = eval_f1(target, pred)

                    print(" | Recall {:.4f} | Precision {:.4f} "
                          "| AP {:.4f} | F1 {:.4f}"
                          .format(rec, pre, ap, f1), end='')

            if time is not None:
                print(" | Time {:.2f}".format(time), end='')

        print()

def pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int, optional
        The offset at the beginning of each line.
    printer : callable, optional
        The function to convert entries to strings, typically
        the builtin str or repr.
    """

    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


def is_fitted(detector, attributes=None):
    """
    Check if the detector is fitted.

    Parameters
    ----------
    detector : pygod.detector.Detector
        The detector to check.
    attributes : list, optional
        The attributes to check.
        Default: ``None``.

    Returns
    -------
    is_fitted : bool
        Whether the detector is fitted.
    """
    if attributes is None:
        attributes = ['model']
    assert all(hasattr(detector, attr) and
               eval('detector.%s' % attr) is not None
               for attr in attributes), \
        "The detector is not fitted yet"


class GAE_Explainable(torch.nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.relu,
                 backbone=GCN,
                 recon_s=False,
                 sigmoid_s=False,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gan = False,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False,
                 compile_model=False,
                 device = "cpu",
                 **kwargs):

        super(GAE_Explainable, self).__init__()
        if num_neigh != 0 and backbone == MLP:
            warnings.warn('MLP does not use neighbor information.')
            num_neigh = 0

        self.recon_s = recon_s
        self.sigmoid_s = sigmoid_s

        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.contamination = contamination
        self.verbose = verbose
        self.decision_score_ = None

        
        # model param
        self.in_dim = None
        self.num_nodes = None
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.backbone = backbone
        self.kwargs = kwargs
        
        self.emb = None 

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = device
        self.gpu = -1 if self.device == "cpu" else int(self.device[-1])
        self.batch_size = batch_size
        self.gan = gan
        self.in_dim = in_dim

        if type(num_neigh) is int:
            self.num_neigh = [num_neigh] * self.num_layers
        elif type(num_neigh) is list:
            if len(num_neigh) != self.num_layers:
                raise ValueError('Number of neighbors should have the '
                                 'same length as hidden layers dimension or'
                                 'the number of layers.')
            self.num_neigh = num_neigh
        else:
            raise ValueError('Number of neighbors must be int or list of int')

   
        self.compile_model = compile_model

        self.backbone = backbone

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.encoder = self.backbone(in_channels=in_dim,
                                     hidden_channels=hid_dim,
                                     out_channels=hid_dim,
                                     num_layers=encoder_layers,
                                     dropout=dropout,
                                     act=act).to(self.device)#, **kwargs)
        self.recon_s = recon_s
        if self.recon_s:
            self.decoder = DotProductDecoder(in_dim=hid_dim,
                                             hid_dim=hid_dim,
                                             num_layers=decoder_layers,
                                             dropout=dropout,
                                             act=act,
                                             sigmoid_s=sigmoid_s,
                                             backbone=self.backbone).to(self.device)#, **kwargs)
        else:
            self.decoder = self.backbone(in_channels=hid_dim,
                                         hidden_channels=hid_dim,
                                         out_channels=in_dim,
                                         num_layers=decoder_layers,
                                         dropout=dropout,
                                         act=act).to(self.device)#, **kwargs)

        self.loss_func = F.mse_loss
       

    def forward(self, x, edge_index, edge_mask = None):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed embeddings.
        """
        
        if self.backbone == MLP:
            self.emb = self.encoder(x, None)
            x_ = self.decoder(self.emb, None)
        else:
            self.emb = self.encoder(x, edge_index, edge_weight = edge_mask)
            x_ = self.decoder(self.emb, edge_index, edge_weight = edge_mask)
        return x_

    def predict(self,
                data=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False, 
                edge_mask = None
                ):
        """Prediction for testing data using the fitted detector.
        Return predicted labels by default.

        Parameters
        ----------
        data : torch_geometric.data.Data, optional
            The testing graph. If ``None``, the training data is used.
            Default: ``None``.
        label : torch.Tensor, optional
            The optional outlier ground truth labels used for testing.
            Default: ``None``.
        return_pred : bool, optional
            Whether to return the predicted binary labels. The labels
            are determined by the outlier contamination on the raw
            outlier scores. Default: ``True``.
        return_score : bool, optional
            Whether to return the raw outlier scores.
            Default: ``False``.
        return_prob : bool, optional
            Whether to return the outlier probabilities.
            Default: ``False``.
        prob_method : str, optional
            The method to convert the outlier scores to probabilities.
            Two approaches are possible:

            1. ``'linear'``: simply use min-max conversion to linearly
            transform the outlier scores into the range of
            [0,1]. The model must be fitted first.

            2. ``'unify'``: use unifying scores,
            see :cite:`kriegel2011interpreting`.

            Default: ``'linear'``.
        return_conf : boolean, optional
            Whether to return the model's confidence in making the same
            prediction under slightly different training sets.
            See :cite:`perini2020quantifying`. Default: ``False``.

        Returns
        -------
        pred : torch.Tensor
            The predicted binary outlier labels of shape :math:`N`.
            0 stands for inliers and 1 for outliers.
            Only available when ``return_label=True``.
        score : torch.Tensor
            The raw outlier scores of shape :math:`N`.
            Only available when ``return_score=True``.
        prob : torch.Tensor
            The outlier probabilities of shape :math:`N`.
            Only available when ``return_prob=True``.
        conf : torch.Tensor
            The prediction confidence of shape :math:`N`.
            Only available when ``return_conf=True``.
        """

        is_fitted(self, ['decision_score_', 'threshold_', 'label_'])

        output = ()
        if data is None:
            score = self.decision_score_
            logger(score=self.decision_score_,
                   target=label,
                   verbose=self.verbose,
                   train=False)
        else:
            score = self.decision_function(data, edge_mask=edge_mask, label=label)
        if return_pred:
            pred = (score > self.threshold_).long()
            output += (pred,)
        if return_score:
            output += (score,)
        if return_prob:
            prob = self._predict_prob(score, prob_method)
            output += (prob,)
        if return_conf:
            conf = self._predict_conf(score)
            output += (conf,)

        if len(output) == 1:
            return output[0]
        else:
            return output

    def process_graph(self, data, recon_s=False):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        recon_s : bool, optional
            Reconstruct the structure instead of node feature .
        """
        if recon_s:
            data.s = to_dense_adj(data.edge_index)[0]

    def forward_model(self, data, edge_mask = None):

        data.batch_size = self.batch_size

        if hasattr(data, 'n_id'):
          node_idx = data.n_id
        else:
          node_idx = torch.arange(len(data.x))

        data.n_id  = node_idx

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        if edge_mask is not None:
            edge_mask = edge_mask.to(self.device)

        if self.recon_s:
            s = data.s.to(self.device)[:, node_idx]

        h = self.forward(x, edge_index, edge_mask)

        target = s if self.recon_s else x
        score = torch.mean(self.loss_func(target[:self.batch_size],
                                                h[:self.batch_size],
                                                reduction='none'), dim=1)
        loss = torch.mean(score)

        return loss, score#.detach().cpu()
    
        
    def fit(self, data, label=None):

        self.process_graph(data)
        self.num_nodes, self.in_dim = data.x.shape
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        if not self.gan:
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            self.opt_in = torch.optim.Adam(self.inner.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)
            optimizer = torch.optim.Adam(self.outer.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

        self.train()
        self.decision_score_ = torch.zeros(data.x.shape[0]).to(self.device)
        for epoch in range(self.epoch):
            start_time = time.time()
            epoch_loss = 0
            if self.gan:
                self.epoch_loss_in = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id

                loss, score = self.forward_model(sampled_data)
                epoch_loss += loss.item() * batch_size

                self.decision_score_[node_idx[:batch_size]] = score

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            loss_value = epoch_loss / data.x.shape[0]
            if self.gan:
                loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)
            logger(epoch=epoch,
                   loss=loss_value,
                   score=self.decision_score_,
                   target=label,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)

        self._process_decision_score()
        return self

    def __call__(self, *args, **kwargs):
        """Make the class instance callable."""

        x = args[0].to(self.device)
        edge_index = args[1].to(self.device)
        if len(args) > 2:
            edge_mask = args[2].to(self.device)
        else:
            edge_mask = None 

        data = Data(x=x, edge_index=edge_index)

        self.eval()
        pred, probs = self.predict(data, edge_mask, return_prob=True)
        pred = torch.stack([torch.tensor([1, 0]) if p.item() == 0 else torch.tensor([0, 1]) for p in pred]).view(-1, 1).to(self.device)
        # Stack the probabilities and ensure gradients propagate
        #probs = torch.stack([1 - probs, probs], dim=1).view(-1, 1).to(self.device)
        print(pred.shape, pred)
        return pred

    

    def _predict_prob(self, score, method='linear'):
        """Predict the probabilities of being outliers. Two approaches
        are possible:

        'linear': simply use min-max conversion to linearly
                  transform the outlier scores into the range of
                  [0,1]. The model must be fitted first.

        'unify': use unifying scores,
                 see :cite:`kriegel2011interpreting`.

        Parameters
        ----------
        score : torch.Tensor
            The outlier scores of shape :math:`N`.

        method : str
            probability conversion method. It must be one of
            'linear' or 'unify'. Default: ``linear``.

        Returns
        -------
        prob : torch.Tensor
            The outlier probabilities of shape :math:`N`.
        """

        if method == 'linear':
            train_score = self.decision_score_
            prob = score - train_score.min()
            prob /= train_score.max() - train_score.min()
            prob = prob.clamp(0, 1)
        elif method == 'unify':
            mu = torch.mean(self.decision_score_)
            sigma = torch.std(self.decision_score_)
            pre_erf_score = (score - mu) / (sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            prob = erf_score.clamp(0, 1)
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')
        return prob

    def _predict_conf(self, score):
        """Predict the model's confidence in making the same prediction
        under slightly different training sets.
        See :cite:`perini2020quantifying`.

        Parameters
        ----------
        score : torch.Tensor
            The outlier score of shape :math:`N`.

        Returns
        -------
        conf : torch.Tensor
            The prediction confidence of shape :math:`N`.
        """

        n = len(self.decision_score_)
        k = n - int(n * self.contamination)

        n_ins = (self.decision_score_.cpu().detach().view(n, 1) <= score.cpu().detach()).count_nonzero(dim=0)

        # Derive the outlier probability using Bayesian approach
        post_prob = (1 + n_ins) / (2 + n)

        # Transform the outlier probability into a confidence value
        conf = torch.Tensor(1 - binom.cdf(k, n, post_prob))

        pred = (score > self.threshold_).long().cpu().detach()
        conf = torch.where(pred == 0, 1 - conf, conf)
        return conf

    def _process_decision_score(self):
        """Internal function to calculate key attributes:
        - threshold_: used to decide the binary label
        - label_: binary labels of training data
        """

        self.threshold_ = torch.quantile(self.decision_score_,
                                        (1 - self.contamination))
        self.label_ = (self.decision_score_ > self.threshold_).long()

    def decision_function(self, data, edge_mask = None, label=None):

        self.process_graph(data)
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.eval()
        outlier_score = torch.zeros(data.x.shape[0]).to(self.device)

        start_time = time.time()
        test_loss = 0
        for sampled_data in loader:
            loss, score = self.forward_model(sampled_data, edge_mask)
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.n_id

            test_loss = loss.item() * batch_size
            outlier_score[node_idx[:batch_size]] = score

        loss_value = test_loss / data.x.shape[0]
        if self.gan:
            loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)

        logger(loss=loss_value,
               score=outlier_score,
               target=label,
               time=time.time() - start_time,
               verbose=self.verbose,
               train=False)
        return outlier_score

    def __repr__(self):

        class_name = self.__class__.__name__
        init_signature = signature(self.__init__)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        params = {}
        for key in sorted([p.name for p in parameters]):
            params[key] = getattr(self, key, None)
        return '%s(%s)' % (class_name, pprint(params, offset=len(class_name)))
