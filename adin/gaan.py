# -*- coding: utf-8 -*-
"""Generative Adversarial Attributed Network Anomaly Detection (GAAN)"""
# Author: Ruitong Zhang <rtzhang@buaa.edu.cn>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause
# Modified by Lomoio Ugo to enable Explainability

import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_adj
from pygod.nn.functional import double_recon_loss
import torch
import warnings
import torch.nn.functional as F
from torch_geometric.nn import MLP
import time
from inspect import signature
from torch_geometric.data import Data
from pygod.metric import eval_roc_auc, eval_f1, eval_average_precision, eval_recall_at_k, eval_precision_at_k
import numpy as np
from scipy.stats import binom
from scipy.special import erf

from torch_geometric.nn import GIN
from torch_geometric import compile
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph

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

class GAAN_Explainable(torch.nn.Module):
    """
    Generative Adversarial Attributed Network Anomaly Detection

    GAAN is a generative adversarial attribute network anomaly
    detection framework, including a generator module, an encoder
    module, a discriminator module, and uses anomaly evaluation
    measures that consider sample reconstruction error and real sample
    recognition confidence to make predictions. This model is
    transductive only.

    See :cite:`chen2020generative` for details.

    Parameters
    ----------
    noise_dim :  int, optional
        Input dimension of the Gaussian random noise. Defaults: ``16``.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the generator, the other half (ceil) of the layers are
       for encoder. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module
        The backbone of GAAN is fixed to be MLP. Changing of this
        parameter will not affect the model. Default: ``None``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    device : str or torch.device, optional
        "cuda:i" where i is the gpu Index, or "cpu" using CPU. Default: ``cpu``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    weight : float, optional
        Weight between reconstruction of node feature and structure.
        Default: ``0.5``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.

    isn: bool, optional
        Whether take in input multiple Individual Specialized Networks (ISNs) rather than one convergence/divergence network.
        Using ISNs we aim to identify anomalous graphs (graph anomaly detection), while using convergence/divergence network is used to identify anomalous nodes in the graph (node anomaly detection). 
        Default: ``False``.
        
    **kwargs
        Other parameters for the backbone.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data. Outliers tend to have
        higher scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N \\times` ``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.
    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    emb : torch.Tensor or tuple of torch.Tensor or None
        The learned node hidden embeddings of shape
        :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        is ``True``. When the detector has not been fitted, ``emb`` is
        ``None``. When the detector has multiple embeddings,
        ``emb`` is a tuple of torch.Tensor.
    """

    def __init__(self,
                 in_dim,
                 noise_dim=16,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.relu,
                 backbone=None,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 device="cpu",
                 batch_size=0,
                 num_neigh=-1,
                 weight=0.5,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 isn = False,
                 **kwargs):


        self.noise_dim = noise_dim
        self.weight = weight

        # self.num_layers is 1 for sample one hop neighbors
        # In GAAN_Explainable, self.model_layers is for model layers
        self.model_layers = num_layers

        self.isn = isn 
        if self.isn:
            print("Graph Anomaly Detection task: expecting multiple ISNs")
        else:
            print("Node Anomaly Detection task: expecting only one convergence/divergence graph")
            
        if backbone is not None:
            warnings.warn('GAAN_Explainable can only use MLP as the backbone.')

        super(GAAN_Explainable, self).__init__()

         # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        generator_layers = math.floor(num_layers / 2)
        encoder_layers = math.ceil(num_layers / 2)

        self.generator = MLP(in_channels=noise_dim,
                             hidden_channels=hid_dim,
                             out_channels=in_dim,
                             num_layers=generator_layers,
                             dropout=dropout,
                             act=act,
                             **kwargs).to(device)

        self.discriminator = MLP(in_channels=in_dim,
                                 hidden_channels=hid_dim,
                                 out_channels=hid_dim,
                                 num_layers=encoder_layers,
                                 dropout=dropout,
                                 act=act,
                                 **kwargs).to(device)
        self.emb = None
        self.score_func = double_recon_loss
        self.noise_dim = noise_dim
        self.device = device
        self.gpu = -1 if self.device == "cpu" else int(self.device[-1])
        self.inner = self.generator
        self.outer = self.discriminator
        self.num_layers = num_layers
        print("GPU: {}".format(self.gpu))
        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.contamination = contamination
        self.verbose = verbose
        self.decision_score_ = None

        self.weight_decay = weight_decay

        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.gan = True
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

        # other param
        self.model = None
        self.save_emb = save_emb
        if save_emb:
            self.emb = None
        self.compile_model = compile_model


    def process_graph(self, data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]

    def forward(self, x):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed node features.
        a : torch.Tensor
            Reconstructed adjacency matrix from real samples.
        a_ : torch.Tensor
            Reconstructed adjacency matrix from fake samples.
        """

        #print("base forward")

        
        noise = torch.randn(x.shape[0], self.noise_dim).to(self.device)
        x_ = self.generator(noise)

        self.emb = self.discriminator(x)
        z_ = self.discriminator(x_)

        a = torch.sigmoid((self.emb @ self.emb.T))
        a_ = torch.sigmoid((z_ @ z_.T))
        #print("end base forward")
        return x_, a, a_

    @staticmethod
    def loss_func_g(a_):
        loss_g = F.binary_cross_entropy(a_, torch.ones_like(a_))
        return loss_g

    @staticmethod
    def loss_func_ed(a, a_):
        loss_r = F.binary_cross_entropy(a, torch.ones_like(a))
        loss_f = F.binary_cross_entropy(a_, torch.zeros_like(a_))
        return (loss_f + loss_r) / 2


    def forward_model(self, data):

        #print("model forward")


        data.batch_size = self.batch_size

        if hasattr(data, 'n_id'):
          node_idx = data.n_id.to(self.device)
        else:
          node_idx = torch.arange(len(data.x))

        data.n_id  = node_idx.to(self.device)

        
        edge_index = data.edge_index.to(self.device)

        # Add re-mapping for sampled subgraphs:
        if self.isn:
            if hasattr(data, 'n_id'):
                edge_index, _ = subgraph(data.n_id, edge_index, relabel_nodes=True)

        x = data.x.to(self.device)

        if hasattr(data, 's'):
          s = data.s.to(self.device)
        else:
          s = to_dense_adj(data.edge_index)[0]
          data.s = s

        x_, a, a_ = self.forward(x)
        #print(a_.shape)
        #print(edge_index)
        
        loss_g = self.loss_func_g(a_[edge_index])
        loss_g.requires_grad_(True)
        self.opt_in.zero_grad()
        loss_g.backward()
        self.opt_in.step()

        self.epoch_loss_in += loss_g.item() * self.batch_size

        loss = self.loss_func_ed(a[edge_index],
                                       a_[edge_index].detach())

        score = self.score_func(x=x[:self.batch_size],
                                      x_=x_[:self.batch_size],
                                      s=s[:self.batch_size, node_idx],
                                      s_=a[:self.batch_size],
                                      weight=self.weight,
                                      pos_weight_s=1,
                                      bce_s=True)

        #print("end model forward")

        return loss, score.detach().cpu()

    def __call__(self, *args, **kwargs):
        """Make the class instance callable."""

        # Get input data
        x = args[0]
        edge_index = args[1]
        print(x.shape, edge_index.shape)
        if self.isn:
            if len(args) >= 3:
                batch = args[2]  # if batch is present, handle it

        # Ensure that x requires gradients
        x.requires_grad_()  # Set requires_grad=True for node features

        # Create the data object
        data = Data(x=x, edge_index=edge_index)

        # Ensure the model is in evaluation mode
        self.eval()

        # Forward pass to compute predictions with gradients
        pred, probs = self.predict(data, return_prob=True)
        probs = probs.float().view(-1, 1)   # Ensure the target is in the right shape [N, 1]
        probs.requires_grad = True
        return probs

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
        self.decision_score_ = torch.zeros(data.x.shape[0])
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
                if self.save_emb:
                    if type(self.emb) is tuple:
                        self.emb[0][node_idx[:batch_size]] = \
                            self.emb[0][:batch_size].cpu()
                        self.emb[1][node_idx[:batch_size]] = \
                            self.emb[1][:batch_size].cpu()
                    else:
                        self.emb[node_idx[:batch_size]] = \
                            self.emb[:batch_size].cpu()
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

    def decision_function(self, data, label=None):

        self.process_graph(data)
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.eval()
        outlier_score = torch.zeros(data.x.shape[0])
        if self.save_emb:
            if type(self.hid_dim) is tuple:
                self.emb = (torch.zeros(data.x.shape[0], self.hid_dim[0]),
                            torch.zeros(data.x.shape[0], self.hid_dim[1]))
            else:
                self.emb = torch.zeros(data.x.shape[0], self.hid_dim)
        start_time = time.time()
        test_loss = 0
        for sampled_data in loader:
            loss, score = self.forward_model(sampled_data)
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.n_id
            if self.save_emb:
                if type(self.hid_dim) is tuple:
                    self.emb[0][node_idx[:batch_size]] = \
                        self.emb[0][:batch_size].cpu()
                    self.emb[1][node_idx[:batch_size]] = \
                        self.emb[1][:batch_size].cpu()
                else:
                    self.emb[node_idx[:batch_size]] = \
                        self.emb[:batch_size].cpu()

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

    def predict(self,
                data=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False):
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
            score = self.decision_function(data, label)
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

        n_ins = (self.decision_score_.view(n, 1) <= score).count_nonzero(dim=0)

        # Derive the outlier probability using Bayesian approach
        post_prob = (1 + n_ins) / (2 + n)

        # Transform the outlier probability into a confidence value
        conf = torch.Tensor(1 - binom.cdf(k, n, post_prob))

        pred = (score > self.threshold_).long()
        conf = torch.where(pred == 0, 1 - conf, conf)
        return conf

    def _process_decision_score(self):
        """Internal function to calculate key attributes:
        - threshold_: used to decide the binary label
        - label_: binary labels of training data
        """

        self.threshold_ = np.percentile(self.decision_score_,
                                        100 * (1 - self.contamination))
        self.label_ = (self.decision_score_ > self.threshold_).long()

    def __repr__(self):

        class_name = self.__class__.__name__
        init_signature = signature(self.__init__)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        params = {}
        for key in sorted([p.name for p in parameters]):
            params[key] = getattr(self, key, None)
        return '%s(%s)' % (class_name, pprint(params, offset=len(class_name)))