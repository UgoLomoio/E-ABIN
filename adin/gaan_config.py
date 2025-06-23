from torch import nn 

class GAAN_config:
    """A class used for GAAN model configs."""

    def __init__(self, noise_dim=16, hid_dim=64, num_layers=2, dropout=0.3, act=nn.ReLU, backbone=None, contamination=None, lr=0.00005, epoch=1000, gpu=-1, batch_size=1, verbose=1, isn = False, th=0.93):
        """
        Parameters
        ----------
        noise_dim (int, optional) . Input dimension of the Gaussian random noise. Defaults: 16.

        hid_dim (int, optional) . Hidden dimension of model. Default: 64.

        num_layers (int, optional) . Total number of layers in model. A half (floor) of the layers are for the generator, the other half (ceil) of the layers are for encoder. Default: 4.

        dropout (float, optional) . Dropout rate. Default: 0.

        act (callable activation function or None, optional) . Activation function if not None. Default: torch.nn.functional.relu.

        backbone (torch.nn.Module) . The backbone of GAAN is fixed to be MLP. Changing of this parameter will not affect the model. Default: None.

        contamination (float, optional) . The amount of contamination of the dataset in (0., 0.5], i.e., the proportion of outliers in the dataset. Used when fitting to define the threshold on the decision function. Default: 0.1.

        lr (float, optional) . Learning rate. Default: 0.004.

        epoch (int, optional) . Maximum number of training epoch. Default: 100.

        gpu (int) . GPU Index, -1 for using CPU. Default: -1.

        batch_size (int, optional) . Minibatch size, 0 for full batch training. Default: 0.

        verbose (int, optional) . Verbosity mode. Range in [0, 3]. Larger value for printing out more log information. Default: 0.

        isn (float, optional) . Input type. If True, the input must be multiple ISNs networks with a graph-level anomaly detection. If False, the input must be only one network, anomaly detection node-level.

        th (float, optional). Similarity threshold value to compute edges between nodes in convergence-divergence networks. 
        https://docs.pygod.org/en/latest/generated/pygod.detector.GAAN.html
        """

        self.noise_dim=noise_dim
        self.hid_dim=hid_dim
        self.num_layers=num_layers
        self.dropout=dropout
        self.act=act
        self.backbone=backbone
        self.contamination=contamination
        self.lr=lr
        self.epoch=epoch
        self.gpu=gpu
        self.batch_size=batch_size
        self.verbose=verbose
        self.isn = isn 
        self.th = th

