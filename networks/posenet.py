"""
docstring
"""

import torch
from torch import nn
import torch.nn.functional as F
from networks.base.basenet import BaseNet, xavier_init_func_, \
    normal_init_func_, learned_weighting_loss, \
    fixed_weighting_loss
from networks.base.googlenet import GoogLeNet


class Regression(nn.Module):
    """Pose regression module.
    Args:
        regid: id to map the length of the last dimension of the input
               feature maps.
        with_embedding: if set True, output activations before pose regression
                        together with regressed poses, otherwise only poses.
    Return:
        xyz: global camera position.
        wpqr: global camera orientation in quaternion.
    """

    def __init__(self, regid, with_embedding=False):
        super().__init__()
        conv_in = {"regress1": 512, "regress2": 528}
        self.with_embedding = with_embedding
        if regid != "regress3":
            self.projection = nn.Sequential(nn.AvgPool2d(kernel_size=5,
                                                         stride=3),
                                            nn.Conv2d(conv_in[regid], 128,
                                                      kernel_size=1),
                                            nn.ReLU())
            self.regress_fc_pose = nn.Sequential(nn.Linear(2048, 1024),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.7))
            self.regress_fc_xyz = nn.Linear(1024, 3)
            self.regress_fc_wpqr = nn.Linear(1024, 4)
        else:
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.regress_fc_pose = nn.Sequential(nn.Linear(1024, 2048),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5))
            self.regress_fc_xyz = nn.Linear(2048, 3)
            self.regress_fc_wpqr = nn.Linear(2048, 4)

    def forward(self, x_val):
        """
        docstring
        """
        x_val = self.projection(x_val)
        x_val = self.regress_fc_pose(x_val.view(x_val.size(0), -1))
        xyz = self.regress_fc_xyz(x_val)
        wpqr = self.regress_fc_wpqr(x_val)
        wpqr = F.normalize(wpqr, p=2, dim=1)
        if self.with_embedding:
            return xyz, wpqr, x_val
        return xyz, wpqr


class PoseNet(BaseNet):
    """PoseNet model in [Kendall2015ICCV] Posenet: A convolutional
    network for real-time 6-dof camera relocalization."""

    def __init__(self, config, with_embedding=False):
        super().__init__(config)
        self.extract = GoogLeNet(with_aux=True)
        self.regress1 = Regression('regress1')
        self.regress2 = Regression('regress2')
        self.regress3 = Regression('regress3', with_embedding=with_embedding)

        # Loss params
        self.learn_weighting = config.learn_weighting
        if self.learn_weighting:
            # Learned loss weighting during training
            sx_val, sq_val = config.homo_init
            # Variances variables to learn
            self.sx_val = nn.Parameter(torch.tensor(sx_val))
            self.sq_val = nn.Parameter(torch.tensor(sq_val))
        else:
            # Fixed loss weighting with beta
            self.beta = config.beta

        self.to(self.device)
        self.init_weights_(config.weights_dict)
        self.set_optimizer_(config)

    def forward(self, x):
        if self.training:
            feat4a, feat4d, feat5b = self.extract(x)
            pose = [self.regress1(feat4a), self.regress2(feat4d),
                    self.regress3(feat5b)]
        else:
            feat5b = self.extract(x)
            pose = self.regress3(feat5b)
        return pose

    def get_inputs_(self, batch, with_label=True):
        im_val = batch['im']
        im_val = im_val.to(self.device)
        xyz = batch['xyz'].to(self.device)
        wpqr = batch['wpqr'].to(self.device)
        return im_val, xyz, wpqr

    def predict_(self, batch):
        pose = self.forward(self.get_inputs_(batch, with_label=False))
        xyz, wpqr = pose[0], pose[1]
        return xyz.data.cpu().numpy(), wpqr.data.cpu().numpy()

    def init_weights_(self, weights_dict):
        """Define how to initialize the model"""

        if weights_dict is None:
            print('Initialize all weigths')
            self.apply(xavier_init_func_)
        elif len(weights_dict.items()) == len(self.state_dict()):
            print('Load all weigths')
            self.load_state_dict(weights_dict)
        else:
            print('Init only part of weights')
            self.apply(normal_init_func_)
            self.load_state_dict(weights_dict, strict=False)

    def loss_(self, batch):
        im_val, xyz_, wpqr_ = self.get_inputs_(batch, with_label=True)
        criterion = nn.MSELoss()
        pred = self.forward(im_val)
        loss = 0
        losses = []
        loss_weighting = [0.3, 0.3, 1.0]
        if self.learn_weighting:
            loss_func = lambda loss_xyz_val, loss_wpqr_val: \
                learned_weighting_loss(loss_xyz_val, loss_wpqr_val,
                                       self.sx_val, self.sq_val)
        else:
            loss_func = lambda loss_xyz_val, loss_wpqr_val: \
                fixed_weighting_loss(loss_xyz_val, loss_wpqr_val,
                                     beta=self.beta)
        for l_val, w_val in enumerate(loss_weighting):
            xyz, wpqr = pred[l_val]
            loss_xyz = criterion(xyz, xyz_)
            loss_wpqr = criterion(wpqr, wpqr_)
            losses.append((loss_xyz, loss_wpqr))  # Remove if not necessary
            loss += w_val * loss_func(loss_xyz, loss_wpqr)
        return loss, losses
