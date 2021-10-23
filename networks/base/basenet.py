"""
docstring
"""

import torch
from torch import nn


def normal_init_func_(m_value):
    """
    docstring
    """
    classname = m_value.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m_value.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m_value.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m_value.weight.data, 1.0, 0.02)
        nn.init.constant_(m_value.bias.data, 0.0)


def xavier_init_func_(m_value):
    """
    docstring
    """
    classname = m_value.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m_value.weight.data)
        nn.init.constant_(m_value.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m_value.weight.data)
        nn.init.constant_(m_value.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m_value.weight.data, 1.0, 0.02)
        nn.init.constant_(m_value.bias.data, 0.0)


def learned_weighting_loss(loss_pos, loss_rot, sx_number, sq_number):
    """The weighted loss function that learns variables sx and sy to balance
    the positional loss and the rotational loss"""
    return (-1 * sx_number).exp() * loss_pos + sx_number +\
           (-1 * sq_number).exp() * loss_rot + sq_number


def fixed_weighting_loss(loss_pos, loss_rot, beta=1.0):
    """The weighted loss function with beta as a hyperparameter to balance the
    positional loss and the rotational loss"""
    return loss_pos + beta * loss_rot


def kaiming_normal_init_func_(m_value):
    """
    docstring
    """
    classname = m_value.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m_value.weight.data, mode='fan_out',
                                nonlinearity='relu')
        if m_value.bias is not None:  # Incase bias is turned off
            nn.init.constant_(m_value.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m_value.weight.data, 1.0)
        nn.init.constant_(m_value.bias.data, 0.0)


class BaseNet(nn.Module):
    """
    docstring
    """
    def __init__(self, config):
        """
        docstring
        """
        super().__init__()
        self.lr_scheduler = None
        self.optimizer = torch.optim.SGD(self.parameters(), lr=config.lr_init,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay,
                                         nesterov=False)
        self.config = config
        self.device = config.device

    @property
    def num_params(self):
        """
        docstring
        """
        return sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        """Defines the computation performed at every call.
        Inherited from Superclass torch.nn.Module.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_inputs_(self, batch, with_label=True):

        """Define how to parse the data batch for the network training/testing.
        This allows the flexibility for different training data formats
        in different applications. Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def init_weights_(self, weights_dict):
        """Define how to initialize the weights of the network.
        Should be overridden by all subclasses, since it
        normally differs according to the network models.
        """
        raise NotImplementedError

    def loss_(self, batch):
        """Define how to calculate the loss  for the network.
        Should be overridden by all subclasses, since different
        applications or network models may have different types
        of targets and the corresponding criterions to evaluate
        the predictions.
        """
        raise NotImplementedError

    def set_optimizer_(self, config):
        """
        docstring
        """
        if config.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=config.lr_init,
                                              eps=config.epsilon,
                                              weight_decay=config.weight_decay)
        elif config.optim == 'SGD':
            pass

        # Initialize optimizer from a state if available
        if config.optimizer_dict and config.training:
            self.optimizer.load_state_dict(config.optimizer_dict)

        if config.lr_decay:
            self.lr_scheduler = \
                torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                step_size=config.lr_decay_step,
                                                gamma=config.lr_decay_factor,
                                                last_epoch=config.start_epoch - 1)
        else:
            pass

    def predict_(self, batch):
        """
        docstring
        """
        return self.forward(*self.get_inputs_(batch, with_label=False))

    def optim_step_(self, batch):
        """
        docstring
        """
        self.optimizer.zero_grad()
        loss, losses = self.loss_(batch)
        loss.backward()
        self.optimizer.step()
        return loss, losses

    def train_epoch(self, data_loader, epoch):
        """
        docstring
        """
        loss = None
        losses = None
        if self.lr_scheduler:
            self.lr_scheduler.step()
        for i, batch in enumerate(data_loader):
            loss, losses = self.optim_step_(batch)
        return loss, losses

    def save_weights_(self, sav_path):
        """
        docstring
        """
        torch.save(self.state_dict(), sav_path)

    def print_(self):
        """
        docstring
        """
        for k_value, v_value in self.state_dict().items():
            print(k_value, v_value.size())
