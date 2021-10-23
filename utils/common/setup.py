"""
docstring
"""
from collections import OrderedDict
import random
import torch
import numpy as np


def load_weights_to_gpu(weights_dir=None, gpu=None):
    """
    docstring
    """
    weights_dict = None
    if weights_dir is not None:
        if gpu is None:
            map_location = lambda storage, loc: storage
        else:
            map_location = lambda storage, loc: storage.cuda(gpu)
        weights = torch.load(weights_dir, map_location=map_location)
        if isinstance(weights, OrderedDict):
            weights_dict = weights
        elif isinstance(weights, dict) and 'state_dict' in weights:
            weights_dict = weights['state_dict']
    return weights_dict


def make_deterministic(seed):
    """
    docstring
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Important also


def lprint(ms_val, log=None):
    """Print message to console and to a log file"""
    print(ms_val)
    if log:
        log.write(ms_val + '\n')
        log.flush()


def config_to_string(config, html=False):
    """
    docstring
    """
    print_ignore = ['weights_dict', 'optimizer_dict']
    args = vars(config)
    separator = '<br>' if html else '\n'
    confstr = ''
    confstr += f'------------ Configuration -------------{separator}'
    for k_val, v_val in sorted(args.items()):
        if k_val in print_ignore:
            if v_val is not None:
                confstr += f'{k_val}:{len(v_val)}{separator}'
            continue
        confstr += f'{k_val}:{str(v_val)}{separator}'
    confstr += f'----------------------------------------{separator}'
    return confstr


def cal_quat_angle_error(label, pred):
    """
    docstring
    """
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    q1_val = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2_val = label / np.linalg.norm(label, axis=1, keepdims=True)
    d_val = np.abs(np.sum(np.multiply(q1_val, q2_val), axis=1,
                          keepdims=True))  # Here we have abs()
    d_val = np.clip(d_val, a_min=-1, a_max=1)
    error = 2 * np.degrees(np.arccos(d_val))
    return error
