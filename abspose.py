"""
docstring
"""
import os
import time
from torch.utils import data
import networks
import torch
import numpy as np
from utils.common.config_parser import AbsPoseConfig
from utils.common.visdom_templates import PoseNetVisTmp, OptimSearchVisTmp
from utils.datasets.abspose import AbsPoseDataset
from utils.datasets.preprocess import get_transform_ops
from utils.common.setup import make_deterministic, lprint,\
    cal_quat_angle_error, config_to_string


def setup_config(config):
    """
    docstring
    """
    print('Setup configurations...')
    # Seeding
    make_deterministic(config.seed)

    # Setup logging dir
    if not os.path.exists(config.odir):
        os.makedirs(config.odir)
    config.log = os.path.join(config.odir, 'log.txt') if config.training else \
        os.path.join(config.odir, 'test_results.txt')
    config.ckpt_dir = os.path.join(config.odir, 'ckpt')
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)

    # Setup running devices
    if torch.cuda.is_available():
        print(f'Use GPU device:{config.gpu}.')
        config.device = torch.device(f'cuda:{config.gpu}')
    else:
        print('No GPU available, use CPU device.')
        config.device = torch.device("cpu")
    delattr(config, 'gpu')

    if config.validate:
        config.validate = config.validate[0]

    # Setup datasets
    config.data_class = AbsPoseDataset

    # Define image preprocessing
    im_mean = os.path.join(config.data_root, config.dataset,
                           config.image_mean) if config.image_mean else None
    if config.crop:
        crop = 'random' if config.training else 'center'
    else:
        crop = None
    config.ops = get_transform_ops(config.rescale, im_mean, crop,
                                   crop_size=config.crop,
                                   normalize=config.normalize)
    config.val_ops = get_transform_ops(config.rescale, im_mean, 'center',
                                       crop_size=config.crop,
                                       normalize=config.normalize)
    delattr(config, 'crop')
    delattr(config, 'rescale')
    delattr(config, 'normalize')

    # Model initialization
    config.start_epoch = 0
    config.weights_dir = None
    config.weights_dict = None
    config.optimizer_dict = None
    if config.pretrained:
        config.weights_dir = config.pretrained[0]
        config.weights_dict = torch.load(config.weights_dir)
    if config.resume:
        config.weights_dir = config.resume[0]
        checkpoint = torch.load(config.weights_dir)
        assert config.network == checkpoint['network']
        config.start_epoch = checkpoint['last_epoch'] + 1
        config.weights_dict = checkpoint['state_dict']
        config.optimizer_dict = checkpoint['optimizer']
    delattr(config, 'resume')
    delattr(config, 'pretrained')

    # Setup optimizer
    # optim = config.optim
    optim_tag = ''
    if config.optim == 'Adam':
        optim_tag = f'Adam_eps{config.epsilon}'
        delattr(config, 'momentum')
    elif config.optim == 'SGD':
        optim_tag = f'SGD_mom{config.momentum}'
        delattr(config, 'epsilon')
    optim_tag = f'{optim_tag}_{config.lr_init}'
    if config.lr_decay:
        config.lr_decay_step = int(config.lr_decay[1])
        config.lr_decay_factor = float(config.lr_decay[0])
        config.lr_decay = True
        optim_tag = f'{optim_tag}_lrd{config.lr_decay_step}-' \
                    f'{config.lr_decay_factor}'
    optim_tag = f'{optim_tag}_wd{config.weight_decay}'
    config.optim_tag = optim_tag


def train(net, config, log, train_loader, val_loader=None):
    """
    docstring
    """
    optim_search = True
    # Setup visualizer
    if not optim_search:
        visman, tloss_meter, pos_acc_meter, rot_acc_meter, _, \
        homo_meters = PoseNetVisTmp.get_meters(with_losses=False,
                                               with_homos=True, config=None)
    else:
        visman, tloss_meter, pos_acc_meter, \
        rot_acc_meter = OptimSearchVisTmp.get_meters(config=None)
        homo_meters = None
    start_time = time.time()
    print(f'Start training from {config.start_epoch} to {config.epochs}.')
    for epoch in range(config.start_epoch, config.epochs):
        net.train()  # Switch to training mode
        loss, _ = net.train_epoch(train_loader, epoch)
        lprint(f'Epoch {epoch + 1}, loss:{loss}', log)
        # Update homo variable meters
        if config.learn_weighting and homo_meters is not None:
            homo_meters[0].update(x_value=epoch + 1, y_value=net.sx)
            homo_meters[1].update(x_value=epoch + 1, y_value=net.sq)
        tloss_meter.update(x_value=epoch + 1, y_value=loss)
        if config.validate and (
                epoch + 1) % config.validate == 0 and epoch > 0:
            # Evaluate on validation set
            abs_err = test(net, config, log, val_loader)
            ckpt = {'last_epoch': epoch,
                    'network': config.network,
                    'state_dict': net.state_dict(),
                    'optimizer': net.optimizer.state_dict(),
                    'abs_err': abs_err,
                    }
            ckpt_name = 'checkpoint_{epoch+1}_{abs_err[0]:.2f}m_' \
                        '{abs_err[1]:.2f}deg.pth'
            torch.save(ckpt, os.path.join(config.ckpt_dir, ckpt_name))
            lprint(f'Save checkpoint: {ckpt_name}', log)

            # Update validation acc
            pos_acc_meter.update(x_value=epoch + 1, y_value=abs_err[0])
            rot_acc_meter.update(x_value=epoch + 1, y_value=abs_err[1])
        visman.save_state()
    lprint(f'Total training time {(time.time() - start_time)}s', log)


def test(net, config, log, data_loader, err_thres=(2, 5)):
    """
    docstring
    """
    print(f'Evaluate on dataset:{data_loader.dataset.dataset}')
    _ = config
    net.eval()
    pos_err = []
    ori_err = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            xyz, wpqr = net.predict_(batch)
            xyz_ = batch['xyz'].data.numpy()
            wpqr_ = batch['wpqr'].data.numpy()
            t_err = np.linalg.norm(xyz - xyz_, axis=1)
            q_err = cal_quat_angle_error(wpqr, wpqr_)
            pos_err += list(t_err)
            ori_err += list(q_err)
    err = (np.median(pos_err), np.median(ori_err))
    passed = 0
    for i, perr in enumerate(pos_err):
        if perr < err_thres[0] and ori_err[i] < err_thres[1]:
            passed += 1
    lprint(f'Accuracy: ({err[0]:.2f}m, {err[1]:.2f}deg) Pass({err_thres[0]}m, '
           f'{err_thres[1]}deg): {100.0 * passed / i}% ', log)
    return err


def main():
    """
    docstring
    """
    # Setup
    config = AbsPoseConfig().parse()
    setup_config(config)
    with open(config.log, 'a', encoding="utf8") as log:

        lprint(config_to_string(config), log)

        # Datasets configuration
        data_src = AbsPoseDataset(config.dataset, config.data_root,
                                  config.pose_txt, config.ops)
        data_loader = data.DataLoader(data_src, batch_size=config.batch_size,
                                      shuffle=config.training,
                                      num_workers=config.num_workers)
        lprint(f'Dataset total samples: {len(data_src)}')

        if config.validate:
            val_data_src = AbsPoseDataset(config.dataset, config.data_root,
                                          config.val_pose_txt, config.val_ops)
            val_loader = data.DataLoader(val_data_src,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         num_workers=config.num_workers)
        else:
            val_loader = None

        if config.weights_dir:
            lprint(f'Load weights dict {config.weights_dir}')
        net = networks.__dict__[config.network](config)
        lprint(f'Model params: {len(net.state_dict())} Optimizer params: '
               f'{len(net.optimizer.param_groups[0]["params"])}')

        if config.training:
            train(net, config, log, data_loader, val_loader)
        else:
            test(net, config, log, data_loader)

if __name__ == '__main__':
    main()
