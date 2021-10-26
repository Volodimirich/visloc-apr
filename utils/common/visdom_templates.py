"""
docstring
"""

from utils.common.visdom_manager import DummyVisManager, DummyVisObj, VisManager


class OptimSearchVisTmp:
    """
    docstring
    """
    def __init__(self):
        """
        docstring
        """
        self.validate = None
        self.start_epoch = None
        self.visport = None
        self.vishost = None
        self.optim_tag = None
        self.viswin = None
        self.visenv = None

    @staticmethod
    def get_meters(config):
        """
        docstring
        """
        if config.visenv is None:
            visman = DummyVisManager()
            loss_meter = DummyVisObj()
            pos_acc_meter = DummyVisObj()
            rot_acc_meter = DummyVisObj()
        else:
            # Setup visualizer
            if config.viswin is not None:
                loss_win_tag = f'{config.viswin}-Loss'
                acc_win_tag = f'{config.viswin}-Val'
            else:
                loss_win_tag = 'Loss'
                acc_win_tag = 'Val'
            legend = config.optim_tag
            pos_legend = f'pos_{legend}'
            rot_legend = f'rot_{legend}'
            targets = {loss_win_tag: [legend], acc_win_tag: [pos_legend,
                                                             rot_legend]}
            visman = VisManager(config.visenv, config.viswin, targets,
                                host=config.vishost, port=config.visport,
                                enable_log=False)
            # now = datetime.datetime.now()
            loss_meter = visman.get_win(loss_win_tag, legend)
            pos_acc_meter = visman.get_win(acc_win_tag, pos_legend)
            rot_acc_meter = visman.get_win(acc_win_tag, rot_legend)

            # Set states
            if config.start_epoch >= 1:
                loss_meter.inserted = True
            if config.start_epoch >= config.validate:
                pos_acc_meter.inserted = True
                rot_acc_meter.inserted = True
        return visman, loss_meter, pos_acc_meter, rot_acc_meter


class PoseNetVisTmp:
    """
    docstring
    """
    def __init__(self):
        """
        docstring
        """
        self.validate = None
        self.start_epoch = None
        self.visport = None
        self.vishost = None
        self.dataset = None
        self.viswin = None
        self.visenv = None

    @staticmethod
    def get_meters(config, with_losses=False, with_homos=False):
        """
        docstring
        """
        if config.visenv is None:
            visman = DummyVisManager()
            tloss_meter = DummyVisObj()
            losses_meters = [(DummyVisObj(), DummyVisObj()) for i in range(3)]
            homo_meters = [DummyVisObj(), DummyVisObj()]
            pos_acc_meter = DummyVisObj()
            rot_acc_meter = DummyVisObj()
        else:
            # Setup visualizer
            if config.viswin is not None:
                acc_win_tag = f'{config.viswin}-Val'
                loss_win_tag = f'{config.viswin}-Losses'
                homo_win_tag = f'{config.viswin}-Homos'
            else:
                acc_win_tag = f'{config.dataset}-Loss-Val'
                loss_win_tag = f'{config.dataset}-Losses'
                homo_win_tag = f'{config.viswin}-Homos'
            loss_win_legends = []
            for i in range(3):
                loss_win_legends.append(f'l{i}_x')
                loss_win_legends.append(f'l{i}_q')
            homo_win_legends = ['sx', 'sq']
            tloss_legend = 'loss'
            pos_legend = 'pos'
            rot_legend = 'rot'
            targets = {acc_win_tag: [tloss_legend, pos_legend, rot_legend]}
            if with_losses:
                targets.update({loss_win_tag: loss_win_legends})
            if with_homos:
                targets.update({homo_win_tag: homo_win_legends})
            visman = VisManager(config.visenv, config.viswin, targets,
                                host=config.vishost, port=config.visport)
            # now = datetime.datetime.now()

            # Meters for losses
            losses_meters = []
            if with_losses:
                for i in range(3):
                    losses_meters.append((visman.get_win(loss_win_tag,
                                                         f'l{i}_x'),
                                          visman.get_win(loss_win_tag,
                                                         f'l{i}_q')))

            # Meters for homoscedastic uncertainties
            homo_meters = []
            if with_homos:
                homo_meters = [visman.get_win(homo_win_tag, 'sx'),
                               visman.get_win(homo_win_tag, 'sq')]

            # Meters for loss and val
            tloss_meter = visman.get_win(acc_win_tag, tloss_legend)
            pos_acc_meter = visman.get_win(acc_win_tag, pos_legend)
            rot_acc_meter = visman.get_win(acc_win_tag, rot_legend)

            # Set states
            if config.start_epoch >= 1:
                tloss_meter.inserted = True
                homo_meters[0].inserted = True
                homo_meters[1].inserted = True
                for i, _ in enumerate(losses_meters):
                    losses_meters[i][0].inserted = True
                    losses_meters[i][1].inserted = True
            if config.start_epoch >= config.validate:
                pos_acc_meter.inserted = True
                rot_acc_meter.inserted = True
        return visman, tloss_meter, pos_acc_meter, rot_acc_meter, \
               losses_meters, homo_meters
