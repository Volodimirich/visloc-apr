"""
docstring
"""
import json
import torch
import visdom
import numpy as np


def validate_input_(x_value):
    """
    docstring
    """
    retval = None
    if isinstance(x_value, np.ndarray):
        retval = x_value
    elif isinstance(x_value, (float, int, np.float32)):
        retval = np.array([x_value])
    elif isinstance(x_value, torch.Tensor):
        x_value = x_value.cpu().data.numpy()
        if x_value.ndim == 0:
            x_value = x_value.reshape(1)
        retval = x_value
    return retval


class VisObj:
    """
    docstring
    """
    def __init__(self, viz, env, win, legend, opts):
        """
        docstring
        """
        self.viz = viz
        self.env = env
        self.win = win
        self.legend = legend
        self.opts = opts
        self.inserted = False

    def update(self, x_value, y_value):
        """
        docstring
        """
        if self.viz.get_window_data(win=self.win, env=self.env) == '':
            update_type = None
        elif self.inserted:
            update_type = 'append'
        else:
            update_type = 'new'
        self.viz.line(X=validate_input_(x_value), Y=validate_input_(y_value),
                      env=self.env, win=self.win, name=self.legend,
                      opts=self.opts, update=update_type)
        self.inserted = True

    def clear(self):
        """
        docstring
        """
        self.viz.line(X=None, Y=None, env=self.env, win=self.win,
                      name=self.legend, update='remove')

    def close(self):
        """
        docstring
        """
        self.viz.close(env=self.env, win=self.win)

    def __repr__(self):
        """
        docstring
        """
        return f'>> Visdom Object with env:{self.env} win:{self.win} ' \
               f'legend:{self.legend}\n'


def get_default_opts_(title):
    """
    docstring
    """
    layout = {'plotly': dict(title=title, xaxis={'title': 'epochs'})}
    opts = dict(mode='lines', showlegend=True, layoutopts=layout)
    # opts=dict(mode='marker+lines',
    #      markersize=5,
    #      markersymbol='dot',
    #      markers={'line': {'width': 0.5}},
    #      showlegend=True, layoutopts=layout)
    return opts


class VisManager:
    """
    docstring
    """
    def __init__(self, env, win_pref='', targets=None, host='localhost',
                 port='8097', enable_log=False):
        """
        docstring
        """
        if targets is None:
            targets = {}
        server = f'http://{host}'
        self.viz = visdom.Visdom(server=server, port=int(port))
        self.env = env
        assert self.viz.check_connection(),\
            f'Visdom server is not active on server {server}:{port}'
        print(f'Visdom server connected on {server}:{port}')

        # Appending to previous log
        if enable_log:
            self.log_win = f'{win_pref}_log'
            prev_txt = self.viz.get_window_data(env=self.env, win=self.log_win)
            if prev_txt == '':
                self.txt = prev_txt
            else:
                self.txt = json.loads(prev_txt)['content']

        # Initialize all target windows
        self.win_pool = {}
        for win in targets:
            self.win_pool[win] = {}
            opts = get_default_opts_(win)
            for legend in targets[win]:
                visobj = VisObj(self.viz, self.env, win, legend, opts)
                self.win_pool[win][legend] = visobj
                print(f'Initialize {str(visobj)}')

    def get_win(self, win, legend):
        """
        docstring
        """
        return self.win_pool[win][legend]

    def save_state(self):
        """
        docstring
        """
        self.viz.save(envs=[self.env])

    def clear_all(self):
        """
        for win in self.win_pool:
            for legend in self.win_pool[win]:
                self.win_pool[win][legend].clear()
        """
        self.win_pool.items().clear()

    def log(self, txt):
        """
        docstring
        """
        self.txt += f'{txt}<br>'
        self.viz.text(text=self.txt, env=self.env, win=self.log_win)

    def print_(self):
        """
        print('Visdom Manager Window Pool:\n')
        for win in self.win_pool:
            for legend in self.win_pool[win]:
                print(self.win_pool[win][legend])
        """
        print('Visdom Manager Window Pool:\n')
        print(self.win_pool.items())


class DummyVisObj:
    """
    docstring
    """
    def __init__(self, *args, **kwargs):
        """
        docstring
        """

    def update(self, *args, **kwargs):
        """
        docstring
        """

    def close(self):
        """
        docstring
        """


class DummyVisManager:
    """
    docstring
    """
    def __init__(self, *args, **kwargs):
        """
        docstring
        """

    def save_state(self):
        """
        docstring
        """
