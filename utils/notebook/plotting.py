"""
docstring
"""
import numpy as np
import matplotlib.pyplot as plt

colors = ['#0F1F90', '#DF6767', '#67DF67', '#DFA367', '#6780DF', '#8C6132',
          '#32798C']


def plot_3d_scatter(vec, label=None):
    """Plot a vector data as a 3D scatter"""
    fig = plt.figure(figsize=(10, 10))
    ax_val = fig.add_subplot(111, projection='3d')
    ax_val.scatter(vec[:, 0], vec[:, 1], vec[:, 2], s=1, marker=">",
                   c='#125D4C', label=label)
    plt.show()


def plot_3d_scatters(data_dict, marker_shape='>', marker_size=10, title='',
                     plot_2d=False):
    """Plot every data in a dict as scatter plots within one figure using
    key names as labels """
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    if plot_2d:
        ax_val = fig.add_subplot(111)
    else:
        ax_val = fig.add_subplot(111, projection='3d')

    fig.subplots_adjust(top=0.98)
    for i, dataset in enumerate(data_dict):
        if i < len(colors):
            color = colors[i]
        else:
            color = np.random.rand(3, )
        vecs = data_dict[dataset]
        if plot_2d:
            ax_val.scatter(vecs[:, 0], vecs[:, 1], s=marker_size,
                           marker=marker_shape, c=color, label=dataset)
        else:
            ax_val.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], s=marker_size,
                           marker=marker_shape, c=color, label=dataset)
    ax_val.legend()
    plt.show()


def plot_im_loader(data_loader, row_max=2):
    """Plot some examples from a data loader"""
    for i, batch in enumerate(data_loader):
        fig = plt.figure(figsize=(20, 5))
        num = len(batch['im'])
        for j, im_val in enumerate(batch['im']):
            im_val = im_val.permute(1, 2, 0).data.numpy().astype(np.uint8)
            ax_val = fig.add_subplot(1, num, j + 1)
            ax_val.imshow(im_val)
        plt.show()
        if i >= row_max:
            break
