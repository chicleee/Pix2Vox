# -*- coding: utf-8 -*-

from PIL import Image 
import numpy as np

import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume, save_dir, n_itr):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    save_path = os.path.join(save_dir, 'voxels-%06d.png' % n_itr)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return np.array(Image.open(save_path))
