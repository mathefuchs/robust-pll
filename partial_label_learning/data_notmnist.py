""" Loads the NotMNIST dataset. """

import os
from glob import glob

import numpy as np
import torch
from matplotlib.pyplot import imread


class NotMNIST:
    """ NotMNIST dataset. """

    def __init__(self, root: str):
        images = []
        targets = []
        folders = os.listdir(root)
        for folder in sorted(folders):
            folder_path = os.path.join(root, folder)
            for ims in sorted(glob(f"{folder_path}/*.png")):
                try:
                    # img_path = os.path.join(folder_path, ims)
                    images.append(np.array(imread(ims))[np.newaxis, :])
                    # Folders are A-J so labels will be 0-9
                    targets.append(ord(folder) - 65)
                except:  # pylint: disable=bare-except
                    # Some images in the dataset are damaged
                    print(f"File {ims} is broken.")

        self.data = torch.tensor(np.concatenate(images, axis=1) * 255)
        self.targets = torch.tensor(np.array(targets))
