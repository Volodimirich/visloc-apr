"""
docstring
"""
import os
import numpy as np
from PIL import Image
from torch.utils import data

__all__ = ['AbsPoseDataset']


class AbsPoseDataset(data.Dataset):
    """
    docstring
    """
    def __init__(self, dataset, root, pose_txt, transforms=None):
        """
        docstring
        """
        self.dataset = dataset
        self.transforms = transforms
        self.pose_txt = os.path.join(root, dataset, pose_txt)
        self.ims, self.poses = self.parse_abs_pose_txt(self.pose_txt)
        self.data_dir = os.path.join(root, dataset)
        self.num = len(self.ims)

    def __getitem__(self, index):
        """Return:
           dict:'im' is the image tensor
                'xyz' is the absolute position of the image
                'wpqr' is the  absolute rotation quaternion of the image
        """
        data_dict = {}
        im_val = self.ims[index]
        data_dict['im_ref'] = im_val
        im_val = Image.open(os.path.join(self.data_dir, im_val))
        if self.transforms:
            im_val = self.transforms(im_val)
        data_dict['im'] = im_val
        data_dict['xyz'] = self.poses[index][0]
        data_dict['wpqr'] = self.poses[index][1]
        return data_dict

    def __len__(self):
        """
        docstring
        """
        return self.num

    @staticmethod
    def parse_abs_pose_txt(fpath):
        """Define how to parse files to get pose labels
           Our pose label format:
                3 header lines
                list of samples with format:
                    image x y z w p q r
        """
        poses = []
        ims = []
        f_path = open(fpath, encoding="utf8")
        for line in f_path.readlines()[3::]:    # Skip 3 header lines
            cur = line.split(' ')
            xyz = np.array([float(v) for v in cur[1:4]], dtype=np.float32)
            wpqr = np.array([float(v) for v in cur[4:8]], dtype=np.float32)
            ims.append(cur[0])
            poses.append((xyz, wpqr))
        f_path.close()
        return ims, poses

    def __repr__(self):
        """
        docstring
        """
        fmt_str = f'AbsPoseDataset {self.dataset}\n'
        fmt_str += f'Number of samples: {self.__len__()}\n'
        fmt_str += f'Root location: {self.data_dir}\n'
        fmt_str += f'Pose txt: {self.pose_txt}\n'
        fmt_str += 'Transforms: {}\n'.format(self.transforms.__repr__().
                                             replace('\n', '\n    '))
        return fmt_str
