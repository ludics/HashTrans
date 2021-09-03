import os
import pdb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.io import loadmat
from dataloader.ImgDataset import ImgDataset, get_transform
import numpy as np
from config.Path import Path

PROTOCOL_ROOT = os.path.join(Path.data_root, 'StanfordCars/devkit')
TRAIN_IMG_ROOT = os.path.join(Path.data_root, 'StanfordCars/cars_train')
TEST_IMG_ROOT = os.path.join(Path.data_root, 'StanfordCars/cars_test')
VAL_IMG_ROOT = os.path.join(Path.data_root, 'StanfordCars/cars_test')


class StanfordCars(object):
    def __init__(self,
                 batch_size,
                 num_workers,
                 val_num,
                 resize: tuple = (224, 224)):
        self._num_classes = 196

        train_list_path = os.path.join(PROTOCOL_ROOT, 'cars_train_annos.mat')
        test_list_path = os.path.join(PROTOCOL_ROOT, 'cars_test_annos_withlabels.mat')
        val_list_path = os.path.join(PROTOCOL_ROOT, 'cars_test_annos_withlabels.mat')

        train_list_mat = loadmat(train_list_path)
        test_list_mat = loadmat(test_list_path)
        val_list_mat = loadmat(val_list_path)

        train_data = np.array([f.item() for f in train_list_mat['annotations']['fname'][0]])
        train_targets = np.array([f.item() - 1 for f in train_list_mat['annotations']['class'][0]])
        test_data = np.array([f.item() for f in test_list_mat['annotations']['fname'][0]])
        test_targets = np.array([f.item() - 1 for f in test_list_mat['annotations']['class'][0]])
        val_data = np.array([f.item() for f in val_list_mat['annotations']['fname'][0]])
        val_targets = np.array([f.item() - 1 for f in val_list_mat['annotations']['class'][0]])

        if val_num is not None:
            perm_index = np.random.permutation(len(val_data))
            val_index = perm_index[:val_num]
            val_data = val_data[val_index]
            val_targets = val_targets[val_index]

        # transform
        train_transform = get_transform(resize, 'train')
        test_transform = get_transform(resize, 'test')
        val_transform = get_transform(resize, 'val')

        # setup dataset
        self._train_set = ImgDataset(data=train_data,
                                     targets=train_targets,
                                     transform=train_transform,
                                     img_root=TRAIN_IMG_ROOT)
        self._test_set = ImgDataset(data=test_data,
                                    targets=test_targets,
                                    transform=test_transform,
                                    img_root=TEST_IMG_ROOT)
        self._val_set = ImgDataset(data=val_data,
                                   targets=val_targets,
                                   transform=val_transform,
                                   img_root=VAL_IMG_ROOT)
        self._database_set = self._train_set

        # setup dataloader
        self._train_loader = DataLoader(dataset=self._train_set,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)
        self._test_loader = DataLoader(dataset=self._test_set,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers)
        self._val_loader = DataLoader(dataset=self._val_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers)
        self._database_loader = DataLoader(dataset=self._database_set,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def val_set(self):
        return self._val_set

    @property
    def database_set(self):
        return self._database_set

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def val_loader(self):
        return self._val_loader

    @property
    def database_loader(self):
        return self._database_loader


if __name__ == '__main__':
    datahub = StanfordCars(val_num=100)
    print(len(datahub.train_set))
    print(len(datahub.test_set))
    print(len(datahub.val_set))
    print(len(datahub.database_set))
    for data, targets in datahub.train_loader:
        print(data.shape)
        print(targets.shape)
        exit(0)
