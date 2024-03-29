# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import os.path as osp
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from .dataset import CUB, Cub2011, Dogs, CarsDataset, Food101
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
from PIL import Image

def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
    dataset_gallery, _ = build_dataset(is_train=True, config=config, do_trans=False)
    dataset_gallery = dataset_train
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    # sampler_val = SubsetRandomSampler(indices)
    sampler_val = SequentialSampler(dataset_val)
    sampler_gallery = SequentialSampler(dataset_gallery)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_gallery = torch.utils.data.DataLoader(
        dataset_gallery, sampler=sampler_gallery,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, data_loader_gallery, mixup_fn


def build_dataset(is_train, config, do_trans=None):
    if do_trans == None:
        do_trans = is_train
    # TODO: should gallery do transform???
    transform = build_transform(do_trans, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'CUB_200_2011':
        dataset = CUB(config.DATA.DATA_PATH, is_train, transform=transform)
        nb_classes = 200
    elif config.DATA.DATASET == 'CUB_200_2011_ADSH':
        Cub2011.init(config.DATA.DATA_PATH)
        if is_train:
            dataset = Cub2011(config.DATA.DATA_PATH, 'train', transform=transform)
        else:
            dataset = Cub2011(config.DATA.DATA_PATH, 'query', transform=transform)
        nb_classes = 200
    elif config.DATA.DATASET == 'Dogs':
        dataset = Dogs(config.DATA.DATA_PATH, train=is_train, transform=transform)
        nb_classes = 120
    elif config.DATA.DATASET == "Cars":
        data_dir = config.DATA.DATA_PATH
        if is_train:
            dataset = CarsDataset(osp.join(data_dir,'devkit/cars_train_annos.mat'),
                            osp.join(data_dir,'cars_train'),
                            osp.join(data_dir,'devkit/cars_meta.mat'),
                            cleaned=None, transform=transform)
        else:
            dataset = CarsDataset(osp.join(data_dir,'devkit/cars_test_annos_withlabels.mat'),
                            osp.join(data_dir,'cars_test'),
                            osp.join(data_dir,'devkit/cars_meta.mat'),
                            cleaned=None, transform=transform)
        nb_classes = 196
    elif config.DATA.DATASET == "Food101":
        dataset = Food101(config.DATA.DATA_PATH, is_train, transform)
        nb_classes = 101
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def sample_dataloader(dataloader, config):
    """
    Sample data from dataloder.

    Args
        dataloader (torch.utils.data.DataLoader): Dataloader.
        num_samples (int): Number of samples.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        sample_index (int): Sample index.
        dataset(str): Dataset name.

    Returns
        sample_dataloader (torch.utils.data.DataLoader): Sample dataloader.
    """
    data = dataloader.dataset.data
    targets = dataloader.dataset.targets

    transform = build_transform(True, config)
    
    sample_index = torch.randperm(data.shape[0])[:config.HASH.NUM_SAMPLES]
    data = data[sample_index]
    targets = targets[sample_index]
    sample = wrap_data(data, targets, config, transform)

    return sample, sample_index


def wrap_data(data, targets, config, transform):
    """
    Wrap data into dataloader.

    Args
        data (np.ndarray): Data.
        targets (np.ndarray): Targets.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        dataloader (torch.utils.data.dataloader): Data loader.
    """
    class MyDataset(Dataset):
        def __init__(self, data, targets, root, transform):
            self.data = data
            self.targets = targets
            self.root = root
            self.transform = transform
            self.onehot_targets = self.targets

        def __getitem__(self, index):
            img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
            img = self.transform(img)
            return img, self.targets[index], index

        def __len__(self):
            return self.data.shape[0]

        def get_onehot_targets(self):
            """
            Return one-hot encoding targets.
            """
            return torch.from_numpy(self.onehot_targets).float()

    dataset_train = MyDataset(data, targets, config.DATA.DATA_PATH, transform)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    return data_loader_train
