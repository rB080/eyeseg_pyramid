import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader


def loader(img_path, map_path, mask_threshold=0.7, size=(256, 256)):
    img = cv2.imread(img_path)
    map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size, cv2.INTER_AREA)
    map = cv2.resize(map, size, cv2.INTER_AREA)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    map = 1 - np.array(map, np.float32) / 255.0
    map = map[np.newaxis, :, :]
    map[map >= mask_threshold] = 1.0
    map[map < mask_threshold] = 0.0
    return img, map


def read_dataset(root_path, split="train"):
    images = []
    maps = []

    img_root = os.path.join(root_path, split, "images")
    map_root = os.path.join(root_path, split, "gts")

    for image_name in sorted(os.listdir(map_root)):
        img_path = os.path.join(img_root, image_name)
        map_path = os.path.join(map_root, image_name[:-4]+".bmp")
        if os.path.exists(img_path) and os.path.exists(map_path):
            images.append(img_path)
            maps.append(map_path)
    return images, maps


class Segmentation_Dataset(Dataset):

    def __init__(self, root_path, split="train"):
        self.root = root_path
        self.images, self.maps = read_dataset(self.root, split)
        print('num img = ', len(self.images))
        print('num map = ', len(self.maps))

    def add_split(self, split):
        new_images, new_maps = read_dataset(self.root, split)
        self.images += new_images
        self.maps += new_maps

    def __getitem__(self, index):
        img, map = loader(self.images[index], self.maps[index])
        img = torch.tensor(img, dtype=torch.float32)
        map = torch.tensor(map, dtype=torch.float32)
        pack = {"img": img, "map": map, "name": self.images[index]}
        return pack

    def __len__(self):
        assert len(self.images) == len(
            self.maps), 'The number of img must be equal to map'
        return len(self.images)


def get_loader(args, split="train", shuffle=True, split_ratio=None):
    assert split in ["train", "test", "val", "all"], "Invalid Split Specified!"

    if split is not "all":
        dataset = Segmentation_Dataset(args.data_root, split=split)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=shuffle, num_workers=args.num_workers)
        return dataset, loader

    else:
        dataset = Segmentation_Dataset(args.data_root, split="train")
        dataset.add_split("val")
        dataset.add_split("test")
        if split_ratio is None:
            loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=shuffle, num_workers=args.num_workers)
            return dataset, loader
        else:
            sp = len(dataset) * \
                split_ratio[0] // (split_ratio[0] + split_ratio[1])
            second_ds = dataset
            dataset.images = dataset.images[:sp]
            dataset.maps = dataset.maps[:sp]
            second_ds.images = second_ds.images[sp:]
            second_ds.maps = second_ds.maps[sp:]
            L1 = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=shuffle, num_workers=args.num_workers)
            L2 = DataLoader(second_ds, batch_size=args.batch_size,
                            shuffle=shuffle, num_workers=args.num_workers)
            return dataset, second_ds, L1, L2
