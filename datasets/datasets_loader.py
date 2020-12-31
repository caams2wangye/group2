import os
import PIL.Image as Image
import numpy as np

__all__ = ['DatasetFolder']


class DatasetFolder(object):
    """
    if miniImageNet
        if train:
            root = 'D:/Group2_Homework/mini-imagenet/train'
            split_dir = 'D:/Group2_Homework/mini-imagenet'
            split_type = 'train'
        if val:
            root = 'D:/Group2_Homework/mini-imagenet/val'
            split_dir = 'D:/Group2_Homework/mini-imagenet'
            split_type = 'val'
        if test:
            root = 'D:/Group2_Homework/mini-imagenet/test'
            split_dir = 'D:/Group2_Homework/mini-imagenet'
            split_type = 'test'
    if Cifar-fs:
        if train:
            root = 'D:/Group2_Homework/cifar-fs/CIFAR-FS/cifar100/data'
            split_dir = 'D:/Group2_Homework/cifar-fs/CIFAR-FS/cifar100'
            split_type = 'train'
        if val:
            root = 'D:/Group2_Homework/cifar-fs/CIFAR-FS/cifar100/data'
            split_dir = 'D:/Group2_Homework/cifar-fs/CIFAR-FS/cifar100'
            split_type = 'val'
        if test:
            root = 'D:/Group2_Homework/cifar-fs/CIFAR-FS/cifar100/data'
            split_dir = 'D:/Group2_Homework/cifar-fs/CIFAR-FS/cifar100'
            split_type = 'test'
    """

    def __init__(self, root, split_dir, split_type, transform,
                 out_name=False):  # here root='D:/Group2_Homework/datasets'
        assert split_type in ['train', 'test', 'val', 'query', 'repr']  # open different files, e.g train
        split_file = split_dir + '/' + split_type + '.csv'   # ./split_dir/train.csv
        data = []
        ori_labels = []
        assert os.path.isfile(split_file)
        with open(split_file, 'r') as f:  # (name.jpg,label)
            next(f)  # skip first line
            for split in f.readlines():
                split = split.strip('\n').split(',')  # split = [name.jpg label]
                data.append(split[0])  # name.jpg
                ori_labels.append(split[1])  # label

        label_key = sorted(np.unique(np.array(ori_labels)))  # input ['a','a','b','b','c','c'] -> ['a','b','c'] output
        label_map = dict(zip(label_key, range(len(label_key))))  # input['a','b','c'] - > {'a': 0, 'b': 1, 'c': 2}output
        mapped_labels = [label_map[x] for x in ori_labels]  # mapped_labels = [0, 0, 1, 1, 2, 2]

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels
        self.out_name = out_name
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert os.path.isfile(self.root + '/' + self.data[index])
        img = Image.open(self.root + '/' + self.data[index]).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            return img, label, self.data[index]
        else:
            return img, label
