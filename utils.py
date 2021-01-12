import os
import time
import torch
import numpy as np
from models.ResNet import resnet18
from models.WRN import WideResNet
from models.Conv4 import conv_4
from models.ResNet12_embedding import resnet12
from models.classification_heads import ClassificationHead
from datasets import transform, sampler, datasets_loader


# choose gpus you want to use
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu', x)


# create folder used to save result
def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def count_accuracy(logits, label):
    pre = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pre.eq(label).float().mean()
    return accuracy


# calculate consumption time
class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(x / 60)
        return '{}s'.format(x)


# create a log to write details
def log(log_file_path, string):
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)


# input category[10, 20, 1, 6] -> output[2, 3, 0, 1] labels
def cls2label(class_type):
    class_type = class_type.cpu()
    class_type = np.array(class_type)

    class_type_key = sorted(np.unique(class_type))
    class_type_map = dict(zip(class_type_key, range(len(class_type_key))))
    label = [class_type_map[x] for x in class_type]
    label = np.array(label)
    label = torch.from_numpy(label)
    label = label.long().cuda()
    return label


# get the feature extracting backbone and classification head
def get_model(options):
    if options.network == 'Conv4':
        network = conv_4().cuda()
    elif options.network == 'ResNet12':
        network = resnet12().cuda()
    elif options.network == 'ResNet18':
        network = resnet18().cuda()
    elif options.network == 'WRN':
        network = WideResNet(5, True).cuda()
    else:
        print("Network type is nor correct!")
        assert False

    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='proto').cuda()
    if options.head == 'cosin':
        cls_head = ClassificationHead(base_learner='cosin').cuda()
    else:
        print("cls_head is not correct!")
        assert False

    return network, cls_head


# get FSL setting which will used to sample from original data
def get_sample_paras(option, state):
    if state == 'train':
        sample_info = [option.batches,
                       option.task_per_batch,
                       option.n_way,
                       option.train_shot,
                       option.train_query]
    if state == 'val':
        sample_info = [option.val_iter,
                       1,
                       option.n_way,
                       option.val_shot,
                       option.val_query]
    if state == 'test':
        sample_info = [option.test_iter,
                       1,
                       option.n_way,
                       option.test_shot,
                       option.test_query]
    return sample_info


# get sampled result
def get_dataloader(option, state):
    if option.aug:
        transform_ = transform.with_augment(84, disable_random_resize=False)
    else:
        transform_ = transform.without_augment(84, enlarge=False)

    sample_info = get_sample_paras(option, state)

    if option.dataset == 'miniImageNet':
        split_dir = './datasets/mini-imagenet'
        pic_root = split_dir + '/' + state
        split_type = state
        dataset = datasets_loader.DatasetFolder(pic_root, split_dir, split_type, transform_,
                                                out_name=False)
        sampler_ = sampler.CategoriesSampler(dataset.labels, *sample_info)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler_,
                                             num_workers=option.workers, pin_memory=True)

    elif option.dataset == 'Cifar-FS':
        root_ = './datasets/cifar-fs/CIFAR-FS/cifar100/data'
        split_dir_ = './datasets/cifar-fs/CIFAR-FS/cifar100'
        split_type = state
        dataset = datasets_loader.DatasetFolder(root_, split_dir_, split_type, transform_,
                                                out_name=False)
        sampler_ = sampler.CategoriesSampler(dataset.labels, *sample_info)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler_,
                                             num_workers=option.workers, pin_memory=True)

    else:
        print("incorrect dataset name!")
        assert False

    return loader


# data:[n_way*n_shot*task_per_batch]+[n_way*n_query*task_per_batch]
# split original data into support data,support label,query data, query label
def data_split(original_data, n_way, n_shot, n_query):  # batch = (n_way * n_shot, 3, 84, 84)  equal to task
    # extract data and label from original data
    data, labels = [x.cuda() for x in original_data]
    total_samples = data.size(0)
    sample_in_each_task = n_way * (n_shot + n_query)
    task_per_batch = int(total_samples / sample_in_each_task)
    labels = cls2label(labels)

    n_support = n_way * n_shot * task_per_batch
    n_query = n_way * n_query * task_per_batch

    data_support_ = data[:n_support]
    labels_support_ = labels[:n_support]
    data_query_ = data[n_support:n_support + n_query]
    labels_query_ = labels[n_support:n_support + n_query]

    return data_support_, labels_support_, data_query_, labels_query_
