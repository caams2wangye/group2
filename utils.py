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


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu', x)


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def count_accuracy(logits, label):
    pre = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pre.eq(label).float().mean()
    return accuracy


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
        cls_head = ClassificationHead(base_learner='proto').cuda()
    elif options.head == 'cls_head1':
        cls_head = ClassificationHead(base_learner='cls_head1').cuda()
    elif options.head == 'cls_head2':
        cls_head = ClassificationHead(base_learner='cls_head2').cuda()
    elif options.head == 'cls_head3':
        cls_head = ClassificationHead(base_learner='cls_head3').cuda()
    elif options.head == 'fusion_head':
        cls_head = ClassificationHead(base_learner='fusion_head').cuda()
    else:
        print("cls_head is not correct!")
        assert False

    return network, cls_head


def get_dataloader(option, state):
    if option.aug:
        transform_ = transform.with_augment(84, disable_random_resize=False)
    else:
        transform_ = transform.without_augment(84, enlarge=False)
    if option.dataset == 'miniImageNet':
        split_dir = './datasets/mini-imagenet'
        if state == 'train':
            sample_info_train = [option.train_iter, option.n_way, option.train_shot, option.train_query]
            root_train = './datasets/mini-imagenet/train'
            split_type_train = 'train'
            dataset_train = datasets_loader.DatasetFolder(root_train, split_dir, split_type_train, transform_,
                                                          out_name=False)
            sampler_train = sampler.CategoriesSampler(dataset_train.labels, *sample_info_train)
            loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler_train,
                                                 num_workers=option.workers, pin_memory=True)
        if state == 'val':
            sample_info_val = [option.val_iter, option.n_way, option.val_shot, option.val_query]
            root_val = './datasets/mini-imagenet/val'
            split_type_val = 'val'
            dataset_val = datasets_loader.DatasetFolder(root_val, split_dir, split_type_val, transform_,
                                                        out_name=False)
            sampler_val = sampler.CategoriesSampler(dataset_val.labels, *sample_info_val)
            loader = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val,
                                                 num_workers=option.workers, pin_memory=True)

        if state == 'test':
            sample_info_test = [option.test_iter, option.n_way, option.test_shot, option.test_query]
            root_test = './datasets/mini-imagenet/test'
            split_type_test = 'test'
            dataset_test = datasets_loader.DatasetFolder(root_test, split_dir, split_type_test, transform_,
                                                         out_name=False)
            sampler_test = sampler.CategoriesSampler(dataset_test.labels, *sample_info_test)
            loader = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test,
                                                 num_workers=option.workers, pin_memory=True)

    elif option.dataset == 'Cifar-FS':
        root_ = './datasets/cifar-fs/CIFAR-FS/cifar100/data'
        split_dir_ = './datasets/cifar-fs/CIFAR-FS/cifar100'
        if state == 'train':
            sample_info_train = [option.train_iter, option.n_way, option.train_shot, option.train_query]
            split_type_train = 'train'
            dataset_train = datasets_loader.DatasetFolder(root_, split_dir_, split_type_train, transform_,
                                                          out_name=False)
            sampler_train = sampler.CategoriesSampler(dataset_train.labels, *sample_info_train)
            loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler_train,
                                                 num_workers=option.workers, pin_memory=True)

        if state == 'val':
            split_type_val = 'val'
            sample_info_val = [option.val_iter, option.n_way, option.val_shot, option.val_query]
            dataset_val = datasets_loader.DatasetFolder(root_, split_dir_, split_type_val, transform_,
                                                        out_name=False)

            sampler_val = sampler.CategoriesSampler(dataset_val.labels, *sample_info_val)
            loader = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val,
                                                 num_workers=option.workers, pin_memory=True)

        if state == 'test':
            split_type_test = 'test'
            sample_info_test = [option.test_iter, option.n_way, option.test_shot, option.test_query]
            dataset_test = datasets_loader.DatasetFolder(root_, split_dir_, split_type_test, transform_,
                                                         out_name=False)
            sampler_test = sampler.CategoriesSampler(dataset_test.labels, *sample_info_test)
            loader = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test,
                                                 num_workers=option.workers, pin_memory=True)

    else:
        print("incorrect dataset name!")
        assert False

    return loader


def get_batch_dataloader(option, state):
    if option.aug:
        transform_ = transform.with_augment(84, disable_random_resize=False)
    else:
        transform_ = transform.without_augment(84, enlarge=False)
    if option.dataset == 'miniImageNet':
        split_dir = './datasets/mini-imagenet'
        if state == 'train':
            sample_info_train = [option.batches,
                                 option.task_per_batch,
                                 option.n_way,
                                 option.train_shot,
                                 option.train_query]
            root_train = './datasets/mini-imagenet/train'
            split_type_train = 'train'
            dataset_train = datasets_loader.DatasetFolder(root_train, split_dir, split_type_train, transform_,
                                                          out_name=False)
            sampler_train = sampler.BatchCategoriesSampler(dataset_train.labels, *sample_info_train)
            loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler_train,
                                                 num_workers=option.workers, pin_memory=True)
        if state == 'val':
            sample_info_val = [option.batches,
                               option.task_per_batch,
                               option.n_way,
                               option.val_shot,
                               option.val_query]
            root_val = './datasets/mini-imagenet/val'
            split_type_val = 'val'
            dataset_val = datasets_loader.DatasetFolder(root_val, split_dir, split_type_val, transform_,
                                                        out_name=False)
            sampler_val = sampler.BatchCategoriesSampler(dataset_val.labels, *sample_info_val)
            loader = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val,
                                                 num_workers=option.workers, pin_memory=True)

        if state == 'test':
            sample_info_test = [option.batches,
                                option.task_per_batch,
                                option.n_way,
                                option.test_shot,
                                option.test_query]
            root_test = './datasets/mini-imagenet/test'
            split_type_test = 'test'
            dataset_test = datasets_loader.DatasetFolder(root_test, split_dir, split_type_test, transform_,
                                                         out_name=False)
            sampler_test = sampler.BatchCategoriesSampler(dataset_test.labels, *sample_info_test)
            loader = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test,
                                                 num_workers=option.workers, pin_memory=True)

    elif option.dataset == 'Cifar-FS':
        root_ = './datasets/cifar-fs/CIFAR-FS/cifar100/data'
        split_dir_ = './datasets/cifar-fs/CIFAR-FS/cifar100'
        if state == 'train':
            sample_info_train = [option.batches,
                                 option.task_per_batch,
                                 option.n_way,
                                 option.train_shot,
                                 option.train_query]
            split_type_train = 'train'
            dataset_train = datasets_loader.DatasetFolder(root_, split_dir_, split_type_train, transform_,
                                                          out_name=False)
            sampler_train = sampler.BatchCategoriesSampler(dataset_train.labels, *sample_info_train)
            loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler_train,
                                                 num_workers=option.workers, pin_memory=True)

        if state == 'val':
            split_type_val = 'val'
            sample_info_val = [option.batches,
                               option.task_per_batch,
                               option.n_way,
                               option.val_shot,
                               option.val_query]
            dataset_val = datasets_loader.DatasetFolder(root_, split_dir_, split_type_val, transform_,
                                                        out_name=False)

            sampler_val = sampler.BatchCategoriesSampler(dataset_val.labels, *sample_info_val)
            loader = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val,
                                                 num_workers=option.workers, pin_memory=True)

        if state == 'test':
            split_type_test = 'test'
            sample_info_test = [option.batches,
                                option.task_per_batch,
                                option.n_way,
                                option.test_shot,
                                option.test_query]
            dataset_test = datasets_loader.DatasetFolder(root_, split_dir_, split_type_test, transform_,
                                                         out_name=False)
            sampler_test = sampler.BatchCategoriesSampler(dataset_test.labels, *sample_info_test)
            loader = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test,
                                                 num_workers=option.workers, pin_memory=True)

    else:
        print("incorrect dataset name!")
        assert False

    return loader


def data_split(original_data, n_way, n_shot, n_query):  # batch = (n_way * n_shot, 3, 84, 84)  equal to task
    # extract data and label from original data
    data, labels = [x.cuda() for x in original_data]
    total_samples = data.size(0)
    if total_samples == n_way:
        task_per_batch = 1
    else:
        sample_in_each_task = n_way * (n_shot + n_query)
        task_per_batch = int(total_samples / sample_in_each_task)

    labels = cls2label(labels)
    # initialize output
    data_support_ = torch.Tensor().cuda()
    labels_support_ = torch.LongTensor().cuda()
    data_query_ = torch.Tensor().cuda()
    labels_query_ = torch.LongTensor().cuda()
    for task_index in range(task_per_batch):
        for cls_index in range(n_way):
            cls = cls_index + task_index * n_way
            support_temp = data[(cls * (n_query + n_shot)):(cls * (n_query + n_shot) + n_shot)]
            support_label_temp = labels[(cls * (n_query + n_shot)):(cls * (n_query + n_shot) + n_shot)]

            query_temp = data[(cls * (n_query + n_shot) + n_shot):(cls * (n_query + n_shot) + n_shot + n_query)]
            query_label_temp = labels[(cls * (n_query + n_shot) + n_shot):(cls * (n_query + n_shot) + n_shot + n_query)]

            data_support_ = torch.cat([data_support_, support_temp], 0)
            labels_support_ = torch.cat([labels_support_, support_label_temp], 0)
            data_query_ = torch.cat([data_query_, query_temp], 0)
            labels_query_ = torch.cat([labels_query_, query_label_temp], 0)

    return data_support_, labels_support_, data_query_, labels_query_


# """
# batches:
#     loader_train:
#             data:
#                 (n_shot, image.size())
#             labels:
#                 (n_query, image.size())
#     loader_train:
#
# """
#
#
# def get_batch_dataloader(option):
#     sample_info_train = [option.train_iter, option.n_way, option.train_shot, option.train_query]
#     sample_info_val = [option.val_iter, option.n_way, option.val_shot, option.val_query]
#     batch_loader_train = []
#     batch_loader_val = []
#     if option.aug:
#         transform_ = transform.with_augment(84, disable_random_resize=False)
#     else:
#         transform_ = transform.without_augment(84, enlarge=False)
#     if option.dataset == 'miniImageNet':
#         root_train = './datasets/mini-imagenet/train'
#         split_dir = './datasets/mini-imagenet'
#         split_type_train = 'train'
#         root_val = './datasets/mini-imagenet/val'
#         split_type_val = 'val'
#         dataset_train = datasets_loader.DatasetFolder(root_train, split_dir, split_type_train, transform_,
#                                                       out_name=False)
#         dataset_val = datasets_loader.DatasetFolder(root_val, split_dir, split_type_val, transform_,
#                                                     out_name=False)
#         for batch in range(option.episodes_per_batch):
#             sampler_train = sampler.CategoriesSampler(dataset_train.labels, *sample_info_train)
#             sampler_val = sampler.CategoriesSampler(dataset_val.labels, *sample_info_val)
#             loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler_train,
#                                                        num_workers=option.workers, pin_memory=True)
#             loader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val,
#                                                      num_workers=option.workers, pin_memory=True)
#             batch_loader_train.append(loader_train)
#             batch_loader_val.append(loader_val)
#     elif option.dataset == 'Cifar-FS':
#         root_ = './datasets/cifar-fs/CIFAR-FS/cifar100/data'
#         split_dir_ = './datasets/cifar-fs/CIFAR-FS/cifar100'
#         split_type_train = 'train'
#         split_type_val = 'val'
#         dataset_train = datasets_loader.DatasetFolder(root_, split_dir_, split_type_train, transform_,
#                                                       out_name=False)
#         dataset_val = datasets_loader.DatasetFolder(root_, split_dir_, split_type_val, transform_,
#                                                     out_name=False)
#         for batch in range(option.episodes_per_batch):
#             sampler_train = sampler.CategoriesSampler(dataset_train.labels, *sample_info_train)
#             sampler_val = sampler.CategoriesSampler(dataset_val.labels, *sample_info_val)
#             loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler_train,
#                                                        num_workers=option.workers, pin_memory=True)
#             loader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val,
#                                                      num_workers=option.workers, pin_memory=True)
#             batch_loader_train.append(loader_train)
#             batch_loader_val.append(loader_val)
#
#     else:
#         print("incorrect dataset name!")
#         assert False
#
#     return batch_loader_train, batch_loader_val
