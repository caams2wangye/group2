import argparse

import torch.nn.functional as F

from models.classification_heads import one_hot
from utils import *

# D:\Group2_Homework
"""
Batch Training Procedure:
for each episode, we sample 'batches' batches
for each batch, we have 'task_per_batch' tasks
for each task, we have n_way * (n_shot + n_query) samples and tasks in same batch share same label space

"""
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def set_parameters(parser):
    # main setting
    parser.add_argument('--network', type=str, default='Conv4',
                        help='feature extracting backbone')
    parser.add_argument('--head', type=str, default='cosin',
                        help='classification head')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        help='datasets')
    parser.add_argument('--aug', type=bool, default=False,
                        help='whether to use augmentation method')
    parser.add_argument('--workers', type=int, default=4,
                        help='num of thread to process image')
    #  task setting
    parser.add_argument('--n-way', type=int, default=5,
                        help='n_way')
    parser.add_argument('--train-shot', type=int, default=1,
                        help='n_shot')
    parser.add_argument('--val_shot', type=int, default=1,
                        help='n_shot for validation')
    parser.add_argument('--train_query', type=int, default=2,
                        help='n_query')
    parser.add_argument('--val_query', type=int, default=1,
                        help='n_query for validation')
    # training setting
    parser.add_argument('--batches', type=int, default=2,
                        help='number of batches used to train in each epoch')
    parser.add_argument('--task-per-batch', type=int, default=2,
                        help='number of tasks in each batch')
    parser.add_argument('--num-epoch', type=int, default=50,
                        help='number of training epochs')
    # validation setting
    parser.add_argument('--val-iter', type=int, default=1,
                        help='num of tasks used to validation')
    # model save setting
    parser.add_argument('--save-epoch', type=int, default=1,
                        help='frequency of model saving')
    parser.add_argument('--save-path', default='./experiments/exp_1',
                        help='save_path')
    # environment setting
    parser.add_argument('--gpu', default='0',
                        help='index of gpu will be used')
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = set_parameters(parser)
    opt = parser.parse_args()
    print(opt)

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, 'train_log.txt')
    log(log_file_path, str(vars(opt)))
    network, cls_head = get_model(opt)
    optimizer = torch.optim.SGD([{'params': network.parameters()},
                                 {'params': cls_head.parameters()}], lr=0.1, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    lambda_epoch = lambda e: 1.0 if e < 12 else (
        0.025 if e < 30 else 0.0032 if e < 45 else (0.0014 if e < 57 else 0.00052))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    max_val_acc = 0.0
    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.num_epoch + 1):
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
        log(log_file_path, 'Train Epoch:{}\tLearning Rate:{:.4f}'.format(epoch, epoch_learning_rate))
        _, _ = [x.train() for x in (network, cls_head)]

        train_accuracies = []
        train_losses = []

        loader_train = get_dataloader(opt, 'train')
        loader_val = get_dataloader(opt, 'val')
        for i, batch in enumerate(loader_train):
            data_support, labels_support, data_query, labels_query = data_split(batch,
                                                                                opt.n_way,
                                                                                opt.train_shot,
                                                                                opt.train_query)
            train_n_support = opt.n_way * opt.train_shot
            train_n_query = opt.n_way * opt.train_query

            emb_support = network(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.task_per_batch, train_n_support, -1)

            emb_query = network(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.task_per_batch, train_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.n_way, opt.train_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.n_way)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.n_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()

            acc = count_accuracy(logit_query.reshape(-1, opt.n_way), labels_query.reshape(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())
            if i % 100 == 0:
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch:{}\tTask:[{}/{}]\tLoss:{:.4f}\tAccuracy:{:.2f} % ({:.2f})%'.format(
                    epoch, i, len(loader_train), loss.item(), train_acc_avg, acc
                ))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        _, _ = [x.eval() for x in (network, cls_head)]
        val_accuracies = []
        val_losses = []
        for i, task_val in enumerate(loader_val):
            data_support, labels_support, data_query, labels_query = data_split(task_val,
                                                                                opt.n_way,
                                                                                opt.val_shot,
                                                                                opt.val_query)

            val_n_support = opt.n_way * opt.val_shot
            val_n_query = opt.n_way * opt.val_query

            emb_support = network(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, val_n_support, -1)

            emb_query = network(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, val_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.n_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.n_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.n_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_aci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_iter)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': network.state_dict(), 'head': cls_head.state_dict()},
                       os.path.join(opt.save_path, 'best_model.pth'))
        else:
            log(log_file_path, 'Validation Epoch:{}\t\t\tLoss:{:.4f}\tAccuracy:{:.2f} ± {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_aci95))

        torch.save({'embedding': network.state_dict(), 'head': cls_head.state_dict()}
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': network.state_dict(), 'head': cls_head.state_dict()},
                       os.path.join(opt.save_path, 'epoch{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))

if __name__ == '__main00__':
    parser = argparse.ArgumentParser()
    # main setting
    parser.add_argument('--network', type=str, default='Conv4',
                        help='feature extracting backbone')
    parser.add_argument('--head', type=str, default='ProtoNet',
                        help='classification head')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        help='datasets')
    parser.add_argument('--aug', type=bool, default=False,
                        help='whether to use augmentation method')
    parser.add_argument('--workers', type=int, default=2,
                        help='num of thread to process image')
    # training setting
    parser.add_argument('--num-epoch', type=int, default=8,
                        help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=8,
                        help='frequency of model saving')
    parser.add_argument('--train-iter', type=int, default=1,
                        help='num of tasks used to train')
    parser.add_argument('--val-iter', type=int, default=1,
                        help='num of tasks used to validation')
    parser.add_argument('--n-way', type=int, default=5,
                        help='n_way')
    parser.add_argument('--train-shot', type=int, default=5,
                        help='n_shot')
    parser.add_argument('--val_shot', type=int, default=5,
                        help='n_shot for validation')
    parser.add_argument('--train_query', type=int, default=1,
                        help='n_query')
    parser.add_argument('--val_query', type=int, default=1,
                        help='n_query for validation')
    # other setting
    parser.add_argument('--gpu', default='0',
                        help='index of gpu will be used')
    parser.add_argument('--save-path', default='./experiments/exp_1',
                        help='save_path')
    opt = parser.parse_args()
    print(opt)

    set_gpu(opt.gpu)
    check_dir('./experiments/')

    check_dir(opt.save_path)
    log_file_path = os.path.join(opt.save_path, 'train_log.txt')
    log(log_file_path, str(vars(opt)))

    # loader_train, loader_val = get_dataloader(opt)

    network, cls_head = get_model(opt)
    optimizer = torch.optim.SGD([{'params': network.parameters()},
                                 {'params': cls_head.parameters()}], lr=0.1, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    lambda_epoch = lambda e: 1.0 if e < 12 else (
        0.025 if e < 30 else 0.0032 if e < 45 else (0.0014 if e < 57 else 0.00052))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    # n-way-k-shot setting
    n_iter = opt.train_iter  # 10
    n_way = opt.n_way  # 5
    n_shot = opt.train_shot  # 5
    n_query = opt.train_query  # 1
    val_iter = opt.val_iter
    val_shot = opt.val_shot
    val_query = opt.val_query

    for epoch in range(1, opt.num_epoch + 1):

        lr_scheduler.step()

        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
        log(log_file_path, 'Train Epoch:{}\tLearning Rate:{:.4f}'.format(epoch, epoch_learning_rate))
        _, _ = [x.train() for x in (network, cls_head)]

        train_accuracies = []
        train_losses = []

        loader_train = get_dataloader(opt, 'train')
        loader_val = get_dataloader(opt, 'val')
        for i, task in enumerate(loader_train):
            # a simple task
            data_support, labels_support, data_query, labels_query = data_split(task, n_way, n_shot, n_query)

            train_n_support = n_way * n_shot
            train_n_query = n_way * n_query

            emb_support = network(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, train_n_support, -1)

            emb_query = network(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, train_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, n_way, n_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), n_way)

            log_prb = F.log_softmax(logit_query.reshape(-1, n_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()

            acc = count_accuracy(logit_query.reshape(-1, n_way), labels_query.reshape(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if i % 100 == 0:
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch:{}\tTask:[{}/{}]\tLoss:{:.4f}\tAccuracy:{:.2f} % ({:.2f})%'.format(
                    epoch, i, len(loader_train), loss.item(), train_acc_avg, acc
                ))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, _ = [x.eval() for x in (network, cls_head)]
        val_accuracies = []
        val_losses = []
        for i, task_val in enumerate(loader_val):
            data_support, labels_support, data_query, labels_query = data_split(task_val, n_way, val_shot, val_query)

            val_n_support = n_way * val_shot
            val_n_query = n_way * val_query

            emb_support = network(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, val_n_support, -1)

            emb_query = network(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, val_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, n_way, val_shot)

            loss = x_entropy(logit_query.reshape(-1, n_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, n_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_aci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(val_iter)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': network.state_dict(), 'head': cls_head.state_dict()},
                       os.path.join(opt.save_path, 'best_model.pth'))
        else:
            log(log_file_path, 'Validation Epoch:{}\t\t\tLoss:{:.4f}\tAccuracy:{:.2f} ± {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_aci95))

        torch.save({'embedding': network.state_dict(), 'head': cls_head.state_dict()}
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': network.state_dict(), 'head': cls_head.state_dict()},
                       os.path.join(opt.save_path, 'epoch{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
