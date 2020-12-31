import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='Conv4',
                        help='feature extracting backbone')
    parser.add_argument('--head', type=str, default='ProtoNet',
                        help='classification head')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        help='datasets')
    parser.add_argument('--load',
                        default='./experiments/exp_1/best_model.pth',
                        help='the model path')
    parser.add_argument('--aug', type=bool, default=False,
                        help='whether to use augmentation method')
    parser.add_argument('--workers', type=int, default=2,
                        help='num of thread to process image')
    parser.add_argument('--test_iter', type=int, default=1,
                        help='num of tasks used to evaluate model')
    parser.add_argument('--n-way', type=int, default=5,
                        help='num of classes in task setting')
    parser.add_argument('--test-shot', type=int, default=1,
                        help='num of examples will be sampled in each cls')
    parser.add_argument('--test-query', type=int, default=1,
                        help='num of samples used to test in each cls')
    parser.add_argument('--save-path', default='./experiments/exp_1',
                        help='save_path')
    parser.add_argument('--gpu', default='0')

    opt = parser.parse_args()

    log_file_path = os.path.join(opt.save_path, 'test_log.txt')
    log(log_file_path, str(vars(opt)))

    network, cls_head = get_model(opt)
    saved_model = torch.load(opt.load)
    network.load_state_dict(saved_model['embedding'])
    network.eval()
    cls_head.load_state_dict(saved_model['head'])
    cls_head.eval()

    test_accuracies = []

    n_way = opt.n_way
    k_shot = opt.test_shot
    n_query = opt.test_query
    loader_test = get_dataloader(opt, 'test')
    for i, task in enumerate(loader_test):
        data_support, labels_support, data_query, labels_query = data_split(task, n_way, k_shot, n_query)

        test_n_support = n_way * k_shot
        test_n_query = n_way * n_query

        emb_support = network(data_support.reshape([-1] + list(data_support.shape[-3:])))
        emb_support = emb_support.reshape(1, test_n_support, -1)

        emb_query = network(data_query.reshape([-1] + list(data_query.shape[-3:])))
        emb_query = emb_query.reshape(1, test_n_query, -1)

        logit_query = cls_head(emb_query, emb_support, labels_support, n_way, k_shot)

        acc = count_accuracy(logit_query.reshape(-1, n_way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())

        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)

        if i % 10 == 0:
            print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                  .format(i, opt.test_iter, avg, ci95, acc))


