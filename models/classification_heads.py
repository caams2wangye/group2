import torch
import torch.nn as nn

"""
an example for classification head
protonet:
-------------------start line------------------
"""
__all__ = ['ClassificationHead', 'one_hot']


def computeGramMatrix(Matrix_A, Matrix_B):
    assert (Matrix_A.dim() == 3)
    assert (Matrix_B.dim() == 3)
    assert (Matrix_A.size(0) == Matrix_B.size(0) and Matrix_A.size(2) == Matrix_B.size(2))

    return torch.bmm(Matrix_A, Matrix_B.transpose(1, 2))


def one_hot(indices, depth):
    encode_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encode_indicies = encode_indicies.scatter_(1, index, 1)

    return encode_indicies


def PowerTransform(m_input, beta, epsilon):
    support_min = torch.min(m_input)
    support_max = torch.max(m_input)
    support_norm = torch.div(m_input - support_min, support_max - support_min)
    support_1 = torch.pow(support_norm + epsilon, beta)
    support_l2 = torch.norm(support_1, dim=2)
    a = torch.randn(1)
    pt = torch.div(support_1, support_l2.view(support_l2.size()+a.size()))
    return pt


# ProtoNetHead is a just a example which used Euclidean distance to measure similarity
# you can reference ProtoNetHead to design your own classification head
def ProtoNetHead(query, support, support_labels, n_way, n_shot, normarlize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and
    returns the classification score (=L2 distance to each class prototype) on the query set.

    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    # transpose label to make same class labels in identical counterpart
    lables_train_transposed = support_labels_one_hot.transpose(1, 2)  # (tasks_per_batch, n_way, n_support)
    # sum elements with same label
    prototypes = torch.bmm(lables_train_transposed, support)  # (tasks_per_batch, n_way, d)
    # get average
    prototypes = prototypes.div(lables_train_transposed.sum(dim=2, keepdims=True).expand_as(prototypes))
    # (A - B)**2 = A**2 + B**2 - 2*A*B
    AB = computeGramMatrix(query, prototypes)  # (tasks_per_batch, n_query, n_way)
    AA = (query * query).sum(dim=2, keepdim=True)  # (tasks_per_batch, n_query, 1)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits

    if normarlize:
        logits = logits / d

    return logits  # (task_per_batch, n_query, n_way) return similarity score


def CosineSimilarity(query, support, support_labels, n_way, n_shot, normarlize=False):
    """
        Parameters:
          query:  a (tasks_per_batch, n_query, d) Tensor.
          support:  a (tasks_per_batch, n_support, d) Tensor.
          support_labels: a (tasks_per_batch, n_support) Tensor.
          n_way: a scalar. Represents the number of classes in a few-shot classification task.
          n_shot: a scalar. Represents the number of support examples given per class.
          normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)

    # power transform
    epsilon = 1e-7
    beta = 0.5
    support = PowerTransform(support, beta, epsilon)
    query = PowerTransform(query, beta, epsilon)

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    # transpose label to make same class labels in identical counterpart
    lables_train_transposed = support_labels_one_hot.transpose(1, 2)  # (tasks_per_batch, n_way, n_support)
    # sum elements with same label
    prototypes = torch.bmm(lables_train_transposed, support)  # (tasks_per_batch, n_way, d)
    # get average
    prototypes = prototypes.div(lables_train_transposed.sum(dim=2, keepdims=True).expand_as(prototypes))
    AB = computeGramMatrix(query, prototypes)  # (tasks_per_batch, n_query, n_way)
    epsilon2 = 1e-8
    A_norm2 = torch.norm(prototypes, dim=2)
    A_norm2 = A_norm2.view(prototypes.size(0), prototypes.size(1), 1)
    B_norm2 = torch.norm(query, dim=2)
    B_norm2 = B_norm2.view(query.size(0), query.size(1), 1)
    logit1 = torch.div(AB, torch.bmm(B_norm2, A_norm2.transpose(1, 2)) + epsilon2)
    # logit1 = -logit1

    if normarlize:
        logit1 = logit1 / d

    return logit1


def MahalanobisDis(query, support, support_labels, n_way, n_shot, normarlize=True):
    """
        Parameters:
          query:  a (tasks_per_batch, n_query, d) Tensor.
          support:  a (tasks_per_batch, n_support, d) Tensor.
          support_labels: a (tasks_per_batch, n_support) Tensor.
          n_way: a scalar. Represents the number of classes in a few-shot classification task.
          n_shot: a scalar. Represents the number of support examples given per class.
          normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)

    logit2 = None
    return logit2


def WassersteinDis(query, support, support_labels, n_way, n_shot, normarlize=True):
    """
        Parameters:
          query:  a (tasks_per_batch, n_query, d) Tensor.
          support:  a (tasks_per_batch, n_support, d) Tensor.
          support_labels: a (tasks_per_batch, n_support) Tensor.
          n_way: a scalar. Represents the number of classes in a few-shot classification task.
          n_shot: a scalar. Represents the number of support examples given per class.
          normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)

    logit3 = None
    return logit3


class ClassificationHead(nn.Module):
    def __init__(self, base_learner='proto', enable_scale=True):
        super(ClassificationHead, self).__init__()
        if 'proto' in base_learner:
            self.head = ProtoNetHead
        if 'cosin' in base_learner:
            self.head = CosineSimilarity
        else:
            assert False

        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
        else:
            return self.head(query, support, support_labels, n_way, n_shot, **kwargs)


# if __name__ == '__main__':
#     support = torch.randn(3, 3, 4).cuda()
#     query = torch.randn(3, 3, 4).cuda()
#     n_way = 3
#     n_shot = 1
#     normarlize = True
#     support_labels = torch.LongTensor([[0, 1, 2], [1, 0, 2], [0, 2, 1]]).cuda()
#     logit = CosineSimilarity(query, support, support_labels, n_way, n_shot, normarlize).cuda()
#     print(logit.cpu())
