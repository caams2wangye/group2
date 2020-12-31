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


"""
below are three prototype function you need to code:
1.classifition_head1,classifition_head2 and classifition_head3 are single classifier 
2.FusionClsHead is used to fusion results of top three classification head
3.The meaning of input paras in each function you can reference 'ProtoNetHead' above
if you want add trainable parameters, maybe you can use torch.nn.Parameter(torch.Tensor())
"""
def classifition_head1(query, support, support_labels, n_way, n_shot, normarlize=True):
    logit1 = None
    return logit1


def classifition_head2(query, support, support_labels, n_way, n_shot, normarlize=True):
    logit2 = None
    return logit2


def classifition_head3(query, support, support_labels, n_way, n_shot, normarlize=True):
    logit3 = None
    return logit3


def FusionClsHead(query, support, support_labels, n_way, n_shot, normarlize=True):
    logit_fusion = None
    return logit_fusion



class ClassificationHead(nn.Module):
    def __init__(self, base_learner='proto', enable_scale=True):
        super(ClassificationHead, self).__init__()
        if 'proto' in base_learner:
            self.head = ProtoNetHead
        if 'cls_head1' in base_learner:
            self.head = classifition_head1
        if 'cls_head2' in base_learner:
            self.head = classifition_head2
        if 'cls_head3' in base_learner:
            self.head = classifition_head3
        if 'fusion_head' in base_learner:
            self.head = FusionClsHead
        else:
            assert False

        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
        else:
            return self.head(query, support, support_labels, n_way, n_shot, **kwargs)


