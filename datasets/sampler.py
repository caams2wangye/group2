import numpy as np
import torch
from torch.utils.data import Sampler

__all__ = ['CategoriesSampler']


# sample n_iter tasks: int iter loop, each task may have different cls in each sample procedure
class CategoriesSampler(Sampler):
    def __init__(self,
                 label,
                 batches,  # num of batches we will go to sample
                 task_per_batch,  # num of tasks in each batch,val,test=1
                 n_way,  # num of cls in each task
                 n_shot,  # num of samples in each cls used to get proto
                 n_query  # num of samples in each cls used to get result
                 ):
        self.task_per_batch = task_per_batch
        self.batches = batches
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        label = np.array(label)
        self.m_ind = []
        unique = np.sort(np.unique(label))  # delete rep ele, then rank by ascent order
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(
                ind)  # in training procedure,m_ind include 64 parts, each part has same label and the position of label

    def __len__(self):
        return self.batches

    def __iter__(self):
        for i in range(self.batches):
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]  # random generate 64 int, and get the first n_way 5
            for j in range(self.task_per_batch):
                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch_gallery.append(l[pos[:self.n_shot]])  # include each label's position
                    batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
            batch = torch.cat(batch_gallery + batch_query)
            yield batch
