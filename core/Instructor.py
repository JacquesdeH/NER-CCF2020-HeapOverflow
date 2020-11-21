from .config.DefaultConfig import DefaultConfig as config
from .dataloader.dataloader import CCFDataloader
from .dataloader.dataloader import KFold
import torch
import torch.nn as nn


class TempModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0

    def forward(self, words):
        self.count += 1
        return torch.zeros((config.HYPER.BATCH_SIZE, config.HYPER.SEQ_LEN, config.HYPER.LABEL_DIM))


def get_loss_fn(reduce=None, size_average=None):
    return nn.CrossEntropyLoss(reduce=reduce, size_average=size_average)


def get_optimizer(params, lr=1e-3):
    return torch.optim.Adam(params=params, lr=lr)


'''
return batch_size and learning_rate
data_content: list: batch_size
label_content: [batch_size, seq_len] 
'''
def n_time_k_fold(n, k, dataloader):
    module = TempModule()
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(module.parameters())
    #schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    k_fold = KFold(dataloader=dataloader, k=k)
    for time in range(n):
        total_loss = 0.
        for fold in range(k):
            for data_content, label_content in k_fold.get_train():
                label_predict = module(dataloader)
                loss = loss_fn(label_predict, label_content)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for data_content, label_content in k_fold.get_valid():
                with torch.no_grad():
                    label_predict = module(dataloader)
                    loss = loss_fn(label_predict, label_content)
                    total_loss += loss.sum().item()

            k_fold.next_fold()
        k_fold.new_k_fold()
        print('total loss: %d' % total_loss)


def train(k_fold=False):
    module = TempModule()
    dataloader = CCFDataloader()
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(module.parameters())
