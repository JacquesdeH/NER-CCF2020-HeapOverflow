from .config.DefaultConfig import DefaultConfig as config
from .dataloader.dataloader import CCFDataloader
from .dataloader.dataloader import KFold
import torch
import torch.nn as nn
import os


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


def save_module(module: nn.Module):
    os.makedirs(config.PATHS.DATA_MODULE)
    torch.save(module.state_dict(), config.PATHS.DATA_MODULE + "/module.th")


def load_module(module: nn.Module):
    module.load_state_dict(torch.load(config.PATHS.DATA_MODULE + 'module.th'))


'''
return batch_size and learning_rate
data_content: list: batch_size
label_content: [batch_size, seq_len] 
'''


def train(n_time, k_fold):
    dataloader = CCFDataloader()
    module = TempModule()
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(module.parameters())
    #schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    k_fold = KFold(dataloader=dataloader, k=k_fold)
    loss_history = list()
    for time in range(n_time):
        total_loss = 0.
        for fold in range(k_fold):
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
        loss_history.append(total_loss)


'''
def train(k_fold=False):
    module = TempModule()
    dataloader = CCFDataloader()
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(module.parameters())
'''
