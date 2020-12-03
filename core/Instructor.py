from .config.DefaultConfig import DefaultConfig as config
from .dataloader.dataloader import CCFDataloader
from .dataloader.dataloader import KFold
import torch
import torch.nn as nn
import os
from .utils import alloc_logger
from core.model.Net import Net
import json
from Main import args


class TempModule(nn.Module):
    def __init__(self):
        super(TempModule, self).__init__()
        self.count = 0

    def forward(self, words):
        self.count += 1
        return torch.zeros((config.HYPER.BATCH_SIZE, config.HYPER.SEQ_LEN, config.HYPER.LABEL_DIM))


class Instructor:
    def __init__(self, model_name, args):
        self.model_name = model_name
        self.args = args
        self.model = Net(self.model_name, self.args)
        pass

    def get_loss_fn(self, reduce=None, size_average=None):
        return nn.CrossEntropyLoss(reduce=reduce, size_average=size_average)

    def get_optimizer(self, params, lr=1e-3):
        # torch.optim.AdamW
        return torch.optim.Adam(params=params, lr=lr)

    def get_scheduler(self, optimizer, rate):
        pass

    def save_module(self, module: nn.Module):
        os.makedirs(config.PATHS.DATA_MODULE)
        torch.save(module.state_dict(), config.PATHS.DATA_MODULE + "/module.th")

    def load_module(self, module: nn.Module):
        module.load_state_dict(torch.load(config.PATHS.DATA_MODULE + 'module.th'))

    '''
    return batch_size and learning_rate
    data_content: list: batch_size
    label_content: [batch_size, seq_len] 
    '''

    def train(self):
        n_time = self.args.n
        k_fold = self.args.k
        train_log = alloc_logger("train.log", "train")
        train_log.log_message("train at n_time: %d, k_fold: %d" % (n_time, k_fold))
        dataloader = CCFDataloader(args=self.args, in_train=True)
        loss_fn = self.get_loss_fn()
        optimizer = self.get_optimizer(self.model.parameters())
        # schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        k_fold = KFold(dataloader=dataloader, k=k_fold)
        loss_history = list()
        for time in range(n_time):
            total_loss = 0.
            for fold in range(len(k_fold)):
                for data_content, label_content in k_fold.get_train():
                    label_predict = self.model(dataloader)
                    loss = loss_fn(label_predict, label_content)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                for data_content, label_content in k_fold.get_valid():
                    with torch.no_grad():
                        label_predict = self.model(dataloader)
                        loss = loss_fn(label_predict, label_content)
                        total_loss += loss.sum().item()

                k_fold.next_fold()
            k_fold.new_k_fold()
            train_log.log_message('total loss: %d' % total_loss)
            loss_history.append(total_loss)

    def test(self, args, module):
        with torch.no_grad():
            dataloader = CCFDataloader(args=args, in_train=False)
            file_count = 0
            for data_contents in dataloader:
                predicts = module(data_contents)
                predicts = predicts.cpu()   #[batch_size, seq_len, label_dim]
                for predict_count in range(predicts.shape[0]):
                    predict = predicts[predict_count]
                    result = list()
                    for col_num in range(predict.shape[0]):
                        col = predict[col_num]
                        result.append(col.argmax().item())
                    with open(os.path.join(config.PATHS.DATA_CCF_CLEANED,'test/label/%d.json' % (file_count + predict_count)),'w') as fw:
                        json.dump(result, fw)
                file_count += predicts.shape[0]


if __name__ == '__main__':
    insructor = Instructor("a", args=args)
    insructor.test(args=args, module=TempModule())