from ..config.DefaultConfig import DefaultConfig
import torch
import torch.utils.data as tud
import os
import json
import random
from ..utils import alloc_logger


class Iterator:
    def __init__(self, target, indices=[]):
        self.target = target
        self.index = -1
        self.indices = indices

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.index += 1
            if len(self.indices) > 0:
                return self.target[self.indices[self.index]]
            else:
                return self.target[self.index]
        except IndexError:
            raise StopIteration


class CCFDataset(tud.Dataset):
    def __init__(self, in_train=True, seq_len=DefaultConfig.HYPER.SEQ_LEN, label_dim=DefaultConfig.HYPER.LABEL_DIM):
        super().__init__()
        self.in_train = in_train
        self.seq_len = seq_len
        self.label_dim = label_dim

        self.data_path = DefaultConfig.PATHS.DATA_CCF_CLEANED
        if self.in_train:
            self.data = self.data_path + "/train/data"
            self.label = self.data_path + "/train/label"
        else:
            self.data = self.data_path + "/test/data"
            self.label = ""

        self.data_file_list = os.listdir(self.data)
        if self.label != "":
            self.label_file_list = os.listdir(self.label)
        else:
            self.label_file_list = list()

        self.file_num = len(self.data_file_list)

    def __len__(self):
        return self.file_num

    '''
    data_content: str
    label_content: torch.Tensor with dtype = torch.long
    '''
    # FIXME: force length to SEQ_LEN now use config,
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        with open(self.data + '/' + self.data_file_list[idx], encoding="UTF-8") as f_data:
            data_content = f_data.read()
        if self.in_train:
            with open(self.label + '/' + self.label_file_list[idx], encoding="utf8") as f_label:
                label_list = json.load(f_label)
                # label_list = [[(1 if count == sequence else 0) for count in range(self.label_dim)] for sequence in label_list]
                if len(label_list) >= self.seq_len:
                    label_list = label_list[:self.seq_len]
                else:
                    label_list += [0 for count in range(self.seq_len - len(label_list))]
                    # label_list += [[0] * DefaultConfig.HYPER.LABEL_DIM for count in range(self.seq_len - len(label_list))]
                label_content = torch.LongTensor(label_list)
                return data_content, label_content
        else:
            return data_content


class CCFDataloader:
    def __init__(self, in_train=True, batch_size=DefaultConfig.HYPER.BATCH_SIZE, thread_num=1):
        self.logger = alloc_logger("CCFDataloader.log", CCFDataloader)
        self.in_train = in_train
        self.dataset = CCFDataset(self.in_train)
        self.batch_size = batch_size
        self.thread_num = thread_num
        self.file_num = len(self.dataset)
        self.dataset_index = list(range(self.file_num))
        self.logger.log_message("file num:\t", self.file_num)

    def __len__(self):
        return self.file_num // self.batch_size + (0 if self.file_num % self.batch_size == 0 else 1)

    def __iter__(self):
        return Iterator(self)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        ret_size = self.batch_size
        if idx == len(self) - 1:
            ret_size = self.file_num % self.batch_size
        if self.in_train:
            data_contents = list()
            label_contents = 0
            for count in range(ret_size):
                data_content, label_content = self.dataset[self.dataset_index[idx * self.batch_size + count]]
                data_contents.append(data_content)
                label_content.unsqueeze_(0)
                if isinstance(label_contents, torch.Tensor):
                    label_contents = torch.cat((label_contents, label_content))
                else:
                    label_contents = label_content
            return data_contents, label_contents
        else:
            data_contents = list()
            for count in range(ret_size):
                data_content = self.dataset[self.dataset_index[idx * self.batch_size + count]]
                data_contents.append(data_content)
            return data_contents

    def shuffle(self):
        random.shuffle(self.dataset_index)


class KFold:
    def __init__(self, dataloader, k=10):
        self.k = k
        self.dataloader = dataloader
        self.dataloader_index = list(range(len(self.dataloader)))
        self.folds = list()
        fold_length = len(self.dataloader) // self.k + 1
        pre_index = - fold_length
        count = -1
        for index in self.dataloader_index:
            if index - pre_index == fold_length:
                count += 1
                pre_index = index
                if count == len(self.dataloader) % self.k:
                    fold_length -= 1
                self.folds.append(list())
            self.folds[len(self.folds)-1].append(index)
        self.fold_count = 0
        self.fold_train = [index for index in self.dataloader_index if index not in self.folds[self.fold_count]]
        self.fold_valid = self.folds[self.fold_count]

    def __len__(self):
        return self.k

    def next_fold(self):
        self.fold_count += 1
        self.fold_count %= self.k
        self.fold_train = [index for index in self.dataloader_index if index not in self.folds[self.fold_count]]
        self.fold_valid = self.folds[self.fold_count]

    def get_train(self):
        return Iterator(self.dataloader, self.fold_train)

    def get_valid(self):
        return Iterator(self.dataloader, self.fold_valid)

    def new_k_fold(self):
        self.dataloader.shuffle()
        self.fold_count = 0


if __name__ == "__main__":
    '''
    ccf_dataloader = CCFDataloader(in_train=True)
    for i, (data_contents, label_contents) in enumerate(ccf_dataloader):
        if i == 5:
            break
        print("=============BATCH %d=============" % i)
        print(data_contents)
        print(label_contents)
    '''
    ccf_dataloader = CCFDataloader(in_train=True)
    print(len(ccf_dataloader))
    k_fold = KFold(ccf_dataloader, 10)
    for fold_count in range(len(k_fold)):
        print("=============NEW FOLD============")
        count = 0
        print('--------TRAIN--------')
        for data_content, label_content in k_fold.get_train():
            '''
            print('---train_%d---' % count)
            print(len(data_content))
            print(label_content.shape)
            '''
            if count == 0:
                print('---train_%d---' % count)
                print(data_content)
                print(label_content)
            count += 1
        count = 0
        print('--------VALID--------')
        for data_content, label_content in k_fold.get_valid():
            '''
            print('---valid_%d---' % count)
            print(len(data_content))
            print(label_content.shape)
            '''
            if count == 0:
                print('---valid_%d---' % count)
                print(data_content)
                print(label_content)
            count += 1
        k_fold.next_fold()
