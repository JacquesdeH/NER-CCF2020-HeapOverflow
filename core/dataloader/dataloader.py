from ..config.DefaultConfig import DefaultConfig
import torch
import torch.utils.data as tud
import os
import json
from ..utils import alloc_logger


class CCFDataset(tud.Dataset):
    def __init__(self, in_train=True):
        super().__init__()
        self.in_train = in_train

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
    def __getitem__(self, idx):
        if idx > self.file_num:
            return
        with open(self.data + '/' + self.data_file_list[idx], encoding="UTF-8") as f_data:
            data_content = f_data.read()
        if self.in_train:
            with open(self.label + '/' + self.label_file_list[idx], encoding="utf8") as f_label:
                label_content = torch.LongTensor(json.load(f_label))
                return data_content,label_content
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
        self.logger.log_message("file num:\t", self.file_num)

    def __len__(self):
        return self.file_num // self.batch_size + (0 if self.file_num % self.batch_size == 0 else 1)

    def __getitem__(self, idx):
        ret_size = self.batch_size
        if idx == len(self) - 1:
            ret_size = self.file_num % self.batch_size
        if self.in_train:
            data_contents = list()
            label_contents = 0
            for count in range(ret_size):
                data_content, label_content = self.dataset[idx * self.batch_size + count]
                data_contents.append(data_content)
                if isinstance(label_contents,torch.Tensor):
                    label_contents = torch.stack((label_contents, label_content))
                else:
                    label_contents = label_content
            return data_contents, label_contents
        else:
            data_contents = list()
            for count in range(ret_size):
                data_content = self.dataset[idx * self.batch_size + count]
                data_contents.append(data_content)
            return data_contents


if __name__ == "__main__":
    ccf_dataloader = CCFDataloader(in_train=True)
    for i, data_contents, label_contents in enumerate(ccf_dataloader):
        if i == 5:
            break
        print("=============BATCH %d=============" % i)
        print(data_contents)
        print(label_contents) 