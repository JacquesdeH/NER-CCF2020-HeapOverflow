# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/18 20:21
# Project Name: NER-CCF2020-HeapOverflow
# File        : bert_demo.py
# --------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModel

if __name__ == '__main__':
    _pretrained_model = 'bert-base-chinese'
    _special_tokens = {"unk_token": "[UNK]",
                       "sep_token": "[SEP]",
                       "pad_token": "[PAD]",
                       "cls_token": "[CLS]",
                       "mask_token": "[MASK]"
                       }

    tokenizer = AutoTokenizer.from_pretrained(_pretrained_model)
    # cnt_added_tokens = tokenizer.add_special_tokens(_special_tokens)
    model = AutoModel.from_pretrained(_pretrained_model)
    # model.resize_token_embeddings(len(tokenizer))

    # input_ = "[CLS]我操你是傻逼吗,朋友?[SEP]"
    input_ = "我操你是傻逼吗?朋友?"
    tokens = tokenizer.tokenize(input_)
    # indexs = tokenizer.convert_tokens_to_ids(tokens)
    indexs = tokenizer.encode(tokens)
    tmp = tokenizer.convert_ids_to_tokens(indexs)
    indexs = torch.tensor(indexs, dtype=torch.long).reshape(1, -1)

    output = model.forward(indexs)


