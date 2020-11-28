# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/27 10:30
# Project Name: NER-CCF2020-HeapOverflow
# File        : Net.py
# --------------------------------------------------

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from keras.preprocessing.sequence import pad_sequences
from torchcrf import CRF


class Net(nn.Module):
    def __init__(self, model_name, args):
        super(Net, self).__init__()
        self.model_name = model_name
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained)
        self.pretrain_base = AutoModel.from_pretrained(self.args.pretrained).to(self.args.device)
        self.lstm = nn.LSTM(input_size=self.args.embed_dim, hidden_size=self.args.lstm_hidden,
                            num_layers=self.args.lstm_layers, batch_first=True,
                            bidirectional=self.args.lstm_directs == 2).to(self.args.device)
        
        self.emissions = torch.autograd.Variable(torch.randn(self.args.seq_len, self.args.batch_size, self.args.num_tags),
                                                 requires_grad=True)
        self.crf = CRF(num_tags=self.args.num_tags, batch_first=True).to(self.args.device)

    def forward(self, texts: list):
        batch_size = len(texts)
        input_ids = [self.tokenizer.encode(text, add_special_tokens=True, max_length=self.args.seq_len, truncation=True)
                     for text in texts]
        input_ids = pad_sequences(input_ids, maxlen=self.args.seq_len, dtype="long",
                                  value=0, truncating="post", padding="post")
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = (input_ids > 0).type(torch.long)

        input_ids, attention_masks = input_ids.to(self.args.device), attention_masks.to(self.args.device)
        embeddings, pools = self.pretrain_base(input_ids, attention_mask=attention_masks)

        h = torch.randn(self.args.lstm_layers * self.args.lstm_directs, batch_size, self.args.lstm_hidden).to(self.args.device)
        c = torch.randn(self.args.lstm_layers * self.args.lstm_directs, batch_size, self.args.lstm_hidden).to(self.args.device)

        # embeddings -> [batch, seq_len, embed_dim]
        lstm_out, (_, _) = self.lstm(embeddings, (h, c))
        # lstm_out -> [batch, seq_len, lstm_hidden * lstm_directs]

        pass


if __name__ == '__main__':
    import sys

    sys.path.append('../../')
    from Main import args

    net = Net('Baseline', args=args)
    net(["你是傻逼吗, 我去?", "很高兴见到你, 我的名字是小花, 什么时候出去喝一杯", "大家一起放屁好不好"])
