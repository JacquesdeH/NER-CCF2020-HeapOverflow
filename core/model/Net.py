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
from model import CRF


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
        if (self.args.lstm_directs):
            self.lstm_directs = 2
        else:
            self.lstm_directs = 1
            
        self.fc1 = nn.Sequential(nn.Linear(in_features=self.lstm_directs * self.args.lstm_hidden, 
                                           out_features=self.lstm_directs * self.args.lstm_hidden),
                                 # nn.BatchNorm
                                 nn.ReLU(),
                                 nn.Linear(in_features=self.lstm_directs * self.args.lstm_hidden, 
                                           out_features=self.lstm_directs * self.args.lstm_hidden),
                                 # nn.BatchNorm
                                 nn.ReLU(),
                                 nn.Linear(in_features=self.lstm_directs * self.args.lstm_hidden, out_features=self.args.lstm_hidden),
                                 nn.Sigmoid()).to(self.args.device)
                                 
        self.fc2 = nn.Sequential(nn.Linear(in_features=self.args.lstm_hidden, out_features=self.args.lstm_hidden),
                                 # nn.BatchNorm
                                 nn.ReLU(),
                                 nn.Linear(in_features=self.args.lstm_hidden, out_features=self.args.lstm_hidden),
                                 # nn.BatchNorm
                                 nn.ReLU(),
                                 nn.Linear(in_features=self.args.lstm_hidden, out_features=self.args.num_tags + 2),
                                 nn.Sigmoid()).to(self.args.device)
        self.dropout = nn.Dropout(0.4).to(self.args.device)
        self.crf = CRF(num_tags=self.args.num_tags, batch_first=True).to(self.args.device)

    
    def get_output_score(self, texts: list):
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
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_directs * self.args.lstm_hidden)
        fc1_out = self.fc1(lstm_out)
        fc2_out = self.fc2(self.dropout(fc1_out))
        # fc2_out -> [batch * seq_len, num_tags + 2]
        
        lstm_emissions = fc2_out.contiguous().view(batch_size, self.arfs.seq_len, -1)
        # lstm_emissions -> [batch, seq_len, num_tags + 2]
        return lstm_emissions
    
    
    def forward(self, texts: list):
        lstm_emissions = self.get_output_score(texts)
        tag_seq = self.crf.decode(emissions=lstm_emissions, mask=attention_masks)
        return tag_seq
    
    def neg_log_likelihood_loss(self, texts, mask, tags):
        lstm_feats = self.get_output_score(texts)
        loss_value = self.crf.neg_log_likelihood_loss(feats=lstm_feats, mask=mask, tags=tags)
        batch_size = lstm_feats.size(0)
        loss_value /= float(batch_size)
        return loss_value


if __name__ == '__main__':
    import sys

    sys.path.append('../../')
    from Main import args

    net = Net('Baseline', args=args)
    net(["你是傻逼吗, 我去?", "很高兴见到你, 我的名字是小花, 什么时候出去喝一杯", "大家一起放屁好不好"])
