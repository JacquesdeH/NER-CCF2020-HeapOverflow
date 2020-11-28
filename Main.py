import argparse
import torch

from core.config.DefaultConfig import DefaultConfig as config


parser = argparse.ArgumentParser()

parser.add_argument('--cuda', action='store_true')
parser.add_argument('--pretrained', default=config.HYPER.PRETRAINED)
parser.add_argument('--batch_size', default=config.HYPER.BATCH_SIZE, type=int)
parser.add_argument('--seq_len', default=config.HYPER.SEQ_LEN, type=int)
parser.add_argument('--embed_dim', default=config.HYPER.EMBED_DIM, type=int)
parser.add_argument('--lstm_hidden', default=config.HYPER.LSTM_HIDDEN, type=int)
parser.add_argument('--lstm_layers', default=config.HYPER.LSTM_LAYERS, type=int)
parser.add_argument('--lstm_directs', default=config.HYPER.LSTM_DIRECTS, type=int)

args = parser.parse_args()

args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
