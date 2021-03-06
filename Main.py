import argparse
import torch
import warnings

from core.config.DefaultConfig import DefaultConfig as config
from core import Instructor

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', action='store_true')
parser.add_argument('--pretrained', default=config.HYPER.PRETRAINED)
parser.add_argument('--batch_size', default=config.HYPER.BATCH_SIZE, type=int)
parser.add_argument('--seq_len', default=config.HYPER.SEQ_LEN, type=int)
parser.add_argument('--embed_dim', default=config.HYPER.EMBED_DIM, type=int)
parser.add_argument('--lstm_hidden', default=config.HYPER.LSTM_HIDDEN, type=int)
parser.add_argument('--lstm_layers', default=config.HYPER.LSTM_LAYERS, type=int)
parser.add_argument('--lstm_directs', default=config.HYPER.LSTM_DIRECTS, type=int)
parser.add_argument('--label_dim', default=config.HYPER.LABEL_DIM)
parser.add_argument('--epoch', default=config.HYPER.EPOCH)
parser.add_argument('--lr', default=config.HYPER.LR)
parser.add_argument('--n', default=config.HYPER.N)
parser.add_argument('--k', default=config.HYPER.K)
parser.add_argument('--cumul_batch', default=config.HYPER.CUMUL_BATCH)
parser.add_argument('--base_lr', default=config.HYPER.BASE_LR)


args = parser.parse_args()

args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    instructor = Instructor.Instructor('Version8', args)
    instructor.train()
    instructor.save_module()
    # instructor.load_module()
    # instructor.genTestJson()

