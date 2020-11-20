import argparse
from core.config.DefaultConfig import DefaultConfig as config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', default=config.HYPER.BATCH_SIZE)
    parser.add_argument('--seq_len', default=config.HYPER.SEQ_LEN)
    parser.add_argument('--label_dim', default=config.HYPER.LABEL_DIM)

    args = parser.parse_args()
