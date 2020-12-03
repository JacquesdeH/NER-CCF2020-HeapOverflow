import os


class DefaultConfig:
    class PATHS:
        IMAGE = os.path.join("image")
        LOG = os.path.join("log")
        CKPT = os.path.join("checkpoint")
        DATA = os.path.join("data")
        DATA_CCF = os.path.join(DATA, os.path.join("CCF"))
        DATA_CCF_RAW = os.path.join(DATA_CCF, os.path.join("raw"))
        DATA_CCF_CLEANED = os.path.join(DATA_CCF, os.path.join("cleaned"))
        DATA_MODULE = os.path.join(DATA, os.path.join("module"))

        # 用于debug时存放输出文件的路径，强烈建议gitignore
        DATA_CCF_DBG = os.path.join(DATA_CCF, os.path.join("debug"))

        DATA_INFO = os.path.join(DATA, os.path.join("info"))

        # 存放格式化标签的路径
        DATA_CCF_FMT = os.path.join(DATA_CCF, os.path.join("formatted"))

    class HYPER:
        PRETRAINED = 'bert-base-chinese'
        BATCH_SIZE = 16
        SEQ_LEN = 128
        EMBED_DIM = 768
        LSTM_HIDDEN = 256
        LSTM_DIRECTS = 2
        LSTM_LAYERS = 2
        LABEL_DIM = 47
        EPOCH = 3
        LR = 2e-5
        N = 1
        K = 10

    class LOG:
        DEFAULT_LOG_DIR = "default.log"
        DEFAULT_HEAD = ''
        DEFAULT_MID = ''
        DEFAULT_NEED_CONSOLE = True
