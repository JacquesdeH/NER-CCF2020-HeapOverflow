import os


class DefaultConfig:

    class PATHS:
        IMAGE = os.path.join("image")
        LOG = os.path.join("log")
        CKPT = os.path.join("checkpoint")
        DATA = os.path.join("data")
        DATA_CCF = os.path.join(DATA, os.path.join("CCF"))
        DATA_CCF_DUP = os.path.join(DATA_CCF, os.path.join("duplication"))
        DATA_CCF_DUP_CLN = os.path.join(DATA_CCF, os.path.join("dup_cleaned"))
        DATA_CCF_RAW = os.path.join(DATA_CCF, os.path.join("raw"))

    class HYPER:
        BATCH_SIZE = 32

    class LOG:
        DEFAULT_LOG_DIR = "default.log"
        DEFAULT_HEAD = ''
        DEFAULT_MID = ''
        DEFAULT_NEED_CONSOLE = True
