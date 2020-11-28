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

        # 用于debug时存放输出文件的路径，强烈建议gitignore
        DATA_CCF_DBG = os.path.join(DATA_CCF, os.path.join("debug"))

        DATA_INFO = os.path.join(DATA, os.path.join("info"))

        # 存放格式化标签的路径
        DATA_CCF_FMT = os.path.join(DATA_CCF, os.path.join("formatted"))

    class HYPER:
        BATCH_SIZE = 32
        SEQ_LEN = 500

    class LOG:
        DEFAULT_LOG_DIR = "default.log"
        DEFAULT_HEAD = '_DEFAULT_'
        DEFAULT_MID = ''
        DEFAULT_NEED_CONSOLE = True
