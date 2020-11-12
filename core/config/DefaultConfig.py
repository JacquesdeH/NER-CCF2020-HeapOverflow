import os


class DefaultConfig:

    class PATHS:
        IMAGE = os.path.join("image")
        LOG = os.path.join("log")
        CKPT = os.path.join("checkpoint")
        DATA = os.path.join("data")
        DATA_CCF = os.path.join(DATA, os.path.join("CCF"))
        DATA_CCF_RAW = os.path.join(DATA_CCF, os.path.join("raw"))

        # 存放重复标签信息的文件的路径
        DATA_CCF_DUP = os.path.join(DATA_CCF, os.path.join("duplication_cleaned"))

        # 存放消除重复标签后的标签文件的路径
        DATA_CCF_DUP_CLN = os.path.join(DATA_CCF, os.path.join("duplication_cleaned"))

        # 存放格式化标签的路径
        DATA_CCF_FMT = os.path.join(DATA_CCF, os.path.join("formatted"))

    class HYPER:
        BATCH_SIZE = 32

    class LOG:
        DEFAULT_LOG_DIR = "default.log"
        DEFAULT_HEAD = ''
        DEFAULT_MID = ''
        DEFAULT_NEED_CONSOLE = True
