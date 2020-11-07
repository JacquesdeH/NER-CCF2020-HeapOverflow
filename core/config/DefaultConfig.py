import os


class DefaultConfig:

    class PATHS:
        IMAGE = os.path.join("image")
        LOG = os.path.join("log")
        CKPT = os.path.join("checkpoint")
        DATA = os.path.join("data")
        DATA_CCF = os.path.join(DATA, os.path.join("CCF"))
        DATA_CCF_RAW = os.path.join(DATA_CCF, os.path.join("raw"))

    class HYPER:
        BATCH_SIZE = 32
