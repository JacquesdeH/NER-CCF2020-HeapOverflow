import os


class DefaultConfig:

    class PATHS:
        IMAGE = os.path.join("image")
        LOG = os.path.join("log")
        CKPT = os.path.join("checkpoint")

    class HYPER:
        BATCH_SIZE = 32
