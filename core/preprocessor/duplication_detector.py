import re


class DuplicationDetector:
    def __init__(self, train_set_dir: str):
        self.train_set_dir = train_set_dir
        self.duplication_list = []  # list of tuple

    def detect(self):
