import re
import os
import sys
from ..config.DefaultConfig import DefaultConfig
from .label_file_reader import LabelFileReader
from ..utils import alloc_logger

class DuplicationDetector:
    def __init__(self, raw_data_set_dir: str = None, duplication_info_dir: str = None, dup_cleaned_set_dir: str = None):
        self.raw_data_set_dir = raw_data_set_dir if raw_data_set_dir is not None else DefaultConfig.PATHS.DATA_CCF_RAW
        self.duplication_info_dir = duplication_info_dir if duplication_info_dir is not None else DefaultConfig.PATHS.DATA_CCF_DUP
        self.dup_cleaned_set_dir = dup_cleaned_set_dir if dup_cleaned_set_dir is not None else DefaultConfig.PATHS.DATA_CCF_DUP_CLN
        self.duplication_list = []  # list of tuple
        self.logger = alloc_logger("dup_detector.log", DuplicationDetector)

    def detect(self):
        # TODO
        raw_data_count = len(os.listdir(self.raw_data_set_dir + "/train/data"))
        self.logger.log_message("detect():", "raw data count:\t", raw_data_count)
        reader = LabelFileReader()
        for i in range(raw_data_count):
            data_file_name = self.raw_data_set_dir + "/train/data/" + str(i) + ".txt"
            label_file_name = self.raw_data_set_dir + "/train/label/" + str(i) + ".csv"
            


        
        
if __name__ == "__main__":
    detector = DuplicationDetector()
    detector.detect()
    