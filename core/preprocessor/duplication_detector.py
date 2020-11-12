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
        with open(self.duplication_info_dir + '/duplication.md', 'w', encoding='utf8') as dupfile:
            dup_count = 0
            for i in range(raw_data_count):
                data_file_name = self.raw_data_set_dir + "/train/data/" + str(i) + ".txt"
                label_file_name = self.raw_data_set_dir + "/train/label/" + str(i) + ".csv"
                with open(label_file_name, 'r', encoding='utf8') as f:
                    infos = reader.load(f)
                infos.sort(key=lambda info: info.Pos_b)
                print(infos)
                for j in range(len(infos) - 1):
                    if infos[j].Pos_e >= infos[j + 1].Pos_b:
                        self.logger.log_message("detect():\t", "detect duplication with id={:d}".format(infos[j].ID))
                        dup_count += 1
                        dupfile.write("---\n\n")
                        dupfile.write("## {:d}\n\n".format(dup_count))
                        dupfile.write("ID=[{:d}]\n\n".format(infos[j].ID))
                        with open(data_file_name, 'r', encoding='utf8') as f:
                            txt = f.read()
                        dupfile.write(txt + '\n\n')
                        dupfile.write("- [] [{:d}:{:d}]\t<{:s}>\t{:s}\n\n".format(infos[j].Pos_b, infos[j].Pos_e, infos[j].Category, infos[j].Privacy) )
                        dupfile.write("- [] [{:d}:{:d}]\t<{:s}>\t{:s}\n\n".format(infos[j+1].Pos_b, infos[j+1].Pos_e, infos[j+1].Category, infos[j+1].Privacy))
                # break
            self.logger.log_message("detect():\t", "detect ", dup_count, " duplications")

        
        
if __name__ == "__main__":
    detector = DuplicationDetector()
    detector.detect()
    