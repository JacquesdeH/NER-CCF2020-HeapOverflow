from ..config.DefaultConfig import DefaultConfig
from ..utils import alloc_logger
from .duplication_detector import DuplicationDetector
from .label_file_reader import LabelFileReader
from .label_formatter import LabelFormatter
from .mismatch_detector import MismatchDetector
import os
import json


class Preprocessor:
    def __init__(self, origin_dir: str=None, target_dir: str=None):
        super().__init__()
        self.origin_dir = origin_dir if origin_dir is not None else DefaultConfig.PATHS.DATA_CCF_RAW
        self.target_dir = target_dir if target_dir is not None else DefaultConfig.PATHS.DATA_CCF_CLEANED
        self.duplication_detector = DuplicationDetector()
        self.reader = LabelFileReader()
        self.label_formatter = LabelFormatter()
        self.mismatch_detector = MismatchDetector()
        self.logger = alloc_logger("preprocessor.log", Preprocessor)
        self.trasformer = self.label_formatter.fit(self.origin_dir + "/train/label")

    def produce_train(self):
        origin_data_count = len(os.listdir(self.origin_dir + "/train/data"))
        self.logger.log_message("format_all():", "origin data count:\t", origin_data_count)
        reader = LabelFileReader()

        for i in range(origin_data_count):
            with open(self.origin_dir + "/train/label/{:d}.csv".format(i), 'r', encoding='utf8') as f:
                infos = reader.load(f)
            with open(self.origin_dir + "/train/data/{:d}.txt".format(i), 'r', encoding='utf8') as f:
                data = f.read()

            data, infos = self.mismatch_detector.fix_mismatch(data, infos)

            if data is None or infos is None:
                continue

            to_remove = self.duplication_detector.auto_clean_judge(infos)

            for info in to_remove:
                infos.remove(info)
                self.logger.log_message("[{:d}]\tremoving dup\t".format(i), reader.dumps(info))

            integer_tags = self.label_formatter.infos_to_integer_list_label(infos, len(data))

            with open(self.target_dir + "/train/data/{:d}.json".format(i), 'w', encoding='utf8') as f:
                json.dump(integer_tags, f)
            with open(self.target_dir + "/train/label/{:d}.json".format(i), 'w', encoding='utf8') as f:
                json.dump(integer_tags, f)
    

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.produce_train()