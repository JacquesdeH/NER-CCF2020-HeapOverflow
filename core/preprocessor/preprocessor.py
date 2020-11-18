from ..config.DefaultConfig import DefaultConfig
from ..utils import alloc_logger
from .duplication_detector import DuplicationDetector
from .label_file_reader import LabelFileReader
from .label_formatter import LabelFormatter
from .mismatch_detector import MismatchDetector
from .label_transformer import LabelTrasformer
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
        signature = "produce_train():\t"

        self.logger.log_message(signature, "start!")

        origin_data_count = len(os.listdir(self.origin_dir + "/train/data"))
        self.logger.log_message(signature, "origin data count:\t", origin_data_count)
        reader = LabelFileReader()

        dup_count = 0
        unsolve_mismatch = []

        for i in range(origin_data_count):
            with open(self.origin_dir + "/train/label/{:d}.csv".format(i), 'r', encoding='utf8') as f:
                infos = reader.load(f)
            with open(self.origin_dir + "/train/data/{:d}.txt".format(i), 'r', encoding='utf8') as f:
                data = f.read()

            data, infos = self.mismatch_detector.fix_mismatch(data, infos)

            if data is None or infos is None:
                unsolve_mismatch.append(i)
                continue

            to_remove = self.duplication_detector.auto_clean_judge(infos)

            for info in to_remove:
                infos.remove(info)
                dup_count += 1
                self.logger.log_message("produce_train():\t", "[{:d}]\tremoving dup\t".format(i), reader.dumps(info))

            integer_tags = self.label_formatter.infos_to_integer_list_label(infos, len(data))

            with open(self.target_dir + "/train/data/{:d}.txt".format(i), 'w', encoding='utf8') as f:
                f.write(data)
            with open(self.target_dir + "/train/label/{:d}.json".format(i), 'w', encoding='utf8') as f:
                json.dump(integer_tags, f)
        
        self.logger.log_message(signature, "remove duplication {:d} times".format(dup_count))
        self.logger.log_message(signature, "detect {:d} unsolved mismatch".format(len(unsolve_mismatch)))
        if len(unsolve_mismatch) != 0:
            self.logger.log_message(signature, "their ID are:")
            for unsolve_id in unsolve_mismatch:
                self.logger.log_message(signature, "\t{:d}".format(unsolve_id))

        self.logger.log_message(signature, "finish!")

    def produce_test(self):
        signature = "produce_test():\t"
        self.logger.log_message(signature, "start!")
        origin_data_count = len(os.listdir(self.origin_dir + "/test"))
        self.logger.log_message(signature, "origin data count:\t", origin_data_count)
        reader = LabelFileReader()

        for i in range(origin_data_count):
            with open(self.origin_dir + "/test/{:d}.txt".format(i), 'r', encoding='utf8') as f:
                data = f.read()
            
            data = data.replace('\n', '')

            with open(self.target_dir + "/test/data/{:d}.txt".format(i), 'w', encoding='utf8') as f:
                f.write(data)
        self.logger.log_message(signature, "finish!")

def quick_preproduce() -> LabelTrasformer:
    logger = alloc_logger()
    try:
        new_dir = DefaultConfig.PATHS.DATA_CCF_CLEANED + "/test/data"
        logger.log_message("mkdir " + new_dir)
        os.makedirs(new_dir)
    except FileExistsError:
        logger.log_message("has existed")
    try:
        new_dir = DefaultConfig.PATHS.DATA_CCF_CLEANED + "/test/label"
        logger.log_message("mkdir " + new_dir)
        os.makedirs(new_dir)
    except FileExistsError:
        logger.log_message("has existed")
    try:
        new_dir = DefaultConfig.PATHS.DATA_CCF_CLEANED + "/train/data"
        logger.log_message("mkdir " + new_dir)
        os.makedirs(new_dir)
    except FileExistsError:
        logger.log_message("has existed")
    try:
        new_dir = DefaultConfig.PATHS.DATA_CCF_CLEANED + "/train/label"
        logger.log_message("mkdir " + new_dir)
        os.makedirs(new_dir)
    except FileExistsError:
        logger.log_message("has existed")
    preprocessor = Preprocessor()
    preprocessor.produce_train()
    preprocessor.produce_test()
    return preprocessor.trasformer

if __name__ == "__main__":
    quick_preproduce()