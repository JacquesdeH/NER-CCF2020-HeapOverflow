from ..utils import alloc_logger
from .label_file_reader import LabelInfo
from ..config.DefaultConfig import DefaultConfig

class MismatchDetector:
    def __init__(self, mismatch_file_dir: str=None):
        self.count = 0
        self.alloc_logger = alloc_logger("mismatch_detector.log", MismatchDetector)
        if mismatch_file_dir is None:
            mismatch_file_dir = DefaultConfig.PATHS.DATA_INFO
        self.report_file = open(mismatch_file_dir + '/mismatch', 'rw', encoding='utf8')

    def __del__(self):
        self.report_file.close()

    def fix_mismatch(self, data:str, infos:"List[LabelInfo]") -> (str, "List[LabelInfo]"):
        new_data = data.replace('\n', '')
        for info in infos:
            if new_data[info.Pos_b : info.Pos_e + 1] != info.Privacy:
                self.count += 1


