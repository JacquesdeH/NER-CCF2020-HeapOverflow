from ..utils import alloc_logger
from .label_file_reader import LabelInfo
from ..config.DefaultConfig import DefaultConfig
import json
from ..utils import alloc_logger



class MismatchDetector:
    def __init__(self, mismatch_file_dir: str=None):
        self.count = 0
        self.alloc_logger = alloc_logger("mismatch_detector.log", MismatchDetector)
        self.mismatch_file_dir = mismatch_file_dir if mismatch_file_dir is not None else DefaultConfig.PATHS.DATA_INFO
        
        
        """
        {
            <ID> : {
                "solved": "<true/false>",
                "data": "<data>",
                "labels": [
                    <str-type LabelInfo>
                ]
            }
        }
        """
        self.tactics = None
        with open(self.mismatch_file_dir + "/mismatch_tactics.json", 'r', encoding='utf8') as f:
            self.tactics = json.load(f)         


    def __del__(self):
        with open(self.mismatch_file_dir + "/mismatch_tactics.json", 'w', encoding='utf8') as f:
            json.dump(self.tactics, f)

    def fix_mismatch(self, data:str, infos:"List[LabelInfo]") -> (str, "List[LabelInfo]"):
        new_data = data.replace('\n', '')
        for info in infos:
            if new_data[info.Pos_b : info.Pos_e + 1] != info.Privacy:
                self.count += 1


