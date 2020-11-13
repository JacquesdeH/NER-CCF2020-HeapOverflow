from ..utils import alloc_logger
from .label_file_reader import LabelInfo
from ..config.DefaultConfig import DefaultConfig
import json
from ..utils import alloc_logger



class MismatchDetector:
    def __init__(self, mismatch_file_dir: str=None):
        self.mismatch_count = 0
        self.fix_count = 0
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
        reader = LabelFileReader()
        for info in infos:
            if new_data[info.Pos_b : info.Pos_e + 1] != info.Privacy:
                self.mismatch_count += 1
                if info.ID in self.tactics.keys():
                    tac = self.tactics[info.ID]
                    if tac["solved"] == "true":
                        self.fix_count += 1
                        new_new_data = tac["data"]
                        new_infos = tac["labels"]
                        new_infos = [map(reader.loads, new_infos)]
                        # 检查是否正确
                        for info in infos:
                            if new_new_data[info.Pos_b : info.Pos_e + 1] != info.Privacy:
                                tac["solved"] = "false"
                                return (None, None)
                        return (new_new_data, new_infos)
                else:
                    msg = {
                        "solved": "false",
                        "data" : new_data,
                        "labels" : infos
                    }
                    self.tactics[info.ID] = 
                    return (None, None)
        return (new_data, infos)

