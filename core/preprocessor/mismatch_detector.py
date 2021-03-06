from ..utils import alloc_logger
from .label_file_reader import LabelFileReader
from .label_file_reader import LabelInfo
from ..config.DefaultConfig import DefaultConfig
import json
import os
from ..utils import alloc_logger



class MismatchDetector:
    def __init__(self, mismatch_file_dir: str=None):
        self.mismatch_count = 0
        self.fix_count = 0
        self.logger = alloc_logger("detectors.log", MismatchDetector)
        self.mismatch_file_dir = mismatch_file_dir if mismatch_file_dir is not None else DefaultConfig.PATHS.DATA_INFO
        self.reader = LabelFileReader()
        
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

    def save(self):
        with open(self.mismatch_file_dir + "/mismatch_tactics.json", 'w', encoding='utf8') as f:
            json.dump(self.tactics, f, ensure_ascii=False)

    # def __del__(self):
    #    self.save()

    @staticmethod
    def remove_special_char(data: str)->str:
        return data                 \
            .replace('\n', '')      \
            .replace('\r', '')      \
            .replace(' ', '')       \
            .replace('\b', '')      \
            .replace('\v', '')      \
            .replace('\f', '')      \
            .replace('\u2028', '')  \
            .replace('\u2029', '')  \
            .replace('\u20A0', '')  \
            .replace('\uFEFF', '')

    def fix_mismatch(self, data:str, infos:"Iterable[LabelInfo]", remove_spacial=True) -> (str, "List[LabelInfo]"):
        new_data = self.remove_special_char(data) if remove_spacial else data
        reader = self.reader
        for no, info in enumerate(infos):
            if new_data[info.Pos_b : info.Pos_e + 1] != info.Privacy:
                self.mismatch_count += 1
                if str(info.ID) in self.tactics.keys():
                    tac = self.tactics[str(info.ID)]
                    if tac["solved"] == "true":
                        self.fix_count += 1
                        new_new_data = tac["data"]
                        new_infos = tac["labels"]
                        new_infos = list(map(reader.loads, new_infos))
                        # 检查是否正确
                        for info in new_infos:
                            if new_new_data[info.Pos_b : info.Pos_e + 1] != info.Privacy:
                                tac["solved"] = "false"
                                self.logger.log_message("mismatch not solve with id={:d}".format(infos[0].ID))
                                return (None, None)
                        self.logger.log_message("solve a mismatch with id={:d}".format(infos[0].ID))
                        return (new_new_data, new_infos)
                self.logger.log_message("detect a mismatch with id={:d}".format(infos[0].ID))
                msg = {
                    "solved": "false",
                    "data" : new_data,
                    "labels" : list(map(reader.dumps, infos)),
                    "no" : no
                }
                self.tactics[str(info.ID)] = msg
                return (None, None)
        return (new_data, infos)


if __name__ == "__main__":
    reader = LabelFileReader()
    detector = MismatchDetector()

    data_dir = DefaultConfig.PATHS.DATA_CCF_RAW + '/train/data'
    label_dir = DefaultConfig.PATHS.DATA_CCF_RAW + '/train/label'

    data_count = len(os.listdir(data_dir))

    for i in range(data_count):
        with open(data_dir + "/{:d}.txt".format(i), 'r', encoding='utf8') as f:
            data = f.read()
        with open(label_dir + "/{:d}.csv".format(i), 'r', encoding='utf8') as f:
            infos = reader.load(f)
        detector.fix_mismatch(data, infos)

    detector.save()
