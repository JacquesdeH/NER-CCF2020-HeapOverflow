import re
import sys
from collections import namedtuple
from ..utils.logger import alloc_logger
from ..config.DefaultConfig import DefaultConfig

"""
named_tuple LabelInfo:
ID: int
Category: str
Pos_b: int
Pos_e: int
Provacy: str
"""
label_info = namedtuple("LabelInfo", ["ID", "Category", "Pos_b", "Pos_e", "Privacy"])

class LabelFileReader:
    def __init__(self):
        self.csv_re = re.compile(r"^(\d+),(\w+),(\d+),(\d+),(.+)$")
        self.logger = alloc_logger("label_file_reader.log", default_head=LabelFileReader)
    
    def loads(self, line_content) -> label_info:
        m = self.csv_re.match(line_content)
        if m == None:
            self.logger.log_message("content cannot match pattern:", line_content)
            return None
        ID = int(m.group(1))
        Category = m.group(2)
        Pos_b = int(m.group(3))
        Pos_e = int(m.group(4))
        Privacy = m.group(5)
        ret = label_info(
            ID = ID,
            Category = Category,
            Pos_b = Pos_b,
            Pos_e = Pos_e,
            Privacy = Privacy
        )
        return ret


    def load(self, file_name: str) -> list or "List<LabelInfo>":
        ret = []
        with open(file_name, 'r', encoding='utf8') as f:
            for row, line_content in enumerate(f.readlines()):
                if row == 0:
                    continue
                new_info = self.loads(line_content)
                if new_info is not None:
                    ret.append(new_info)
        self.logger.log_message(ret)
        return ret
                
    def dumps(self, info: label_info):
        return str(info.ID) + ',' + info.Category + ',' + str(info.Pos_b) + ',' + str(info.Pos_e) + ',' + info.Privacy


if __name__ == "__main__":
    reader = LabelFileReader()
    infos = reader.load(DefaultConfig.PATHS.DATA_CCF_RAW + "/train/label/0.csv")
    print(reader.dumps(infos[0]))
