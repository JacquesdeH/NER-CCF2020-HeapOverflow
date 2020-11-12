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
LabelInfo = namedtuple("LabelInfo", ["ID", "Category", "Pos_b", "Pos_e", "Privacy"])

class LabelFileReader:
    def __init__(self):
        self.csv_re = re.compile(r"^(\d+),(\w+),(\d+),(\d+),(.+)$")
        self.logger = alloc_logger("label_file_reader.log", default_head=LabelFileReader)
    
    def loads(self, line_content) -> LabelInfo:
        m = self.csv_re.match(line_content)
        if m == None:
            self.logger.log_message("loads()\t:", "content cannot match pattern:", line_content)
            return None
        ID = int(m.group(1))
        Category = m.group(2)
        Pos_b = int(m.group(3))
        Pos_e = int(m.group(4))
        Privacy = m.group(5)
        ret = LabelInfo(
            ID = ID,
            Category = Category,
            Pos_b = Pos_b,
            Pos_e = Pos_e,
            Privacy = Privacy
        )
        return ret


    def load(self, fp) -> "List[LabelInfo]":
        ret = []
        for row, line_content in enumerate(fp.readlines()):
            if row == 0:
                continue
            new_info = self.loads(line_content)
            if new_info is not None:
                ret.append(new_info)
        self.logger.file_message("load():\n", "infos:\t", ret)
        return ret
                
    def dumps(self, info: LabelInfo):
        return str(info.ID) + ',' + info.Category + ',' + str(info.Pos_b) + ',' + str(info.Pos_e) + ',' + info.Privacy

    def dump(self, infos : "List[LabelInfo]", fp):
        header = "ID,Category,Pos_b,Pos_e,Privacy"
        fp.write(header + '\n')
        for info in infos:
            fp.write(self.dumps(info) + '\n')

if __name__ == "__main__":
    reader = LabelFileReader()
    with open(DefaultConfig.PATHS.DATA_CCF_RAW + "/train/label/0.csv", 'r', encoding='utf8') as f:
        infos = reader.load(f)
    print(reader.dumps(infos[0]))
    with open("label_file_reader.debug", 'w', encoding='utf8') as f:
       reader.dump(infos, f)
    
