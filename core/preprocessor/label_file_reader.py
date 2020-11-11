import re
from collections import namedtuple

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
        self.csv_re = re.compile(r"^(\d+),(\w+),(\d+),(\d+),(\w+)$")

    def read_file(self, file_name: str) -> list or "List<LabelInfo>":
        with open(file_name, 'r', encoding='utf8') as f:
            for row, line_content in enumerate(f.readlines()):
                if row == 0:
                    continue
                m = self.csv_re.match(line_content)
                if m == None:
                    