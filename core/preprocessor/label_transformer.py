import enum
import os
import json
from ..utils import alloc_logger
from ..config.DefaultConfig import DefaultConfig
from collections import namedtuple

BMESOLabel = namedtuple("BMESOLabel", ["name", "type"])

class LabelType(enum.IntEnum):
    O = 0
    B = 1
    M = 2
    E = 3
    S = 4

class LabelTrasformer:
    def __init__(self, possible_labels: "Iterable[BMESOLabel]"):
        self.LABEL_O = BMESOLabel("", LabelType.O)
        self.label_table = []
        self.label_index = {}        # k=label: BMESOLabel, v=index: int

        self.logger = alloc_logger("label_utils.log",LabelTrasformer)
        self.logger.log_message("init(): build index:")

        _set = set(possible_labels)
        self.label_table = [label for label in _set if label.type is not LabelType.O]   # delete all O
        self.label_table.sort(key=lambda label: label.type)
        self.label_table.sort(key=lambda label: label.name)

        self.label_table = [self.LABEL_O] + self.label_table     # make O as zero
        for idx, label in enumerate(self.label_table):
            self.label_index[label] = idx
            self.logger.log_message(idx, "\t:\t", self.label_to_string(label))

    def log_bio_type_to_file(self):
        table_file_name = os.path.join(DefaultConfig.PATHS.DATA_INFO, "bio_table.json")

        save_label_table = []
        s = set()
        for label in self.label_table:
            bio_string = self.label_to_bio_string(label)
            if bio_string not in s:
                s.add(bio_string)
                save_label_table.append(bio_string)

        with open(table_file_name, 'w', encoding='utf8') as f:
            json.dump(save_label_table, f)
            self.logger.log_message("save bio label table in file [", table_file_name, ']')

    def save_to_file(self):
        map_file_name = os.path.join(DefaultConfig.PATHS.DATA_INFO, "trans_map.json")
        table_file_name = os.path.join(DefaultConfig.PATHS.DATA_INFO, "trans_table.json")

        save_label_table = list(map(self.label_to_string, self.label_table))
        save_label_index = {}
        for k, v in self.label_index.items():
            save_label_index[self.label_to_string(k)] = v

        with open(table_file_name, 'w', encoding='utf8') as f:
            json.dump(save_label_table, f)
            self.logger.log_message("save label table in file [", table_file_name, ']')
        with open(map_file_name, 'w', encoding='utf8') as f:
            json.dump(save_label_index, f)
            self.logger.log_message("save trans map in file [", map_file_name, ']')

    @staticmethod
    def load_from_file(map_file_name: str = None, table_file_name: str = None):
        if map_file_name is None:
            map_file_name = os.path.join(DefaultConfig.PATHS.DATA_INFO, "trans_map.json")
        if table_file_name is None:
            table_file_name = os.path.join(DefaultConfig.PATHS.DATA_INFO, "trans_table.json")

        
        ret = LabelTrasformer([])
        with open(table_file_name, 'r', encoding='utf8') as f:
            save_label_table = json.load(f)
            ret.logger.log_message("load label table from file [", table_file_name, ']')
        with open(map_file_name, 'r', encoding='utf8') as f:
            save_label_index = json.load(f)
            ret.logger.log_message("load trans map from file [", map_file_name, ']')
        
        ret.label_table = list(map(ret.string_to_label, save_label_table))
        for k, v in save_label_index.items():
            ret.label_index[ret.string_to_label(k)] = v
        
        return ret

        

    def label_to_integer(self, label: BMESOLabel) -> int:
        if label.type is LabelType.O:
            return 0
        return self.label_index[label]
    
    def integer_to_label(self, integer: int) -> BMESOLabel:
        return self.label_table[integer]

    def label_to_bio_string(self, label: BMESOLabel) -> str:
        if label.type is LabelType.O:
            return "O"
        tp = label.type
        if tp == LabelType.M or tp == LabelType.E:
            tp = "I"
        elif tp == LabelType.S:
            tp = "B"
        else:
            tp = tp.name
        return tp + '-' + label.name

    def label_to_string(self, label: BMESOLabel) -> str:
        if label.type is LabelType.O:
            return "O"
        return label.type.name + '-' + label.name

    def string_to_label(self, string: str) -> BMESOLabel:
        if string == "O":
            return self.LABEL_O
        return BMESOLabel(string[2:], LabelType[string[:1]])



if __name__ == "__main__":
    l = BMESOLabel("shit", LabelType.B)
    transformer = LabelTrasformer(
        [
            BMESOLabel("shit", LabelType.B), 
            BMESOLabel("shit", LabelType.M),
            BMESOLabel("shit", LabelType.E),
            BMESOLabel("fuck", LabelType.B)
        ]
    )
    print(l)
    print(transformer.label_to_string(l))
    print(transformer.label_to_integer(l))
    ll = BMESOLabel("shit", LabelType.B)
    lll = BMESOLabel("shit", LabelType.M)
    llll = BMESOLabel("fuck", LabelType.B)
    print(transformer.label_to_string(ll))
    print(transformer.label_to_string(lll))
    print(transformer.label_to_string(llll))
    print(transformer.label_to_integer(llll))

    l = transformer.string_to_label("B-shit")
    print(transformer.label_to_integer(l))
    ll = transformer.string_to_label("E-shit")
    print(transformer.label_to_integer(ll))
    lll = transformer.string_to_label("B-fuck")
    print(transformer.label_to_integer(lll))
