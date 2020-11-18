import enum
from ..utils import alloc_logger
from collections import namedtuple

BMESOLabel = namedtuple("BMESOLabel", ["name", "type"])

class LabelType(enum.IntEnum):
    O = 0
    B = 1
    M = 2
    E = 3
    S = 4

class LabelTrasformer:
    LABEL_O = BMESOLabel("", LabelType.O)

    def __init__(self, possible_labels: "Iterable[BMESOLabel]"):
        self.label_table = []
        self.index = {}        # k=label_name: str, v=base_index: int

        self.logger = alloc_logger("label_transformer.log",LabelTrasformer)
        self.logger.log_message("init(): build index:")

        _set = set(possible_labels)
        self.label_table = [label for label in _set if label.type is not LabelType.O]   # delete all O
        self.label_table.sort(key=lambda label: label.type)
        self.label_table.sort(key=lambda label: label.name)

        self.label_table = [LABEL_O] + self.label_table     # make O as zero
        for idx, label in enumerate(self.label_table):
            self.index[label] = idx
            self.logger(idx, "\t:\t", self.label_to_string(label))

    def label_to_integer(self, label: BMESOLabel) -> int:
        if label.type is LabelType.O:
            return 0
        if label.name in self.name_index.keys():
            return self.name_index[label.name] * len(LabelType) + label.type
        else:
            return self._alloc_integer_for_label_name(label.name) * len(LabelType) + label.type
    
    def integer_to_label(self, integer: int) -> BMESOLabel:
        if integer == 0:
            return LABEL_O
        name_index = integer // len(LabelType)
        location_index = integer % len(LabelType)
        return BMESOLabel(self.name_table[name_index], LabelType(location_index))

    @staticmethod
    def label_to_string(label: BMESOLabel) -> str:
        if label.type is LabelType.O:
            return "O"
        return label.type.name + '-' + label.name

    @staticmethod
    def string_to_label(string: str) -> BMESOLabel:
        if string == "O":
            return LABEL_O
        return BMESOLabel(string[2:], LabelType[string[:1]])



if __name__ == "__main__":
    l = BMESOLabel("shit", LabelType.B)
    transformer = LabelTrasformer()
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
