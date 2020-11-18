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
    def __init__(self, possible_labels: "Iterable[BMESOLabel]"):
        self.LABEL_O = BMESOLabel("", LabelType.O)
        self.label_table = []
        self.label_index = {}        # k=label: BMESOLabel, v=index: int

        self.logger = alloc_logger("label_transformer.log",LabelTrasformer)
        self.logger.log_message("init(): build index:")

        _set = set(possible_labels)
        self.label_table = [label for label in _set if label.type is not LabelType.O]   # delete all O
        self.label_table.sort(key=lambda label: label.type)
        self.label_table.sort(key=lambda label: label.name)

        self.label_table = [self.LABEL_O] + self.label_table     # make O as zero
        for idx, label in enumerate(self.label_table):
            self.label_index[label] = idx
            self.logger.log_message(idx, "\t:\t", self.label_to_string(label))

    def label_to_integer(self, label: BMESOLabel) -> int:
        if label.type is LabelType.O:
            return 0
        return self.label_index[label]
    
    def integer_to_label(self, integer: int) -> BMESOLabel:
        return self.label_table[integer]

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
