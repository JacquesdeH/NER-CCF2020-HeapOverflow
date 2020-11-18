import enum

from collections import namedtuple

BIESOLabel = namedtuple("BIESOLabel", ["name", "type"])

class LabelType(enum.IntEnum):
    O = 0
    B = 1
    M = 2
    E = 3
    S = 4

class LabelTrasformer:
    LABEL_O = BIESOLabel("", LabelType.O)

    def __init__(self, possible_labels: "Iterable[BIESOLabel]"):
        self.label_table = []
        self.index = {}        # k=label_name: str, v=base_index: int

        _set = set(possible_labels)
        self.label_table = [label for label in _set if label.type is not LabelType.O]   # delete all O
        self.label_table.sort(key=lambda label: label.type)
        self.label_table.sort(key=lambda label: label.name)
        self.label_table = [LABEL_O] + self.label_table     # make O as zero
        for idx, label in enumerate(self.label_table):
            self.index[label] = idx

    def label_to_integer(self, label: BIESOLabel) -> int:
        if label.type is LabelType.O:
            return 0
        if label.name in self.name_index.keys():
            return self.name_index[label.name] * len(LabelType) + label.type
        else:
            return self._alloc_integer_for_label_name(label.name) * len(LabelType) + label.type
    
    def integer_to_label(self, integer: int) -> BIESOLabel:
        if integer == 0:
            return LABEL_O
        name_index = integer // len(LabelType)
        location_index = integer % len(LabelType)
        return BIESOLabel(self.name_table[name_index], LabelType(location_index))

    def label_to_string(self, label: BIESOLabel) -> str:
        if label.type is LabelType.O:
            return "O"
        return label.type.name + '-' + label.name

    def string_to_label(self, string: str) -> BIESOLabel:
        if string == "O":
            return LABEL_O
        return BIESOLabel(string[2:], LabelType[string[:1]])



if __name__ == "__main__":
    l = BIESOLabel("shit", LabelType.B)
    transformer = LabelTrasformer()
    print(l)
    print(transformer.label_to_string(l))
    print(transformer.label_to_integer(l))
    ll = BIESOLabel("shit", LabelType.B)
    lll = BIESOLabel("shit", LabelType.M)
    llll = BIESOLabel("fuck", LabelType.B)
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
