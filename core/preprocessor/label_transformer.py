import enum

from collections import namedtuple

BIESOLabel = namedtuple("BIESOLabel", ["name", "type"])

class LabelType(enum.IntEnum):
    B = 0
    M = 1
    E = 2
    S = 3
    O = 4

class LabelTrasformer:
    def __init__(self):
        self.name_table = []
        self.name_index = {}        # k=label_name: str, v=base_index: int

    def _alloc_integer_for_label_name(self, label_name: str) -> int:
        base_index = len(self.name_table)
        self.name_table.append(label_name)
        self.name_index[label_name] = base_index
        return base_index

    def label_to_integer(self, label: BIESOLabel) -> int:
        if label.name in self.name_index.keys():
            return self.name_index[label.name] * len(LabelType) + label.type
        else:
            return self._alloc_integer_for_label_name(label.name) * len(LabelType) + label.type
    
    def integer_to_label(self, integer: int) -> BIESOLabel:
        name_index = integer // len(LabelType)
        location_index = integer % len(LabelType)
        return BIESOLabel(self.name_table[name_index], LabelType(location_index))

    def label_to_string(self, label: BIESOLabel) -> str:
        return label.type.name + '-' + label.name

    def string_to_label(self, string: str) -> BIESOLabel:
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
