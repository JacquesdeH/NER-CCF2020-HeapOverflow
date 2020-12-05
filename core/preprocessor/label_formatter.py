import json
import os
from ..config.DefaultConfig import DefaultConfig
from .label_file_reader import LabelFileReader
from .label_file_reader import LabelInfo
from ..utils import alloc_logger
from .label_transformer import BMESOLabel
from .label_transformer import LabelType
from .label_transformer import LabelTrasformer
from collections import Counter


class LabelFormatter:
    def __init__(self,
              data_dir=None,
             label_dir=None,
            target_dir=None,
            console_output: bool=True
            ):
        self.data_dir    = data_dir   if data_dir   is not None else DefaultConfig.PATHS.DATA_CCF_RAW + "/train/data"
        self.label_dir   = label_dir  if label_dir  is not None else DefaultConfig.PATHS.DATA_CCF_RAW + "/train/label"
        self.target_dir  = target_dir if target_dir is not None else DefaultConfig.PATHS.DATA_CCF_FMT + "/label"
        self.logger = alloc_logger("label_utils.log", LabelFormatter,console_output=console_output)
        self.logger.log_message("creating - - - - - - - - - - - - - - - - - - -")
        self.logger.log_message("data_dir=", self.data_dir)
        self.logger.log_message("label_dir=", self.label_dir)
        self.logger.log_message("target_dir=", self.target_dir)
        self.logger.log_message("end - - - - - - - - - - - - - - - - - - - - -")
        self.transformer = None


    def fit(self, label_dir: str=None) -> LabelTrasformer:
        label_dir = label_dir if label_dir is not None else self.label_dir
        origin_data_count = len(os.listdir(label_dir))
        self.logger.log_message("fit():", "label file count:\t", origin_data_count)
        reader = LabelFileReader()
        label_set = set()       # set[BMESOLabel]
        label_set.add(BMESOLabel("", LabelType.O))

        for i in range(origin_data_count):
            with open(label_dir + "/{:d}.csv".format(i), 'r', encoding='utf8') as f:
                infos = reader.load(f)

            for info in infos:
                type_name = info.Category
                start_index = info.Pos_b
                end_index = info.Pos_e
                if end_index - start_index > 1:
                    label_set.add(BMESOLabel(type_name, LabelType.B))
                    label_set.add(BMESOLabel(type_name, LabelType.M))
                    label_set.add(BMESOLabel(type_name, LabelType.E))
                elif start_index == end_index:   
                    label_set.add(BMESOLabel(type_name, LabelType.S))
                else:
                    label_set.add(BMESOLabel(type_name, LabelType.B))
                    label_set.add(BMESOLabel(type_name, LabelType.E))

        self.transformer = LabelTrasformer(label_set)
        return self.transformer

    def load_transformer_from_file(self, file_name: str=None):
        self.transformer = LabelTrasformer.load_from_file(file_name)
                    

    def infos_to_integer_list_label(self, infos: "Iterable[LabelInfo]", length: int) -> "List[int]":
        lst = [0] * length
        for info in infos:
            type_name = info.Category
            start_index = info.Pos_b
            end_index = info.Pos_e

            # 单字
            if start_index == end_index:           
                lst[start_index] = self.transformer.label_to_integer(BMESOLabel(type_name, LabelType.S))     # 标记单字短语
                continue

            # 多字
            m_sym = self.transformer.label_to_integer(BMESOLabel(type_name, LabelType.M))            # 名词短语中间的标记
            for i in range(start_index, end_index + 1):
                if i == start_index:
                    lst[i] = self.transformer.label_to_integer(BMESOLabel(type_name, LabelType.B))   # 标记名词短语的开头
                    continue
                if i == end_index:
                    lst[i] = self.transformer.label_to_integer(BMESOLabel(type_name, LabelType.E))   # 标记名词短语的结尾
                    continue
                lst[i] = m_sym                  # 标记名词短语的中间部分
        return lst

    def bio_str_list_label_and_token_list_to_infos(self, ID: int, bio_str_list:"Iterable[str]", tokens:"Iterable[str]", receive_rate:float) -> "List[LabelIndo]":
        ret = []
        reading = False
        counting = None
        beg = 0
        content = ""
        for i in range(len(bio_str_list)):
            label = bio_str_list[i]
            token = tokens[i].replace("##", '')
            if reading:
                if label[0] in ('B', 'O'):
                    counter = Counter(counting)
                    most, times = counter.most_common(1)[0]
                    if (times / len(counting) >= receive_rate):
                        ret.append(LabelInfo(
                            ID = ID,
                            Category = most,
                            Pos_b = beg,
                            Pos_e = len(content) - 1,
                            Privacy = content[beg:]
                            ))
                if label[0] == 'I':
                    counting.append(label[2:])
                if label[0] == 'O':
                    reading = False
            if label[0] == 'B':
                counting = [label[2:]]
                beg = len(content)
                reading = True
            content += token
        return ret

    def integer_list_label_and_data_to_infos(self, ID: int, integer_list:"Iterable[int]", data:str) -> "List[LabelIndo]":
        label_list = [None] * len(integer_list)
        ret = []
        for i in range(len(integer_list)):
            label_list[i] = self.transformer.integer_to_label(integer_list[i])

        reading = False
        name = ""
        beg = 0
        for row, label in enumerate(label_list):
            if reading:
                if label.type == LabelType.E and label.name == name:
                    new_info = LabelInfo(
                        ID = ID,
                        Category = name,
                        Pos_b = beg,
                        Pos_e = row,
                        Privacy = data[beg: row+1]
                        )
                    ret.append(new_info)
                    continue
                if label.type == LabelType.M and label.name == name:
                    continue
                if label.type == LabelType.B:
                    name = label.name
                    beg = row
                    continue
                reading = False
            else:
                # ! reading
                if label.type == LabelType.B:
                    name = label.name
                    beg = row
                    reading = True
                    continue
                if label.type == LabelType.S:
                    new_info = LabelInfo(
                        ID = ID,
                        Category = label.name,
                        Pos_b = row,
                        Pos_e = row,
                        Privacy = data[row: row+1]
                    )
                    ret.append(new_info)
                    reading = False
                    continue
                reading = False
        return ret

    def infos_to_bio_str_list_label(self, infos: "Iterable[LabelInfo]", length: int) -> "List[str]":
        lst = ["O"] * length
        for info in infos:
            type_name = info.Category
            start_index = info.Pos_b
            end_index = info.Pos_e

            # 多字
            m_sym = "I-" + type_name            # 名词短语中间的标记
            for i in range(start_index, end_index + 1):
                if i == start_index:
                    lst[i] = "B-" + type_name   # 标记名词短语的开头
                    continue
                lst[i] = m_sym                  # 标记名词短语的中间和结尾部分
        return lst

    def infos_to_str_list_label(self, infos: "Iterable[LabelInfo]", length: int) -> "List[str]":
        lst = ["O"] * length
        for info in infos:
            type_name = info.Category
            start_index = info.Pos_b
            end_index = info.Pos_e

            # 单字
            if start_index == end_index:           
                lst[start_index] = "S-" + type_name     # 标记单字短语
                continue

            # 多字
            m_sym = "M-" + type_name            # 名词短语中间的标记
            for i in range(start_index, end_index + 1):
                if i == start_index:
                    lst[i] = "B-" + type_name   # 标记名词短语的开头
                    continue
                if i == end_index:
                    lst[i] = "E-" + type_name   # 标记名词短语的结尾
                    continue
                lst[i] = m_sym                  # 标记名词短语的中间部分
        return lst

    def format_all(self):
        origin_data_count = len(os.listdir(self.data_dir))
        self.logger.log_message("format_all():", "origin data count:\t", origin_data_count)
        reader = LabelFileReader()

        for i in range(origin_data_count):
            with open(self.label_dir + "/{:d}.csv".format(i), 'r', encoding='utf8') as f:
                infos = reader.load(f)
            with open(self.data_dir + "/{:d}.txt".format(i), 'r', encoding='utf8') as f:
                length = len(f.read())

            lst = self.infos_to_str_list_label(infos, length)

            # 保存标记列表
            with open(self.target_dir + "/{:d}.json".format(i), 'w', encoding='utf8') as f:
                json.dump(lst, f)
                    

if __name__ == "__main__":
    formatter = LabelFormatter(
        data_dir=None,
        label_dir=DefaultConfig.PATHS.DATA_CCF_DBG + "/duplication_cleaned",
        target_dir=None,
        console_output=True
    )
    formatter.format_all()
