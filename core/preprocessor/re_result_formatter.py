from ..utils import alloc_logger
from ..config.DefaultConfig import DefaultConfig
from .label_transformer import LabelTrasformer
from .label_formatter import LabelFormatter
from .label_file_reader import LabelFileReader
from .label_file_reader import LabelInfo
import os
import json
import re

class ReResultFormatter:
    def __init__(self, split_index_file: str=None, end='\n'):
        super().__init__()
        
        self.logger = alloc_logger("re_result_formatter.log", ReResultFormatter)
        self.end = end
        self.combine_index = {} # origin: (beg, target)
        split_index_file = split_index_file if split_index_file is not None else os.path.join(DefaultConfig.PATHS.DATA_INFO, "split_index_test.json")
        
        self.logger.log_message("loading split_index from file:\t", split_index_file)
        with open(split_index_file, 'r', encoding='utf8') as f:
            m = json.load(f)
            for item in m:
                target = item["target"]
                origin = item["origin"]
                beg = item["beg"]
                if origin in self.combine_index.keys():
                    self.combine_index[origin].append((beg, target))
                else:
                    self.combine_index[origin] = [(beg, target)]

    def combine_all(self, receive_rate: float=0.8, result_dir: str=None):
        def file_num_to_tokens_and_labels(num: int)->(list, list):
            with open(os.path.join(result_dir, "{:d}.json".format(i)), 'r', encoding='utf8') as f:
                result = json.load(f)
                return result["data"][1:-1], result["tags"][1:-1]

        signature = "conbine_all()\t"

        self.logger.log_message(signature, "start!")

        result_dir = result_dir if result_dir is not None else os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "test/label")
        self.logger.log_message(signature, "origin result dir:\t", result_dir)

        origin_result_count = len(os.listdir(os.path.join(DefaultConfig.PATHS.DATA_CCF_RAW, "test")))
        # TODO
        origin_result_count = 500
        self.logger.log_message(signature, "origin result count:\t", origin_result_count)


        label_formatter = LabelFormatter()
        label_formatter.load_transformer_from_file()

        reader = LabelFileReader()

        output_csv = open(os.path.join(DefaultConfig.PATHS.DATA_INFO, "predict_origin.csv"), 'w', encoding='utf8')


        label_token_len_mismatch = []
        for i in range(origin_result_count):
            # labels = None
            tokens, labels = file_num_to_tokens_and_labels(i)
            if i in self.combine_index.keys():
                targets = self.combine_index[i]
                targets.sort(key=lambda t: t[0])
                for _, target in targets:
                    new_tokens, new_labels = file_num_to_tokens_and_labels(target)
                    labels += new_labels
                    tokens += new_tokens
            if len(tokens) != len(labels):
                label_token_len_mismatch.append(i)
                continue
            infos = label_formatter.bio_str_list_label_and_token_list_to_infos(ID=i, bio_str_list=labels, tokens=tokens, receive_rate=receive_rate)
            for info in infos:
                string = reader.dumps(info)
                output_csv.write(string + self.end)
        output_csv.close()

        self.logger.log_message("mismatches:\t", label_token_len_mismatch)

    def trans_origin_to_raw(self, data_dir: str=None):
        signature = "trans_origin_to_raw()\t"

        input_csv = open(os.path.join(DefaultConfig.PATHS.DATA_INFO, "predict_origin.csv"), 'r', encoding='utf8')
        output_csv = open(os.path.join(DefaultConfig.PATHS.DATA_INFO, "predict.csv"), 'w', encoding='utf8')

        reader = LabelFileReader()

        data_dir = data_dir if data_dir is not None else os.path.join(DefaultConfig.PATHS.DATA_CCF_RAW, "test")

        not_in_raw_count = 0
        for line in input_csv.readlines():
            info = reader.loads(line)
            content = info.Privacy
            beg = info.Pos_b
            end = info.Pos_e
            ID = info.ID
            with open(os.path.join(data_dir, "{:d}.txt".format(ID)), 'r', encoding='utf8') as f:
                raw_content = f.read()
            new_beg = raw_content.lower().find(content, max(beg - 13, 0), min(end + 13, len(raw_content)))
            new_info = None
            if new_beg < 0:
                m = re.search(content.replace("[UNK]", '.*?'), raw_content.lower()[max(beg - 5, 0): min(end + 5, len(raw_content))])
                if m is not None:
                    new_info = LabelInfo(
                        ID = ID,
                        Category = info.Category,
                        Pos_b = m.start(),
                        Pos_e = m.end() - 1,
                        Privacy = content
                    )
            else:
                new_end = new_beg + end - beg
                new_info = LabelInfo(
                    ID = ID,
                    Category = info.Category,
                    Pos_b = new_beg,
                    Pos_e = new_end,
                    Privacy = content
                )
            if new_info is None:
                self.logger.log_message(signature, "[{:d}]content not found in raw:\t".format(ID), reader.dumps(info))
                self.logger.log_message(signature, "in:\t", content in raw_content)
                self.logger.log_message(signature, "raw_content:\t", raw_content)
                not_in_raw_count += 1
                continue
            new_line = reader.dumps(new_info)
            output_csv.write(new_line + self.end)


        self.logger.log_message(signature, "not in raw count=", not_in_raw_count)
        self.logger.log_message(signature, "finish")
        input_csv.close()
        output_csv.close()

if __name__ == "__main__":
    # formatter = ReResultFormatter(os.path.join(DefaultConfig.PATHS.DATA_INFO, "split_index_train.json"))
    # formatter.combine_all(
    #    origin_data_count=2515, 
    #    label_dir=os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train/label"),
    #    data_dir=os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train/data"))
    # formatter.trans_origin_to_raw(data_dir=os.path.join(DefaultConfig.PATHS.DATA_CCF_RAW, "train/data"))
    formatter = ReResultFormatter()
    formatter.combine_all()
    formatter.trans_origin_to_raw()