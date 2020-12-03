from ..utils import alloc_logger
from ..config.DefaultConfig import DefaultConfig
from .label_transformer import LabelTrasformer
from .label_formatter import LabelFormatter
from .label_file_reader import LabelFileReader
from .label_file_reader import LabelInfo
import os
import json

class ResultFormatter:
    def __init__(self, end='\n'):
        super().__init__()
        
        self.logger = alloc_logger("result_formatter.log", ResultFormatter)
        self.end = end
        self.combine_index = {} # origin: (beg, target)
        with open(os.path.join(DefaultConfig.PATHS.DATA_INFO, "split_index_test.json"), 'r', encoding='utf8') as f:
            m = json.load(f)
            for item in m:
                target = item["target"]
                origin = item["origin"]
                beg = item["beg"]
                if origin in self.combine_index.keys():
                    self.combine_index[origin].append((beg, target))
                else:
                    self.combine_index[origin] = [(beg, target)]

    def combine_all(self, origin_data_count: int=-1, label_dir: str=None, data_dir: str=None):
        signature = "conbine_all()\t"

        self.logger.log_message(signature, "start!")

        label_dir = label_dir if label_dir is not None else os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "test/label")
        data_dir = data_dir if data_dir is not None else os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "test/data")
        self.logger.log_message(signature, "origin data dir:\t", data_dir)
        self.logger.log_message(signature, "origin label dir:\t", label_dir)

        origin_data_count = origin_data_count if origin_data_count >= 0 else len(os.listdir(os.path.join(DefaultConfig.PATHS.DATA_CCF_RAW, "test")))
        self.logger.log_message(signature, "origin data count:\t", origin_data_count)


        label_formatter = LabelFormatter()
        label_formatter.load_transformer_from_file()

        reader = LabelFileReader()

        output_csv = open(os.path.join(DefaultConfig.PATHS.DATA_INFO, "predict_origin.csv"), 'w', encoding='utf8')

        for i in range(origin_data_count):
            # labels = None
            with open(os.path.join(label_dir, "{:d}.json".format(i)), 'r', encoding='utf8') as f:
                labels = json.load(f)
            with open(os.path.join(data_dir, "{:d}.txt".format(i)), 'r', encoding='utf8') as f:
                data = f.read()
            if i in self.combine_index.keys():
                targets = self.combine_index[i]
                targets.sort(key=lambda t: t[0])
                for _, target in targets:
                    with open(os.path.join(label_dir, "{:d}.json".format(i))) as f:
                        new_labels = json.load(f)
                    with open(os.path.join(data_dir, "{:d}.txt".format(i))) as f:
                        new_data = f.read()
                    labels += new_labels
                    data += new_data
            infos = label_formatter.integer_list_label_and_data_to_infos(ID=i, integer_list=labels, data = data)
            for info in infos:
                string = reader.dumps(info)
                output_csv.write(string + self.end)
        output_csv.close()

    def trans_origin_to_raw(self):
        input_csv = open(os.path.join(DefaultConfig.PATHS.DATA_INFO, "predict_origin.csv"), 'r', encoding='utf8')
        output_csv = open(os.path.join(DefaultConfig.PATHS.DATA_INFO, "predict.csv"), 'w', encoding='utf8')

        reader = LabelFileReader()

        for line in input_csv.readlines():
            info = reader.loads(line)
            content = info.Privacy
            beg = info.Pos_b
            end = info.Pos_e
            ID = info.ID
            with open(os.path.join(DefaultConfig.PATHS.DATA_CCF_RAW, "/test/{:d}.txt".format(ID))) as f:
                raw_content = f.read()
            new_beg = raw_content.find(content, beg - 5, end + 5)
            new_end = new_beg + end - beg
            new_info = LabelInfo(
                ID = ID,
                Category = info.Category,
                Pos_b = new_beg,
                Pos_e = new_end,
                Privacy = content
            )
            new_line = reader.dumps(new_info)
            output_csv.write(new_line + self.end)

        input_csv.close()
        output_csv.close()

if __name__ == "__main__":
    formatter = ResultFormatter()
    formatter.combine_all(
        origin_data_count=2515, 
        label_dir=os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train/label"),
        data_dir=os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train/data"))
    formatter.trans_origin_to_raw()
