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
    def __init__(self, split_index_file: str = None, end='\n'):
        super().__init__()

        self.logger = alloc_logger(
            "re_result_formatter.log", ReResultFormatter)
        self.end = end
        email_pattern = r"[a-zA-Z0-9_\.]+@[a-zA-Z0-9_\.]+"
        mobile_pattern = r"[\+＋]?([\(（]\d+[\)）])?\d[\-\d]*\d([转轉]\d+)?"
        QQ_pattern = r"\d{9,10}"
        date_pattern = r"([12]\d{3}\-\d{1,2}-\d{1,2})|([12]\d{3}\-[12]\d{3})"
        self.email_re = re.compile(email_pattern)
        self.mobile_re = re.compile(mobile_pattern)
        self.QQ_re = re.compile(QQ_pattern)
        self.date_re = re.compile(date_pattern)
        self.combine_index = {}  # origin: (beg, target)
        split_index_file = split_index_file if split_index_file is not None else os.path.join(
            DefaultConfig.PATHS.DATA_INFO, "split_index_test.json")
        self.logger.log_message(
            "loading split_index from file:\t", split_index_file)

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

    def combine_all(self, receive_rate: float = 0.8, result_dir: str = None):
        signature = "conbine_all()\t"

        self.logger.log_message(signature, "start!")

        result_dir = result_dir if result_dir is not None else os.path.join(
            DefaultConfig.PATHS.DATA_CCF_CLEANED, "test/label")
        self.logger.log_message(signature, "origin result dir:\t", result_dir)

        def file_num_to_tokens_and_labels(num: int) -> (list, list):
            with open(os.path.join(result_dir, "{:d}.json".format(num)), 'r', encoding='utf8') as f:
                result = json.load(f)
                return result["data"][1:-1], result["tags"][1:-1]

        origin_result_count = len(os.listdir(
            os.path.join(DefaultConfig.PATHS.DATA_CCF_RAW, "test")))
        self.logger.log_message(
            signature, "origin result count:\t", origin_result_count)

        label_formatter = LabelFormatter()
        label_formatter.load_transformer_from_file()

        reader = LabelFileReader()

        output_csv = open(os.path.join(
            DefaultConfig.PATHS.DATA_INFO, "predict_origin.csv"), 'w', encoding='utf8')

        label_token_len_mismatch = []
        for i in range(origin_result_count):
            # labels = None
            tokens, labels = file_num_to_tokens_and_labels(i)
            if i in self.combine_index.keys():
                targets = self.combine_index[i]
                targets.sort(key=lambda t: t[0])
                for _, target in targets:
                    new_tokens, new_labels = file_num_to_tokens_and_labels(
                        target)
                    labels += new_labels
                    tokens += new_tokens
            if len(tokens) != len(labels):
                label_token_len_mismatch.append(i)
                continue
            infos = label_formatter.bio_str_list_label_and_token_list_to_infos(
                ID=i, bio_str_list=labels, tokens=tokens, receive_rate=receive_rate)
            for info in infos:
                string = reader.dumps(info)
                output_csv.write(string + self.end)
        output_csv.close()

        self.logger.log_message("mismatches:\t", label_token_len_mismatch)

    def detect_email(self, ID: int, data: str) -> "List[LabelInfo]":
        signature = "detect_email()\t"
        ret = []
        for m in self.email_re.finditer(data):
            ret.append(LabelInfo(
                ID=ID,
                Category="email",
                Pos_b=m.start(),
                Pos_e=m.end() - 1,
                Privacy=m.group()
            ))
        if len(ret) != 0:
            # self.logger.log_message(signature, "[{:d}] found {:d} emails by re".format(ID, len(ret)))
            # self.logger.log_message(ret)
            pass
        return ret

    def detect_mobile_and_QQ(self, ID: int, data: str) -> "List[LabelInfo]":
        signature = "detect_mobile_and_QQ()\t"
        ret = []
        for m in self.mobile_re.finditer(data):
            content = m.group()
            if len(content) < 8:
                continue
            if self.date_re.fullmatch(content):
                continue
            clz = "mobile"
            if self.QQ_re.fullmatch(content):
                clz = "QQ"
            ret.append(LabelInfo(
                ID=ID,
                Category=clz,
                Pos_b=m.start(),
                Pos_e=m.end() - 1,
                Privacy=m.group()
            ))
        if len(ret) != 0:
            self.logger.log_message(signature, "[{:d}] found {:d} QQ/mobile by re".format(ID, len(ret)))
            self.logger.log_message(ret)
        return ret

    def _print_infos_to_csv_for_id(self, ID: int, infos: list, csv_ofs, data_dir: str, detect_email=True) -> (int, int, int):
        with open(os.path.join(data_dir, "{:d}.txt".format(ID)), 'r', encoding='utf8') as f:
            raw_content = f.read()
        reader = LabelFileReader()
        signature = "_print_infos_to_csv_for_id()\t"
        not_in_raw_count = 0
        head = 0
        to_print_infos = []
        for info in infos:
            content = info.Privacy

            current_content = raw_content[head:].lower()
            if content in current_content:
                new_beg = head + current_content.find(content)
                new_end = new_beg + len(content) - 1
                head = new_end + 1
                to_print_infos.append(LabelInfo(
                    ID=ID,
                    Category=info.Category,
                    Pos_b=new_beg,
                    Pos_e=new_end,
                    Privacy=content
                ))
                continue
            pattern = content               \
                .replace('\\', '\\\\')      \
                .replace('(', '\\(')        \
                .replace(')', '\\)')        \
                .replace('?', '\\?')        \
                .replace('.', '\\.')        \
                .replace('*', '\\*')        \
                .replace('+', '\\+')        \
                .replace("[UNK]", ".+?")    \
                .replace('[', '\\[')        \
                .replace(']', '\\]')        \
                .replace('{', '\\{')        \
                .replace('}', '\\}')
            m = re.search(pattern, current_content)
            if m is not None:
                new_beg = head + m.start()
                new_content = m.group()
                new_end = new_beg + len(new_content) - 1
                head = new_end + 1
                to_print_infos.append(LabelInfo(
                    ID=ID,
                    Category=info.Category,
                    Pos_b=new_beg,
                    Pos_e=new_end,
                    Privacy=new_content
                ))
                continue
            not_in_raw_count += 1
            if not_in_raw_count == 1:
                self.logger.log_message(
                    signature, "[{:d}] found mismatch(es):".format(ID))
            self.logger.log_message(signature, "pattern=", pattern)
            self.logger.log_message(signature, '\t', reader.dumps(info))

        for info in to_print_infos:
            csv_ofs.write(reader.dumps(info) + '\n')
        email_count = 0
        mobile_QQ_count = 0
        if detect_email:
            for info in self.detect_email(ID, raw_content):
                email_count += 1
                csv_ofs.write(reader.dumps(info) + '\n')
            for info in self.detect_mobile_and_QQ(ID, raw_content):
                mobile_QQ_count += 1
                csv_ofs.write(reader.dumps(info) + '\n')

        if not_in_raw_count != 0:
            self.logger.log_message(
                signature, "[{:d}] detect {:d} info not in raw".format(ID, not_in_raw_count))

        return not_in_raw_count, email_count, mobile_QQ_count

    def trans_origin_to_raw(self, data_dir: str = None, detect_email=True):
        signature = "trans_origin_to_raw()\t"

        input_csv = open(os.path.join(DefaultConfig.PATHS.DATA_INFO,
                                      "predict_origin.csv"), 'r', encoding='utf8')
        output_csv = open(os.path.join(
            DefaultConfig.PATHS.DATA_INFO, "predict_raw.csv"), 'w', encoding='utf8')

        reader = LabelFileReader()

        data_dir = data_dir if data_dir is not None else os.path.join(
            DefaultConfig.PATHS.DATA_CCF_RAW, "test")

        not_in_raw_count = 0
        email_count = 0
        mobile_QQ_count = 0
        current_id = 0
        infos = []
        for line in input_csv.readlines():
            info = reader.loads(line)
            if (info.ID == current_id):
                infos.append(info)
                continue
            not_in_raw_count_delta, email_count_delta, mobile_QQ_count_delta =  \
                self._print_infos_to_csv_for_id(
                    ID=current_id,
                    infos=infos,
                    csv_ofs=output_csv,
                    data_dir=data_dir,
                    detect_email=detect_email)
            not_in_raw_count += not_in_raw_count_delta
            email_count += email_count_delta
            mobile_QQ_count += mobile_QQ_count_delta

            current_id = info.ID
            infos = [info]

        not_in_raw_count_delta, email_count_delta, mobile_QQ_count_delta =  \
            self._print_infos_to_csv_for_id(
                ID=current_id,
                infos=infos,
                csv_ofs=output_csv,
                data_dir=data_dir,
                detect_email=detect_email)
        not_in_raw_count += not_in_raw_count_delta
        email_count += email_count_delta
        mobile_QQ_count += mobile_QQ_count_delta

        self.logger.log_message(
            signature, "not in raw count=", not_in_raw_count)
        self.logger.log_message(signature, "reg email count=", email_count)
        self.logger.log_message(signature, "reg mobile&QQ count=", mobile_QQ_count)
        self.logger.log_message(signature, "finish")
        input_csv.close()
        output_csv.close()

    def final_format(self):
        signature = "final_format()\t"
        self.logger.log_message(signature, "start")
        with open(os.path.join(DefaultConfig.PATHS.DATA_INFO, "predict_raw.csv"), 'r', encoding='utf8') as f:
            lines = f.read().splitlines()
            self.logger.log_message(
                signature, "total entity count=", len(lines))
            all_content = list(set(lines))
            self.logger.log_message(
                signature, "unique entity count=", len(all_content))

        def get_beg(line: str) -> int:
            ID, clz, beg, end, content = tuple(line.split(','))
            return int(beg)

        def get_end(line: str) -> int:
            ID, clz, beg, end, content = tuple(line.split(','))
            return int(end)

        def get_id(line: str) -> int:
            ID, clz, beg, end, content = tuple(line.split(','))
            return int(ID)

        def is_single(line: str) -> int:
            ID, clz, beg, end, content = tuple(line.split(','))
            return int(beg) == int(end)

        def is_mobile(line: str) -> int:
            ID, clz, beg, end, content = tuple(line.split(','))
            return clz == "mobile"

        def is_QQ(line: str) -> int:
            ID, clz, beg, end, content = tuple(line.split(','))
            return clz == "QQ"

        def get_content(line: str) -> int:
            ID, clz, beg, end, content = tuple(line.split(','))
            return content
        all_content.sort(key=get_end)
        all_content.sort(key=get_beg)
        all_content.sort(key=get_id)
        self.logger.log_message(signature, "length-1 entity count=",
                                sum(1 for line in all_content if is_single(line)))
        for line in all_content:
            if (is_mobile(line)):
                self.logger.log_message(
                    signature, "[mobiles]\t", get_content(line))
        for line in all_content:
            if (is_QQ(line)):
                self.logger.log_message(signature, "[QQ]\t", get_content(line))
        with open(os.path.join(DefaultConfig.PATHS.DATA_INFO, "predict.csv"), 'w', encoding='utf8') as f:
            f.write('ID,Category,Pos_b,Pos_e,Privacy\n')
            for content in all_content:
                f.write(content + "\n")


if __name__ == "__main__":
    # formatter = ReResultFormatter(os.path.join(DefaultConfig.PATHS.DATA_INFO, "split_index_train.json"))
    # formatter.combine_all(
    #    origin_data_count=2515,
    #    label_dir=os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train/label"),
    #    data_dir=os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train/data"))
    # formatter.trans_origin_to_raw(data_dir=os.path.join(DefaultConfig.PATHS.DATA_CCF_RAW, "train/data"))
    formatter = ReResultFormatter()
    formatter.combine_all(result_dir=os.path.join(
        DefaultConfig.PATHS.DATA_CCF_CLEANED, "test/label/20"))
    formatter.trans_origin_to_raw()
    formatter.final_format()
