from ..config.DefaultConfig import DefaultConfig
from ..utils import alloc_logger
from .duplication_detector import DuplicationDetector
from .label_file_reader import LabelFileReader
from .label_formatter import LabelFormatter
from .mismatch_detector import MismatchDetector
from .label_transformer import LabelTrasformer
from .divider import Divider
import os
import json
import random


class RePreprocessor:
    def __init__(self, origin_dir: str = None, target_dir: str = None):
        super().__init__()
        self.origin_dir = origin_dir if origin_dir is not None else DefaultConfig.PATHS.DATA_CCF_RAW
        self.target_dir = target_dir if target_dir is not None else DefaultConfig.PATHS.DATA_CCF_CLEANED
        self.duplication_detector = DuplicationDetector()
        self.reader = LabelFileReader()
        self.label_formatter = LabelFormatter()
        self.mismatch_detector = MismatchDetector()
        self.logger = alloc_logger("re_reprocessor.log", RePreprocessor)
        self.trasformer = self.label_formatter.fit(self.origin_dir + "/train/label")

    def produce_train(self, max_size: int = None):
        """
        如果不指定 max_size 或指定为None, 将不会对原始样本进行分割, 
        如果传入了某个整数, 将会把长度大于 max_size 的文本按照尽可能靠后的句号拆分为长度小于 max_size 的若干段.
        拆分索引将保存至 data/info/split_index_train.json.
        @parm max_size: 单个输入样例的最大长度.
        """
        signature = "produce_train():\t"
        self.logger.log_message(signature, "maxsize=", max_size)
        self.logger.log_message(signature, "start!")

        origin_data_count = len(os.listdir(self.origin_dir + "/train/data"))
        self.logger.log_message(signature, "origin data count:\t", origin_data_count)
        file_name = os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train_origin_bio.txt")
        ofs = open(file_name, 'w', encoding='utf8')

        reader = LabelFileReader()
        dup_count = 0
        unsolve_mismatch = []
        origin_total_entity_count = 0
        cleaned_entity_count = 0

        remove_vx_count = 0
        remove_email_count = 0
        remove_mobile_count = 0
        remove_QQ_count = 0

        for i in range(origin_data_count):
            with open(self.origin_dir + "/train/label/{:d}.csv".format(i), 'r', encoding='utf8') as f:
                infos = reader.load(f)
            with open(self.origin_dir + "/train/data/{:d}.txt".format(i), 'r', encoding='utf8') as f:
                data = f.read()

            data, infos = self.mismatch_detector.fix_mismatch(data, infos)

            if data is None or infos is None:
                unsolve_mismatch.append(i)
                continue
            origin_total_entity_count += len(infos)

            new_infos = [info for info in infos if info.Category != 'vx']
            remove_vx_count += len(infos) - len(new_infos)
            infos = new_infos
            
            new_infos = [info for info in infos if info.Category != 'email']
            remove_email_count += len(infos) - len(new_infos)
            infos = new_infos
            
            new_infos = [info for info in infos if info.Category != 'mobile']
            remove_mobile_count += len(infos) - len(new_infos)
            infos = new_infos
            
            new_infos = [info for info in infos if info.Category != 'QQ']
            remove_QQ_count += len(infos) - len(new_infos)
            infos = new_infos

            to_remove = self.duplication_detector.auto_clean_judge(infos)

            for info in to_remove:
                infos.remove(info)
                dup_count += 1
                self.logger.log_message(signature, "[{:d}]\tremoving dup\t".format(i), reader.dumps(info))

            cleaned_entity_count += len(infos)

            labels = self.label_formatter.infos_to_bio_str_list_label(infos, len(data))

            for idx, ch in enumerate(data):
                ofs.write(ch + ' ' + labels[idx] + '\n')
            ofs.write("\n")
        
        ofs.close()
        self.logger.log_message(signature, "origin file count={:d}".format(origin_data_count))
        self.logger.log_message(signature, "save result in file:", file_name)
        self.logger.log_message(signature, "origin entity count=", origin_total_entity_count)
        self.logger.log_message(signature, "remove vx ", remove_vx_count, " times")
        self.logger.log_message(signature, "remove email ", remove_email_count, " times")
        self.logger.log_message(signature, "remove mobile ", remove_mobile_count, " times")
        self.logger.log_message(signature, "remove QQ ", remove_QQ_count, " times")
        self.logger.log_message(signature, "remove duplication {:d} times".format(dup_count))
        self.logger.log_message(signature, "cleaned entity count=", cleaned_entity_count)
        self.logger.log_message(signature, "detect {:d} unsolved mismatch".format(len(unsolve_mismatch)))
        # self.logger.log_message(signature, "output file count={:d}".format(alloc_file_num))
        if len(unsolve_mismatch) != 0:
            self.logger.log_message(signature, "their ID are:")
            for unsolve_id in unsolve_mismatch:
                self.logger.log_message(signature, "\t{:d}".format(unsolve_id))

        self.mismatch_detector.save()
        self.logger.log_message(signature, "finish!")

    def divide_train_by_index(self, test_start_index: int):
        signature = "divide_train_by_index()\t"
        origin_file_name = os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train_origin_bio.txt")
        train_file_name = os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train_train_bio.txt")
        test_file_name = os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train_test_bio.txt")
        self.logger.log_message(signature, "start!")
        self.logger.log_message(signature, origin_file_name)
        self.logger.log_message(signature, "\t|")
        self.logger.log_message(signature, "\t+ -[train]- ->", train_file_name)
        self.logger.log_message(signature, "\t+ -[test ]- ->", test_file_name)
        with open(origin_file_name, 'r', encoding='utf8') as f:
            samples = f.read().strip().split('\n\n')
        train_ofs = open(train_file_name, 'w', encoding='utf8')
        test_ofs = open(test_file_name, 'w', encoding='utf8')
        train_count = 0
        test_count = 0
        total_count = 0
        for i, sample in enumerate(samples):
            sample = sample.replace('\r', '')
            if i < test_start_index:
                train_ofs.write(sample + "\n\n")
                train_count += 1
            else:
                test_ofs.write(sample + "\n\n")
                test_count += 1
            
            total_count += 1
        train_ofs.close()
        test_ofs.close()
        self.logger.log_message(signature, "train count=", train_count)
        self.logger.log_message(signature, "test count=", test_count)
        self.logger.log_message(signature, "total count=", total_count)
        self.logger.log_message(signature, "finish!")



    def divide_train(self, train_rate: float=0.8):
        signature = "divide_train()\t"
        origin_file_name = os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train_origin_bio.txt")
        train_file_name = os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train_train_bio.txt")
        test_file_name = os.path.join(DefaultConfig.PATHS.DATA_CCF_CLEANED, "train_test_bio.txt")
        self.logger.log_message(signature, "start!")
        self.logger.log_message(signature, origin_file_name)
        self.logger.log_message(signature, "\t|")
        self.logger.log_message(signature, "\t+ -[train]- ->", train_file_name)
        self.logger.log_message(signature, "\t+ -[test ]- ->", test_file_name)
        with open(origin_file_name, 'r', encoding='utf8') as f:
            samples = f.read().strip().split('\n\n')
        train_ofs = open(train_file_name, 'w', encoding='utf8')
        test_ofs = open(test_file_name, 'w', encoding='utf8')
        train_count = 0
        test_count = 0
        total_count = 0
        for sample in samples:
            sample = sample.replace('\r', '')
            if random.random() < train_rate:
                train_ofs.write(sample + "\n\n")
                train_count += 1
            else:
                test_ofs.write(sample + "\n\n")
                test_count += 1
            
            total_count += 1
        train_ofs.close()
        test_ofs.close()
        self.logger.log_message(signature, "train count=", train_count)
        self.logger.log_message(signature, "test count=", test_count)
        self.logger.log_message(signature, "total count=", total_count)
        self.logger.log_message(signature, "finish!")

    def produce_test(self, max_size: int = None):
        """
        如果不指定 max_size 或指定为None, 将不会对原始样本进行分割, 
        如果传入了某个整数, 将会把长度大于 max_size 的文本按照尽可能靠后的句号拆分为长度小于 max_size 的若干段.
        拆分索引将保存至  data/info/split_index_test.json.
        @parm max_size: 单个输入样例的最大长度.
        """
        signature = "produce_test():\t"
        self.logger.log_message(signature, "start!")
        self.logger.log_message(signature, "maxsize=", max_size)

        origin_data_count = len(os.listdir(self.origin_dir + "/test"))
        alloc_file_num = origin_data_count
        self.logger.log_message(signature, "origin data count:\t", origin_data_count)

        if max_size is not None:
            divider = Divider(max_size)
            divide_index = []  # [{"target": target_id, "origin": origin_id, "beg": beg_index}]

        count = 0
        for i in range(origin_data_count):
            with open(self.origin_dir + "/test/{:d}.txt".format(i), 'r', encoding='utf8') as f:
                data = f.read()

            new_data = self.mismatch_detector.remove_special_char(data)
            if len(data) != len(new_data):
                count += 1

            data = new_data

            # divide & save
            if max_size is not None:
                divide_result = divider.detect_division(data)
                if len(divide_result) > 1:
                    self.logger.log_message(signature, "[{:d}]\tlen={:d}".format(i, len(data)))
                    # self.logger.log_message(signature, "[{:d}]\tdivide points:\t".format(i), divide_result)
                for j in range(len(divide_result)):
                    beg = divide_result[j]
                    end = divide_result[j + 1] if j < len(divide_result) - 1 else -1
                    target = i
                    if j != 0:
                        target = alloc_file_num
                        alloc_file_num += 1
                        divide_index.append({"target": target, "origin": i, "beg": beg})
                    if len(divide_result) > 1:
                        self.logger.log_message(signature, "[{:d}]\t".format(i),
                                                "({:3d}:{:3d})->[{:d}]".format(beg, end, target))
                    with open(self.target_dir + "/test/data/{:d}.txt".format(target), 'w', encoding='utf8') as f:
                        f.write(data[beg: end])

            else:
                with open(self.target_dir + "/test/data/{:d}.txt".format(i), 'w', encoding='utf8') as f:
                    f.write(data)

        if max_size is not None:
            split_index_file_name = os.path.join(DefaultConfig.PATHS.DATA_INFO, "split_index_test.json")
            self.logger.log_message(signature, "saving split index in file[", split_index_file_name, ']')
            with open(split_index_file_name, 'w', encoding='utf8') as f:
                json.dump(divide_index, f)
        self.logger.log_message(signature, "changed {:d} data files!".format(count))
        self.logger.log_message(signature, "finish!")


def quick_preproduce(max_size: int, train_rate: float=1, origin_dir: str = None, target_dir: str = None, test_start_index: int=-1) -> LabelTrasformer:
    """
    封装好的快速进行预处理的函数.
    如果不指定 max_size 或指定为None, 将不会对原始样本进行分割, 
    如果传入了某个整数, 将会把长度大于 max_size 的文本按照尽可能靠后的句号拆分为长度小于 max_size 的若干段.
    拆分索引将分别保存至 data/info/split_index_train.json 和 data/info/split_index_test.json.
    @parm max_size: 单个输入样例的最大长度.
    """
    logger = alloc_logger()
    if target_dir is None:
        try:
            new_dir = DefaultConfig.PATHS.DATA_CCF_CLEANED + "/test/data"
            logger.log_message("mkdir " + new_dir)
            os.makedirs(new_dir)
        except FileExistsError:
            logger.log_message("has existed")
        try:
            new_dir = DefaultConfig.PATHS.DATA_CCF_CLEANED + "/test/label"
            logger.log_message("mkdir " + new_dir)
            os.makedirs(new_dir)
        except FileExistsError:
            logger.log_message("has existed")
        try:
            new_dir = DefaultConfig.PATHS.DATA_CCF_CLEANED + "/train/data"
            logger.log_message("mkdir " + new_dir)
            os.makedirs(new_dir)
        except FileExistsError:
            logger.log_message("has existed")
        try:
            new_dir = DefaultConfig.PATHS.DATA_CCF_CLEANED + "/train/label"
            logger.log_message("mkdir " + new_dir)
            os.makedirs(new_dir)
        except FileExistsError:
            logger.log_message("has existed")
    re_reprocessor = RePreprocessor(origin_dir=origin_dir, target_dir=target_dir)
    re_reprocessor.produce_train()
    if test_start_index < 0:
        re_reprocessor.divide_train(train_rate)
    else:
        re_reprocessor.divide_train_by_index(test_start_index=test_start_index)
    re_reprocessor.produce_test(max_size)
    re_reprocessor.trasformer.save_to_file()
    re_reprocessor.trasformer.log_bio_type_to_file()
    return re_reprocessor.trasformer


if __name__ == "__main__":
    from core.config.DefaultConfig import DefaultConfig as config
    quick_preproduce(max_size=256 - 2, test_start_index=13263)
