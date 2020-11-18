import re
import os
import sys
from ..config.DefaultConfig import DefaultConfig
from .label_file_reader import LabelFileReader
from .label_file_reader import LabelInfo
from ..utils import alloc_logger
from ..utils.logger import Logger


class DuplicationDetector:
    def __init__(self, raw_data_set_dir: str = None, duplication_info_dir: str = None, dup_cleaned_set_dir: str = None, console_output: bool=True):
        self.raw_data_set_dir = raw_data_set_dir if raw_data_set_dir is not None else DefaultConfig.PATHS.DATA_CCF_RAW
        self.duplication_info_dir = duplication_info_dir if duplication_info_dir is not None else DefaultConfig.PATHS.DATA_INFO
        self.dup_cleaned_set_dir = dup_cleaned_set_dir if dup_cleaned_set_dir is not None else DefaultConfig.PATHS.DATA_CCF_DBG + "/duplication_cleaned"
        self.duplication_list = []  # list of tuple
        self.logger = alloc_logger("dup_detector.log", DuplicationDetector, console_output=console_output)

    def detect(self):
        raw_data_count = len(os.listdir(self.raw_data_set_dir + "/train/data"))
        self.logger.log_message("detect():", "raw data count:\t", raw_data_count)
        reader = LabelFileReader()
        with open(self.duplication_info_dir + '/duplication.md', 'w', encoding='utf8') as dupfile:
            dupfile.write("# duplications\n\n")
            dup_count = 0
            for i in range(raw_data_count):
                data_file_name = self.raw_data_set_dir + "/train/data/" + str(i) + ".txt"
                label_file_name = self.raw_data_set_dir + "/train/label/" + str(i) + ".csv"
                with open(label_file_name, 'r', encoding='utf8') as f:
                    infos = reader.load(f)
                infos.sort(key=lambda info: info.Pos_b)
                # print(infos)
                for j in range(len(infos) - 1):
                    if infos[j].Pos_e >= infos[j + 1].Pos_b:
                        self.logger.log_message("detect():\t", "detect duplication with id={:d}".format(infos[j].ID))
                        dup_count += 1
                        dupfile.write("---\n\n")
                        dupfile.write("## {:d}\n\n".format(dup_count))
                        dupfile.write("ID=[{:d}]\n\n".format(infos[j].ID))
                        with open(data_file_name, 'r', encoding='utf8') as f:
                            txt = f.read()
                        dupfile.write(txt + '\n\n')
                        dupfile.write("- [ ] [{:d}:{:d}]\t<{:s}>\t{:s}\n\n".format(infos[j].Pos_b, infos[j].Pos_e, infos[j].Category, infos[j].Privacy) )
                        dupfile.write("- [ ] [{:d}:{:d}]\t<{:s}>\t{:s}\n\n".format(infos[j+1].Pos_b, infos[j+1].Pos_e, infos[j+1].Category, infos[j+1].Privacy))
                # break
            self.logger.log_message("detect():\t", "detect ", dup_count, " duplications")

    @staticmethod
    def auto_clean_judge(infos: "List[LabelInfo]") -> "Set[LabelInfo]":
        """
        决议消除infos列表中重叠的标签，返回判断应当删去的标签。
        对于重叠的部分，将保留较长的一个。
        """
        
        def length(info: LabelInfo)->int:
            return info.Pos_e - info.Pos_b + 1
        
        ret = set()
        infos.sort(key=lambda info: info.Pos_b)
        for i in range(len(infos)):
            j = i + 1
            while infos[i] not in ret and j < len(infos) and infos[i].Pos_e >= infos[j].Pos_b:
                if length(infos[i]) < length(infos[j]):
                    ret.add(infos[i])
                else:
                    ret.add(infos[j])
                j += 1
        return ret


    def auto_clean_all(self):

        # def length(info: LabelInfo):
        #     return info.Pos_e - info.Pos_b + 1

        signature = "auto_clean():\t"
        raw_data_count = len(os.listdir(self.raw_data_set_dir + "/train/data"))
        self.logger.log_message("auto_clean():", "raw data count:\t", raw_data_count)
        reader = LabelFileReader()
        
        dup_count = 0
        for i in range(raw_data_count):
            clean_data_name = self.dup_cleaned_set_dir + "/auto_cleaned/" + str(i) + ".csv"
            label_file_name = self.raw_data_set_dir + "/train/label/" + str(i) + ".csv"
            with open(label_file_name, 'r', encoding='utf8') as f:
                infos = reader.load(f)
            # infos.sort(key=lambda info: info.Pos_b)
                # print(infos)
            # j = 0
            # while j < len(infos) - 1:
            #     if infos[j].Pos_e >= infos[j + 1].Pos_b:
            #         self.logger.log_message(signature, "detect duplication with id={:d}".format(infos[j].ID))
            #         dup_count += 1
            #         to_remove = infos[j + 1] if length(infos[j]) > length(infos[j + 1]) else infos[j]
            #         self.logger.log_message(signature, "remove [", to_remove, "] from label-set")
            #         infos.remove(to_remove)
            #     else:
            #         j += 1
            to_remove = self.auto_clean_judge(infos)
            if len(to_remove) != 0:
                dup_count += len(to_remove)
                self.logger.log_message(signature, "detect {:d} duplication with id={:d}".format(len(to_remove), infos[0].ID))
                self.logger.log_message(signature, "removing ", to_remove)
                for info in to_remove:
                    infos.remove(info)

            with open(clean_data_name, 'w', encoding='utf8') as f:
                reader.dump(infos, f)
        self.logger.log_message(signature, "detect ", dup_count, " duplications")

        
        
if __name__ == "__main__":
    detector = DuplicationDetector()
    detector.auto_clean_all()
    