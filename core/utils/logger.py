import time
import sys
import os
import json

class Logger:
    """
    日志类.
    各个head参数都可以是str, 或者是拥有__name__属性的对象(如类名, 函数名).
    每条日志的格式为: [<time_stampe>]\\t@head\\t
    """
    def __init__(self, 
            log_file_name: str, 
            default_head: str or "__name__", 
            default_mid: str, 
            console_output: bool):
        self.log_file = open(log_file_name, 'a', encoding='utf8')
        self.console_output = console_output
        self._default_mid = default_mid
        self._default_signature = self.format_signature(default_head) if default_head is not None else None

    def __del__(self):
        self.log_file.close()
    
    @staticmethod
    def get_time_stampe():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    @staticmethod
    def format_signature(signature) -> str or None:
        if isinstance(signature, str):
            return signature
        else:
            return signature.__name__

    def log_message(self, *msg:"can to str", head:str or "__name__"=None, mid:str=None, end:str='\n'):
        time_stampe = self.get_time_stampe()
        total_msg = '[' + time_stampe + ']\t'
        if head != None:
            fmt_signature = self.format_signature(head)
            total_msg += '@' + fmt_signature + ':\t'
        elif self._default_signature != None:
            total_msg += '@' + self._default_signature + ':\t'
        if mid is None:
            content = self._default_mid.join(str(m) for m in msg)
        else:
            content = mid.join(str(m) for m in msg)
        total_msg += content + end
        if self.console_output:
            print(total_msg, end='')
        self.log_file.write(total_msg)

"""
获取Main.py所在目录
"""
def get_top_dir():
    tmp = os.path.abspath(__file__)
    for _ in range(3):
        tmp = os.path.dirname(tmp)
    return tmp

_default_logger = None

"""
创建一个Logger.
当log_file_name
"""
def alloc_logger(log_file_name: str=None, default_head: str or "__name__"=None, default_mid:str='',console_output:bool=True):
    global _default_logger
    if log_file_name == None:
        if _default_logger is None:
            setting_file_name = os.path.dirname(os.path.abspath(__file__)) + "/default_logger_setting.json"
            with open(setting_file_name, 'r', encoding='utf8') as f:
                setting = json.load(f)
            log_file_name = get_top_dir() + '/' + setting["default_logfile_dir"]
            signature = setting["default_head"]
            mid = setting["default_mid"]
            need_console = True if setting["default_need_console"] == "true" else False
            _default_logger = Logger(log_file_name, signature, mid, need_console)
        return _default_logger
    return Logger(log_file_name, default_head, default_mid, console_output)
        


if __name__ == "__main__":
    logger1 = alloc_logger()
    logger1.log_message("test logger")
    logger1.log_message("test string signature", head=__file__)
    logger1.log_message("test class signature", head=Logger)
    logger1.log_message("test func signature", head=alloc_logger)
    logger1.log_message("test", " multi", " msg")
    logger1.log_message("test", "multi", "msg", "with", "mid", mid=' ')