import time
import sys
import os
import json
from ..config.DefaultConfig import DefaultConfig



_default_logger = None
total_file_reference = 0
total_file = None
file_pool = {} # filename: (file, reference)


class Logger:
    """
    日志类.
    各个head参数都可以是str, 或者是拥有__name__属性的对象(如类名, 函数名).
    每条日志的格式为: [<time_stampe>]\\t@head\\tq
    """

    def __init__(self, 
            log_file_name: str, 
            default_head: str or "__name__", 
            default_mid: str, 
            console_output: bool):
        
        # 引用全局日志文件
        global total_file_reference
        global total_file
        total_file_reference += 1
        if total_file is None or total_file.closed:
            total_file = open(os.path.join(DefaultConfig.PATHS.LOG, "total.log"), 'a', encoding='utf8')
        
        global file_pool
        if log_file_name in file_pool:
            self._log_file = file_pool[log_file_name][0] 
            file_pool[log_file_name][1] += 1
            # print("using log_file in pool [", log_file_name, ']')
        else:
            self._log_file = open(os.path.join(DefaultConfig.PATHS.LOG, log_file_name), 'a', encoding='utf8')
            file_pool[log_file_name] = [self._log_file, 1]
            # print("opening new log_file [", log_file_name, ']')

        self._log_file_name = log_file_name
        self.console_output = console_output
        self._default_mid = default_mid
        self._default_signature = self.format_signature(default_head) if default_head is not None else None

    def __del__(self):
        # print("deleting logger [file=", self._log_file_name, ", head=", self._default_signature, "]")

        global file_pool
        file_pool[self._log_file_name][1] -= 1
        if file_pool[self._log_file_name][1] == 0:
            self._log_file.write("\n\n\n")
            self._log_file.close()
            # print("closing log_file [", self._log_file_name, ']')
            del file_pool[self._log_file_name]

        # 解除全局日志文件的引用
        global total_file_reference
        global total_file
        
        total_file_reference -= 1
        if total_file_reference == 0:
            total_file.close()
            # print("closing total_log_file")
        

    
    @staticmethod
    def get_time_stampe():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    
    @staticmethod
    def get_fs_legal_time_stampe():
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    @staticmethod
    def format_signature(signature) -> str or None:
        if isinstance(signature, str):
            return signature
        else:
            return signature.__name__

    def format_msg(self, *msg:"can to str", head:str or "__name__"=None, mid:str=None, end:str='\n'):
        """
        当 head 与 mid 为 None 时, 将使用创建本logger时指定的默认值, 若默认值仍为None, 则为空字符串.
        不会在结尾添加 '\\n'.

        @param *msg: 一列可以通过str()转换为字符串的对象, 将通过mid属性连接;
        @param head: 头部, 以 @xxx 形式添加到时间戳之后, head需要是一个字符串或者拥有__name__属性的对象;
        @param mid: 连接 msg 各个内容的连接符;
        """
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
        total_msg += content
        return total_msg

    def console_message(self, *msg:"can to str", head:str or "__name__"=None, mid:str=None, end:str='\n'):
        """
        向stdout输出信息.
        将转发 msg, head, mid 至 format_msg() 函数进行格式化, 具体如下:
        当 head 与 mid 为 None 时, 将使用创建本logger时指定的默认值, 若默认值仍为None, 则为空字符串

        @param *msg: 一列可以通过str()转换为字符串的对象, 将通过mid属性连接;
        @param head: 头部, 以 @xxx 形式添加到时间戳之后, head需要是一个字符串或者拥有__name__属性的对象;
        @param mid: 连接 msg 各个内容的连接符;
        @param end: 结尾的符号, 仅对console内容有效, 写入日志文件时必定以回车结尾;
        """
        total_msg = self.format_msg(*msg, head=head, mid=mid)
        print(total_msg, end=end)

    def file_message(self, *msg:"can to str", head:str or "__name__"=None, mid:str=None, need_total:bool=True):
        """
        向日志文件写入信息.
        将转发 msg, head, mid 至 format_msg() 函数进行格式化, 具体如下:
        当 head 与 mid 为 None 时, 将使用创建本logger时指定的默认值, 若默认值仍为None, 则为空字符串

        @param *msg: 一列可以通过str()转换为字符串的对象, 将通过mid属性连接;
        @param head: 头部, 以 @xxx 形式添加到时间戳之后, head需要是一个字符串或者拥有__name__属性的对象;
        @param mid: 连接 msg 各个内容的连接符;
        @param end: 结尾的符号, 仅对console内容有效, 写入日志文件时必定以回车结尾;
        @param need_total: 是否需要向全局日志文件写入;
        """
        global total_file
        total_msg = self.format_msg(*msg, head=head, mid=mid)
        self._log_file.write(total_msg + '\n')
        if need_total:
            total_file.write(total_msg + '\n')

    def log_message(self, *msg:"can to str", head:str or "__name__"=None, mid:str=None, end:str='\n'):     
        """
        向日志文件写入信息, 如果创建Logger时console_output为True, 则同时向stdout输出相同的信息.
        将转发 msg, head, mid 至 format_msg() 函数进行格式化, 具体如下:
        当 head 与 mid 为 None 时, 将使用创建本logger时指定的默认值, 若默认值仍为None, 则为空字符串

        @param *msg: 一列可以通过str()转换为字符串的对象, 将通过mid属性连接;
        @param head: 头部, 以 @xxx 形式添加到时间戳之后, head需要是一个字符串或者拥有__name__属性的对象;
        @param mid: 连接 msg 各个内容的连接符;
        @param end: 结尾的符号, 仅对console内容有效, 写入日志文件时必定以回车结尾;
        """
        global total_file
        total_msg = self.format_msg(*msg, head=head, mid=mid)
        if self.console_output:
            print(total_msg, end=end)
        self._log_file.write(total_msg + '\n')
        total_file.write(total_msg + '\n')




def alloc_logger(log_file_name: str=None, default_head: str or "__name__"=None, default_mid:str='',console_output:bool=True):
    """
    创建一个Logger.
    当log_file_name为空时, 将返回默认的logger, 该logger只有一个实例.
    log_file_name 是相对于 log 文件夹的目录
    """
    global _default_logger
    if log_file_name == None:
        if _default_logger is None:
            log_file_name = DefaultConfig.LOG.DEFAULT_LOG_DIR
            signature = DefaultConfig.LOG.DEFAULT_HEAD
            mid = DefaultConfig.LOG.DEFAULT_MID
            need_console = DefaultConfig.LOG.DEFAULT_NEED_CONSOLE
            _default_logger = Logger(log_file_name, signature, mid, need_console)
        return _default_logger
    ret = Logger(log_file_name, default_head, default_mid, console_output)
    return ret
     
def log_message(*msg:"can to str", head:str or "__name__"=None, mid:str=None, end:str='\n'):   
    """
    代理默认logger的log_message.
    """
    alloc_logger().log_message(*msg, head=head, mid=mid, end=end)

def file_message(*msg:"can to str", head:str or "__name__"=None, mid:str=None):
    """
    file_message.
    """
    alloc_logger().file_message(*msg, head=head, mid=mid)

def console_message(*msg:"can to str", head:str or "__name__"=None, mid:str=None, end:str='\n'):
    """
    代理默认logger的console_message.
    """
    alloc_logger().console_message(*msg, head=head, mid=mid, end=end)


if __name__ == "__main__":
    print(DefaultConfig.PATHS.LOG)
    log_message("test", "default", "log_message", mid=' ')
    log_message(1, 2, 3, 4, (1,2), mid='\t')
    logger1 = alloc_logger()
    logger1.log_message("test logger")
    logger1.log_message("test string signature", head=__file__)
    logger1.log_message("test class signature", head=Logger)
    logger1.log_message("test func signature", head=alloc_logger)
    logger1.log_message("test", " multi", " msg")
    logger1.log_message("test", "multi", "msg", "with", "mid", mid=' ')