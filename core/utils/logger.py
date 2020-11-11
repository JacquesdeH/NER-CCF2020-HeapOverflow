import time
import sys

class Logger:
    def __init__(self, log_file_name: str, default_signature:str='', console_output: bool=True):
        # self.log_file = open(log_file_name, 'a', encoding='utf8')
        self.console_output = console_output
        self.default_signature = default_signature

    def __del__(self):
        pass
        # self.log_file.close()
        
    def get_time_stampe(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    def log_message(self, msg:str, signature:str='', end:str='\n'):
        time_stampe = self.get_time_stampe()
        total_msg = '[' + time_stampe + ']\t'
        if signature != '':
            total_msg += '@' + signature + ':\t'
        elif self.default_signature != '':
            total_msg += '@' + self.default_signature + ':\t'
        total_msg += msg + end
        if self.console_output:
            print(total_msg, end='')
        # self.log_file.write(total_msg)

if __name__ == "__main__":
    print(__file__)
    logger = Logger("")
    logger.log_message("shit")
    logger.log_message("fuck", __file__)
    logger = Logger("", "foo")
    logger.log_message("shiiit")
    logger.log_message("fuuuuck", Logger.__name__)
    print(type(Logger))
    t = Logger
    print(type("aaa"))
    b = isinstance(Logger, object)
    b = isinstance
    print(b)
