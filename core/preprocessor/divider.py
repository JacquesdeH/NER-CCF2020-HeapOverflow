from ..utils import alloc_logger

class Divider:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.base_index = max_size
        self.logger = alloc_logger("preprocessor.log", Divider)

    def check(self, data: str, ret: list, ch: str) -> bool:
        for i in range(self.max_size):
            index = self.base_index - i
            if data[index - 1] == ch:
                ret.append(index)
                if len(data) - index <= self.max_size:
                    return True
                self.base_index = self.max_size + index
        return False
    
    def detect_division(self, data: str) -> list:
        """
        @return: 每一个被拆分的段的起始的下标.
        """
        if len(data) <= self.max_size:
            return [0]
        ret = [0]
        self.base_index = self.max_size
        while True:
            if self.check(data, ret, '。'):
                if len(data) - ret[-1] <= self.max_size:
                    return ret
                else:
                    continue
            if self.check(data, ret, '，'):
                if len(data) - ret[-1] <= self.max_size:
                    return ret
                else:
                    continue
            if self.check(data, ret, '.'):
                if len(data) - ret[-1] <= self.max_size:
                    return ret
                else:
                    continue
            ret.append(self.base_index)
            self.base_index = self.max_size + self.base_index
            if len(data) - ret[-1] <= self.max_size:
                return ret

if __name__ == "__main__":
    divider = Divider(5)
    sample1 = "01234567890123456789012345678901234567890123456789012345678901234567890123456789"
    result1 = divider.detect_division(sample1)
    divider.logger.log_message(result1)
    sample2 = "0。2345678。012345678901。3456789012，。5678901234567，901234567890123，567890123456789"
    result2 = divider.detect_division(sample2)
    divider.logger.log_message(result2)