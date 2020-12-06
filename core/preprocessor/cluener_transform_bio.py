# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/12/6 11:25
# Project Name: NER-CCF2020-HeapOverflow
# File        : cluener_transform_bio.py
# --------------------------------------------------

import json
import os
import pandas as pd


class CLUENERTransformer:
    def __init__(self, fIn: str, fOutData: str, fOutLabel: str, nextBaseIndex: int):
        self.fIn = open(fIn, mode='r', encoding='utf8')
        self.fOutData = fOutData
        self.fOutLabel = fOutLabel
        self.nextCnt = nextBaseIndex

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fIn.close()

    def beginLabel(self, label: str):
        return 'B-' + label

    def interLabel(self, label: str):
        return 'I-' + label

    def otherLabel(self):
        return 'O'

    def transform(self):
        while True:
            entry = self.fIn.readline()
            if entry == '':
                break
            entry = json.loads(entry)
            with open(os.path.join(self.fOutData, '{:}.txt'.format(self.nextCnt)), 'w', encoding='utf8') as fpOutData:
                text = entry['text']
                fpOutData.write(text)
            with open(os.path.join(self.fOutLabel, '{:}.csv'.format(self.nextCnt)), 'w', encoding='utf8') as fpOutLabel:
                df = pd.DataFrame(columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy'])
                label = entry['label']
                for each in label.items():
                    cur_label = each[0]
                    cur_entities = each[1]
                    for cur_entity in cur_entities.items():
                        cur_entity_name = cur_entity[0]
                        cur_entity_indexes = cur_entity[1]
                        for cur_entity_index in cur_entity_indexes:
                            L, R = cur_entity_index
                            newdf = pd.DataFrame(data=[{'ID': self.nextCnt,
                                                        'Category': cur_label,
                                                        'Pos_b': L,
                                                        'Pos_e': R,
                                                        'Privacy': cur_entity_name}])
                            df = df.append(newdf, ignore_index=True)
                df.to_csv(fpOutLabel, index_label=False, index=False)
            self.nextCnt += 1
        return self.nextCnt


if __name__ == '__main__':
    trainTrans = CLUENERTransformer(
        fIn='..\\..\\data\\CLUENER\\train.json',
        fOutData='..\\..\\data\\CCF\\raw\\train\\data',
        fOutLabel='..\\..\\data\\CCF\\raw\\train\\label',
        nextBaseIndex=2515
    )
    nextCnt = trainTrans.transform()
    del trainTrans
    print('NextCnt: {:}'.format(nextCnt))
    devTrans = CLUENERTransformer(
        fIn='..\\..\\data\\CLUENER\\dev.json',
        fOutData='..\\..\\data\\CCF\\raw\\train\\data',
        fOutLabel='..\\..\\data\\CCF\\raw\\train\\label',
        nextBaseIndex=nextCnt
    )
    nextCnt = devTrans.transform()
    del devTrans
    print('NextCnt: {:}'.format(nextCnt))
