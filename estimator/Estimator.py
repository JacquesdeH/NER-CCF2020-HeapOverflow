# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math

class Estimators:
    '''
    label分布：统计每个label出现时的长度
    文本长度分布统计:统计每个文本的长度
    关联度分析：统计类别在文件中的出现情况
    '''
    def __init__(self,fileset:list):
        self.fileset=fileset
        self.labels=set()
        self.startPrintStatus()
        for i in range(len(self.fileset)):
            df=pd.read_csv('../data/CCF/raw/train//label/'+self.fileset[i]+'.csv')
            self.labels|=set(df.Category.values)
            self.continuePrintStatus(i)
        self.endPrintStatus()
        print("INIT DONE.")
    
    def calculate(self):
        self.appearance=dict()
        self.coinCount=dict()
        self.entityLength4Label=dict()
        self.fileLengths=list()
        self.potentialNoise=list()
        for label in self.labels:
            self.appearance[label]=list()
            self.entityLength4Label[label]=list()
            self.coinCount[label]=dict()
            for label0 in self.labels:
                self.coinCount[label][label0]=0
        self.startPrintStatus()
        for i in range(len(self.fileset)):
            df=pd.read_csv('../data/CCF/raw/train//label/'+self.fileset[i]+'.csv')
            if True in set(df.Privacy.value_counts()>1):
                self.potentialNoise.append(self.fileset[i])
                continue
            for label in self.appearance.keys():
                if label in df.Category.unique():
                    self.appearance[label].append(1)
                else:
                    self.appearance[label].append(0)
            for j in range(len(df)):
                self.entityLength4Label[df.iloc[j]['Category']].append(df.iloc[j]['Pos_e']-df.iloc[j]['Pos_b']+1)
            uni=df.Category.unique()
            for j in range(len(uni)):
                for k in range(j+1,len(uni)):
                    self.coinCount[uni[j]][uni[k]]+=1
                    self.coinCount[uni[k]][uni[j]]+=1
            with open('../data/CCF/raw/train//data/'+self.fileset[i]+'.txt','r',encoding='utf8') as f:
                l=0
                for j in f.readlines():
                    l+=len(j)
                self.fileLengths.append(l)
            self.continuePrintStatus(i)
        self.endPrintStatus()
        print('CALCULATE DONE')
                
            
    
    def startPrintStatus(self):
        self.lastStatus=0
        print('Process:',end='')
    
    def continuePrintStatus(self,n):
        status=n/len(self.fileset)
        status=math.floor(status*10)/10
        if status>self.lastStatus:
            print('▇▇▇▇▇',end='')
            self.lastStatus=status
    
    def endPrintStatus(self):
        self.lastStatus=0
        print('100% completed.')

    def showSimilarity(self):
        '''
        第一种直接由共同出现次数计算热点图
        第二种将出现次数除以二者的几何均值后再计算热点图
        第三种计算二者出现向量的余弦相似度
        '''
        df=pd.DataFrame(self.coinCount)
        #_4sort=df.index.tolist()
        _4sort=['name','position','vx','QQ','email','mobile','movie','book','scene','game','government','address','organization','company']
        df.sort_values(by=_4sort,axis=1,inplace=True)
        df.sort_values(by=_4sort,axis=0,ascending=False,inplace=True)
        df=df.astype('float64')
        plt.figure()
        sns.heatmap(data=df,cmap='OrRd')
        for label0 in _4sort:
            for label1 in _4sort:
                df[label0][label1]=df[label0][label1]/((len(self.entityLength4Label[label0])*len(self.entityLength4Label[label1]))**0.5)
        plt.figure()
        sns.heatmap(data=df,cmap='OrRd')
        sqrtL2=dict()
        for label in _4sort:
            sqrtL2[label]=sum(self.appearance[label])**0.5
        temp=dict()
        for label0 in _4sort:
            temp[label0]=dict()
            for label1 in _4sort:
                mulsum=0
                for i in range(len(self.appearance[label0])):
                    mulsum+=self.appearance[label0][i]*self.appearance[label1][i]
                temp[label0][label1]=mulsum/sqrtL2[label0]/sqrtL2[label1]
        df=pd.DataFrame(temp)
        df.sort_values(by=_4sort,axis=1,inplace=True)
        df.sort_values(by=_4sort,axis=0,inplace=True,ascending=False)
        plt.figure()
        sns.heatmap(data=df,cmap='OrRd')
        for label in _4sort:
            df[label][label]=0
        plt.figure()
        sns.heatmap(data=df,cmap='OrRd')
        
    def showLabelDistribution(self):
        '''
        第一种是label出现次数排序图
        第二种是label所对应entity平均长度排序图
        '''
        count1=dict()
        xLabel=list(self.entityLength4Label.keys())
        for label in xLabel:
            count1[label]=len(self.entityLength4Label[label])
        xLabel.sort(key=lambda x:count1[x])
        plt.figure()
        sns.barplot(x=xLabel,y=[count1[i] for i in xLabel])
        
        count2=dict()
        for label in xLabel:
            count2[label]=sum(self.entityLength4Label[label])/len(self.entityLength4Label[label])
        xLabel.sort(key=lambda x:count2[x])
        plt.figure()
        sns.barplot(x=xLabel,y=[count2[i] for i in xLabel])
        
    def showFileLength(self):
        '''
        第一张图是长度分布图
        第二张图是通过核密度估计算出的分布图
        '''
        plt.figure()
        sns.distplot(a=self.fileLengths,kde=False)
        plt.figure()
        sns.kdeplot(data=self.fileLengths,shade=True)
        

if __name__=='__main__':
    fileset=[str(i) for i in range(2515)]
    es=Estimators(fileset)
    es.calculate()
    es.showSimilarity()
    es.showLabelDistribution()
    es.showFileLength()