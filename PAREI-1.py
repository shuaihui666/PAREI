#20220420 by Aohui
from sklearn.model_selection import train_test_split
from sklearn import preprocessing  
import gensim

from gensim.summarization import bm25
import numpy as np 
# import torch 
# from scipy.spatial.distance import cosine
# from transformers import AutoModel, AutoTokenizer
import os
import time
# device=torch.device('cuda:0')
X = list(range(4099))
mashup_id=X[719:]
train_mashup_id, test_mashup_id = train_test_split(mashup_id, test_size=0.2, random_state=1)#测试集划分
train_mashup_id, eval_mashup_id = train_test_split(train_mashup_id, test_size=0.25, random_state=1)#测试集划分
MashupNumWithoutTest=3379
proportionOfNode=0.5
stepOneMashupNum=50
apDict={}
maDict={}
description_text=[]
topNNum=[1,3,5,7,9]
resPath="./PAREI-1"

def initdsecDataset():
    #获取文本内容
    file_description=open('./data/dscps_encoded.txt','r',encoding='utf-8')
    for line in file_description.readlines():
        description_text.append(line.split(' =->= ')[1].replace("\n",''))
    file_description.close()
    #获取MA关系
    edgeFile=open('./data/mashup_api_graph.txt','r',encoding='utf-8')
    temp=[]
    nowIndex='719'
    for line in edgeFile.readlines():
        Index=line.split()[0]
        value=line.split()[1]
        if value in apDict.keys():
            apDict[value]=apDict[value]+1
        else:
            apDict[value]=1
        # print(Index,value)
        if(Index==nowIndex):
            temp.append(value)
        else:
            maDict[nowIndex]=temp
            temp=[]
            temp.append(value)
            nowIndex=Index
    maDict[nowIndex]=temp
    edgeFile.close()


def BM25Process():
    print("正在使用BM25计算文本相似度。。。")
    
    # print(description_text)
    all_description=[]
    for index in X:
        all_description.append(description_text[index].split())
    # print(all_description)

    testIndexItem=0
    Res=[]
    for testIndex in eval_mashup_id:
        bmDesArray=[]
        tempIndex=[]
        Res.append([])
        for textId in mashup_id:
            if textId==testIndex:#跳过当前QueryId
                continue
            else:
                tempIndex.append(textId)
                bmDesArray.append(all_description[textId])
        bm25Model = bm25.BM25(bmDesArray)
        scores=bm25Model.get_scores(all_description[testIndex])
        fullIndex=0
        for item in tempIndex:
            Res[testIndexItem].append([])
            Res[testIndexItem][fullIndex].append(item)
            Res[testIndexItem][fullIndex].append(scores[fullIndex])
            fullIndex=fullIndex+1
        testIndexItem=testIndexItem+1
    Index=0
    for Item in Res:
        Res[Index].sort(key=lambda x:x[0])
        Index=Index+1
    # print(Res[0])
    # print("BM25 List Length:"+str(len(Res[0])))
    return Res


def getApiIdWithMashupId(mashupId):
    return maDict[str(mashupId)]

def setStepTwoDataset(step2MashupIdList):
    leftDscps=[]
    rightDscps=[]
    leftApi=[]
    rightApi=[]
    rightData=[]
    rightDataId=[]
    for Index in eval_mashup_id:
        leftDscps.append(description_text[Index])
        leftApiList=getApiIdWithMashupId(Index)
        leftApi.append(leftApiList)
    for ItemIndex in range(0,len(step2MashupIdList)):
        step2MashupIdList[ItemIndex].sort(key=lambda x:x[1],reverse=True)
    print(step2MashupIdList[0])
    for ItemIndex in range(0,len(step2MashupIdList)):
        rightDscps.append([])
        rightApi.append([])
        step2MashupIdList[ItemIndex].sort(key=lambda x:x[1],reverse=True)
        for item in step2MashupIdList[ItemIndex][:60]:
            tempList=getApiIdWithMashupId(item[0])
            rightApi[ItemIndex]=rightApi[ItemIndex]+tempList
            for apiIndex in tempList:
                apiIndexNum=int(apiIndex)
                rightDscps[ItemIndex].append(description_text[apiIndexNum])
    for Index in range(0,len(rightApi)):
        rightData.append([])
        rightDataId.append([])
        thisIndex=0
        for minIndex in range(0,len(rightApi[Index])):
            apiId=rightApi[Index][minIndex]
            apiDesc=rightDscps[Index][minIndex]
            rightDataId[Index].append(apiId)
            rightData[Index].append([])
            rightData[Index][thisIndex].append(apiId)
            rightData[Index][thisIndex].append(apiDesc)
            thisIndex=thisIndex+1
    return leftDscps,rightData,rightDataId

def sim(eval_mashup_id,rightApi):
    print("正在使用BM25计算文本相似度。。。")
    
    # print(description_text)
    all_description=[]
    for index in X:
        all_description.append(description_text[index].split())
    # print(all_description)

    testIndexItem=0
    Res=[]
    for testIndex in eval_mashup_id:
        bmDesArray=[]
        tempIndex=[]
        Res.append([])
        for textId in rightApi:
            tempIndex.append(textId)
            bmDesArray.append(all_description[int(textId)])
        bm25Model = bm25.BM25(bmDesArray)
        scores=bm25Model.get_scores(all_description[testIndex])
        print(testIndex,len(scores))
        os.makedirs(resPath, exist_ok=True)
        file=open(resPath+'/demo'+str(testIndex)+".txt",'w+')
        apiItem=0
        for item in scores:
            file.write(str(rightApi[apiItem])+" "+str(item)+'\n')
            apiItem=apiItem+1
        file.close()

def evalData(testMashupId):
    print(resPath)
    rows = []
    for topn in topNNum:
        recallSum=0
        precisionSum=0
        recallAll=0
        leftAll=0
        AP=0
        for item in testMashupId:
            filepath=resPath+'/demo'+str(item)+".txt"
            # print(filepath)
            file=open(filepath,'r')
            leftApiList=getApiIdWithMashupId(item)
            rightDataTemp=[]
            Index=0
            for line in file.readlines():
                rightDataTemp.append([])
                rightDataTemp[Index].append(int(line.split()[0]))
                rightDataTemp[Index].append(float(line.split()[1]))
                Index=Index+1
            rightDataTemp.sort(key=lambda x:x[0],reverse=True)#倒序排序
            tempDict={}
            for ndItem in rightDataTemp:
                if(ndItem[0] in tempDict.keys()):
                    tempDict[ndItem[0]]=tempDict[ndItem[0]]+ndItem[1]
                else:
                    tempDict[ndItem[0]]=ndItem[1]
            # print(tempDict)
            rightData=[]
            keyIndex=0
            for key in tempDict.keys():
                rightData.append([])
                rightData[keyIndex].append(str(key))
                rightData[keyIndex].append(tempDict[key])
                keyIndex=keyIndex+1
            rightData.sort(key=lambda x:x[1],reverse=True)#倒序排序
            # print(rightData)
            recallNum=0
            IndexRe=0
            precisionSumAp=0
            for rightApiItem in rightData[:topn]:
                IndexRe=IndexRe+1
                isrel=0
                if(rightApiItem[0] in leftApiList):
                    recallNum=recallNum+1
                    isrel=1
                precisionSumAp=precisionSumAp+(recallNum/IndexRe)*isrel
            AP=AP+precisionSumAp/len(leftApiList)
            recallSum=recallSum+recallNum/len(leftApiList)
            precisionSum=precisionSum+recallNum/topn
            recallAll=recallAll+recallNum
            leftAll=leftAll+len(leftApiList)
            file.close()
        rows.append([topn, precisionSum/len(testMashupId), recallSum/len(testMashupId), AP/len(testMashupId)])

    # 表格形式输出
    header = ["top-N", "precision", "recall", "MAP"]
    col_widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(4)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    def fmt_row(r):
        return "| " + " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(r)) + " |"
    print("\n" + sep)
    print(fmt_row(header))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep + "\n")


if __name__=="__main__":
    ems=initdsecDataset()
    apDict = dict(sorted(apDict.items(),key=lambda x:x[1],reverse=True))
    rightApi=list(apDict.keys())[:50]

    simQA=sim(eval_mashup_id,rightApi)
    #获取simcse相似度比较结果

    evalData(eval_mashup_id)

    










