#20220420 by Aohui
from sklearn.model_selection import train_test_split
from sklearn import preprocessing  
import gensim
from gensim.summarization import bm25
import numpy as np 
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import time
device=torch.device('cuda:0')
X = list(range(4099))
mashup_id=X[719:]
train_mashup_id, test_mashup_id = train_test_split(mashup_id, test_size=0.2, random_state=1)#测试集划分
train_mashup_id,eval_mashup_id=train_test_split(train_mashup_id, test_size=0.25, random_state=1)#测试集划分
MashupNumWithoutTest=3379
proportionOfNode=0.5 
stepOneMashupNum=50
maDict={}
description_text=[]
sumResNum=40
trussTopN=5


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

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
    textIndex=0
    mainIndex=0
    textSim=[]
    emb=torch.Tensor().to(device)
    for textItem in description_text:
        textSim.append(textItem)
        textIndex=textIndex+1
        mainIndex=mainIndex+1
        if(textIndex==100):
            inputs = tokenizer(textSim, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.to(device)
            textSim=[]
            emb=torch.cat([emb,embeddings],0)
            # print("正在获取第"+str(mainIndex)+"个词嵌入")
            textIndex=0
    inputs = tokenizer(textSim, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.to(device)
    textSim=[]
    emb=torch.cat([emb,embeddings],0)
    print("已获取"+str(mainIndex)+"个句嵌入")
    textIndex=0

    return emb

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

def setStepTwoMashupId(sumScores):
    step2MashupList=[]
    for Index in range(0,len(sumScores)):
        step2MashupList.append([])
        for itemIndex in range(0,stepOneMashupNum):
            step2MashupList[Index].append(sumScores[Index][itemIndex])
    return step2MashupList

def getApiIdWithMashupId(mashupId):
    return maDict[str(mashupId)]

def setStepTwoDataset(step2MashupIdList,MashupScoreRes):
    leftDscps=[]
    rightDscps=[]
    leftApi=[]
    rightApi=[]
    rightData=[]
    rightDataId=[]
    rightMashupScore=[]
    for Index in eval_mashup_id:
        leftDscps.append(description_text[Index])
        leftApiList=getApiIdWithMashupId(Index)
        leftApi.append(leftApiList)
    for ItemIndex in range(0,len(step2MashupIdList)):
        rightDscps.append([])
        rightApi.append([])
        rightMashupScore.append([])
        mashupIndex=0
        for item in step2MashupIdList[ItemIndex]:
            tempList=getApiIdWithMashupId(item)
            rightApi[ItemIndex]=rightApi[ItemIndex]+tempList
            for apiIndex in tempList:
                rightMashupScore[ItemIndex].append([MashupScoreRes[ItemIndex][mashupIndex][0],MashupScoreRes[ItemIndex][mashupIndex][1]])
                apiIndexNum=int(apiIndex)
                rightDscps[ItemIndex].append(description_text[apiIndexNum])
            mashupIndex=mashupIndex+1
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
    return leftDscps,rightData,rightDataId,rightMashupScore

def simCsePro(eval_mashup_id,rightApi,embeddings,rightMashupScore):
    leftId=eval_mashup_id
    rightId=rightApi
    Res=[]
    IndexLeft=0
    for leftItem in leftId:
        Res.append([])
        IndexRight=0
        texts = []
        file=open("demo"+str(leftItem)+".txt",'w+')
        start =time.clock()
        texts.append(description_text[leftItem])
        # print(str(IndexLeft)+"/"+str(len(eval_mashup_id)))
        for rightItem in rightId[IndexLeft]:
            texts.append(description_text[int(rightItem)])        
        for rightItem in rightId[IndexLeft]:
            cosine_sim = 1 - cosine(embeddings[leftItem].cpu(), embeddings[int(rightItem)].cpu())
            Res[IndexLeft].append([])
            Res[IndexLeft][IndexRight].append(rightItem)
            Res[IndexLeft][IndexRight].append(cosine_sim)
            file.write(str(rightItem)+" "+str(cosine_sim)+' '+str(rightMashupScore[IndexLeft][IndexRight][0])+' '+str(rightMashupScore[IndexLeft][IndexRight][1])+'\n')
            IndexRight=IndexRight+1
        file.close()
        end =time.clock()
        print('query '+str(IndexLeft+1)+"/"+str(len(eval_mashup_id))+' Running time: '+str(end-start)+' Seconds')
        IndexLeft=IndexLeft+1
    return Res

def ListRebuild2(BM25_Scores):
    PATH = './data/mashup_embedding_100_temp_output.txt'
    node_embeddings = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=False)
    Index=0
    Res=[]
    MashupScoreRes=[]
    for testIndexItem in BM25_Scores:
        testIndexItem.sort(key=lambda x:x[1],reverse=True)
        topNodeListBefore=testIndexItem[:trussTopN]
        topNodeList=[]
        for item in topNodeListBefore:
            topNodeList.append(item[0])
        topNodeSimList=[]
        revNodeList=[]
        scoreNodeList=[]
        itemIndexthis=0
        for item in testIndexItem[trussTopN:]:
            revNodeList.append([])
            revNodeList[itemIndexthis].append(int(item[0]))
            revNodeList[itemIndexthis].append(item[1])
            itemIndexthis=itemIndexthis+1
            scoreNodeList.append([])
        revNodeList.sort(key=lambda x:x[0],reverse=True)
        lsList=[]
        for item in revNodeList:
            lsList.append([item[0],item[1]])
        # 归一化
        npBm25=np.array(lsList)
        min_max_scaler = preprocessing.MinMaxScaler()
        npBm25_minMax = min_max_scaler.fit_transform(npBm25)
        lsList=npBm25_minMax.tolist()
        for item in range(0,len(revNodeList)):
            revNodeList[item][1]=lsList[item][1]
        # if(Index==0):
        #     print(revNodeList)
            
        # 归一化
        for topNItem in range(0,trussTopN):
            nodeSimRes=node_embeddings.similar_by_word(str(topNodeList[topNItem]),topn=3379)
            # print(len(nodeSimRes))
            nodeTopIndex=0
            for nodeItem in nodeSimRes:
                if((int(nodeItem[0]) not in topNodeList) and int(nodeItem[0])!=eval_mashup_id[Index]):
                    scoreNodeList[topNItem].append([])
                    scoreNodeList[topNItem][nodeTopIndex].append(int(nodeItem[0]))
                    scoreNodeList[topNItem][nodeTopIndex].append(nodeItem[1])
                    nodeTopIndex=nodeTopIndex+1
            scoreNodeList[topNItem].sort(key=lambda x:x[0],reverse=True)
        nodeSimSum=[]
        for revLineIndex in range(0,len(revNodeList)):
            nodeSimSum.append(0)
        for topNItem in range(0,trussTopN):
            for revLineIndex in range(0,len(revNodeList)):
                if(revNodeList[revLineIndex][0]==scoreNodeList[topNItem][revLineIndex][0]):
                    nodeSimSum[revLineIndex]=nodeSimSum[revLineIndex]+scoreNodeList[topNItem][revLineIndex][1]/trussTopN
                else:
                    print("error!!!")
        # print(nodeSimSum)
        for revLineIndex in range(0,len(revNodeList)):
            revNodeList[revLineIndex][1]=revNodeList[revLineIndex][1]/2+nodeSimSum[revLineIndex]/2
        revNodeList.sort(key=lambda x:x[1],reverse=True)
        # print(len(revNodeList))
        # print(revNodeList)
        listResTemp=[]
        mashupScores=[]
        for item in topNodeList:
            mashupScores.append([item,1])
        for item in revNodeList[:sumResNum-trussTopN]:
            listResTemp.append(item[0])
            mashupScores.append([item[0],item[1]])
        Res.append(topNodeList+listResTemp)
        MashupScoreRes.append(mashupScores)
        # if(Index==0):
        #     print(Res)
        Index=Index+1
    return Res,MashupScoreRes




if __name__=="__main__":
    ems=initdsecDataset() 
    #读取文本描述数据集

    BM25_Scores=BM25Process()
    #计算BM25分数

    RebuildNodeList,MashupScoreRes=ListRebuild2(BM25_Scores)
    #队列融合图特征重排序

    leftDscps,rightData,rightApi,rightMashupScore=setStepTwoDataset(RebuildNodeList,MashupScoreRes)
    #构建步骤二数据集

    simQA=simCsePro(eval_mashup_id,rightApi,ems,rightMashupScore)
    #获取simcse相似度比较结果

    
    










