# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Name:        ItemBasedCF.py
# Purpose:     基于已知的训练集，"测试集"中的user的item进行评分预测.
# Data:        MovieLens ml-100k
# Language:    Python 2.7
# --------------------------------------------------------
from __future__ import division
import math, random, sys, datetime
from math import sqrt
from loadMovieLens import loadMovieLensTrain
from loadMovieLens import loadMovieLensTest

### 计算pearson相关度
def sim_pearson(prefer, item1, item2, shrinkage = 100):
    sim = {}
    # 查找评价过这两个item的user
    for (user,rated_items) in prefer.items():
        if (item1 in rated_items and item2 in rated_items):
            sim[user] = 1  # 将相同项添加到字典sim中
    # 元素个数
    n = len(sim)
    if len(sim) == 0:
        return 0

    # 商品偏好平均值
    mu1 = sum([prefer[user][item1] for user in sim])/n
    mu2 = sum([prefer[user][item2] for user in sim])/n

    # 求分子 ∑(Xi-mu1)(Yi-mu2)
    num1 = sum((prefer[user][item1] - mu1) * (prefer[user][item2] - mu2) for user in sim)
    
    # 求平方和
    sum1Sq = sum([pow(prefer[user][item1] - mu1, 2) for user in sim])
    sum2Sq = sum([pow(prefer[user][item2] - mu2, 2) for user in sim])
	
    #　求分母
    num2 = sqrt(sum1Sq) * sqrt(sum2Sq)

    if num2 == 0:  ### 如果分母为0，本处将返回0.
        return 0

    result = num1 / num2
	
    if(shrinkage > 0):	#考虑到数据的稀疏性　进行压缩
        result *= n / (n + shrinkage + 0.0)
	
    return result

### 计算cosine相关度
def sim_cosine(prefer, item1, item2, shrinkage = 100):
    sim = {}
    # 查找评价过这两个item的user
    for (user,rated_items) in prefer.items():
        if (item1 in rated_items and item2 in rated_items):
            sim[user] = 1  # 将相同项添加到字典sim中
    # 元素个数
    n = len(sim)
    if len(sim) == 0:
        return 0
    
    #求分子
    num1 = sum(prefer[user][item1] * prefer[user][item2] for user in sim)
    
    #求分母
    sum1Sq = sum([pow(prefer[user][item1], 2) for user in sim])
    sum2Sq = sum([pow(prefer[user][item2], 2) for user in sim])
    num2 = sqrt(sum1Sq) * sqrt(sum2Sq)
    
    result = num1 / num2
    
    if(shrinkage > 0):	#考虑到数据的稀疏性　进行压缩
    	result *= n / (n + shrinkage + 0.0)
    
    return result


### 计算jaccard相关度
def sim_jaccard(prefer, item1, item2, shrinkage = 100):
    sim = {} 	#评价过这两个item的user
    u1 = {} 	#评价过item1 的user
    u2 = {} 	#评价过item2 的user
    # 查找评价过item的user
    for (user,rated_items) in prefer.items():
        if (item1 in rated_items):
            u1[user] = 1
        if(item2 in rated_items):
            u2[user] = 1
        if (item1 in rated_items and item2 in rated_items):
            sim[user] = 1  # 将相同项添加到字典sim中
    # 元素个数
    n = len(sim)
    if len(sim) == 0:
        return 0
    
    #求分子，评分过item交集
    num1 = n

    #求分母 评分过item并集
    num2 = len(u1) + len(u2) - n
    
    result = num1 / num2
    
    if(shrinkage > 0):	#考虑到数据的稀疏性　进行压缩
    	result *= n / (n + shrinkage + 0.0)
    
    return result


### 获取对item评分的K个最相似用户（K默认20）
def topKMatches(prefer, userId, itemId, k=20, sim=sim_pearson):
    itemSet = []
    scores = []
    items = []
    # 找出所有prefer中用户评价过的Item,存入itemSet
    for item in prefer[userId]:
        itemSet.append(item)
    # 计算相似性
    scores = [(sim(prefer, itemId, other), other) for other in itemSet if other != itemId]

    # 按相似度排序
    scores.sort()
    scores.reverse()

    if len(scores) <= k:  # 如果小于k，只选择这些做推荐。
        for item in scores:
            items.append(item[1])  # 提取每项的itemId
        return items
    else:  # 如果>k,截取k个用户
        kscore = scores[0:k]
        for item in kscore:
            items.append(item[1])  # 提取每项的itemId
        return items  # 返回K个最相似用户的ID

### 计算项目的平均评分
def getAverage(prefer, itemId):
    userSet = {}		#评价过itemId的所有用户
    for (user,rated_items) in prefer.items():
        if (itemId in rated_items):
            userSet[user] = 1  # 将相同项添加到字典sim中

    count = 0
    sum = 0
    for user in userSet:
        sum = sum + prefer[user][itemId]
        count = count + 1
    if (count == 0):
        return 3.0
    return sum / count

### 平均加权策略，预测userId对itemId的评分
def getRating(prefer1, userId, itemId, knumber=20, similarity=sim_pearson):
    sim = 0.0
    averageOther = 0.0
    jiaquanAverage = 0.0
    simSums = 0.0
    # 获取K近邻项目(评过分的用户集)
    items = topKMatches(prefer1, userId, itemId, k=knumber, sim=similarity)

    # 获取itemId 的平均值
    averageOfItem = getAverage(prefer1, itemId)

    # 计算每个项目的加权，预测
    for other in items:
        sim = similarity(prefer1, itemId, other)  # 计算比较其他项目的相似度
        averageOther = getAverage(prefer1, other)  # 该项目的平均分
        # 累加
        simSums += abs(sim)  # 取绝对值
        jiaquanAverage += (prefer1[userId][other] - averageOther) * sim  # 累加，一些值为负

    # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回商品平均分
    if simSums == 0:
        return averageOfItem
    else:
        return (averageOfItem + jiaquanAverage / simSums)


        ##==================================================================

### 计算RMSE评分预测
def getRMSE(records):
    return math.sqrt(sum([(rui-pui)*(rui-pui) for u,i,rui,pui in records])/float(len(records)))

### 计算MAE评分预测
def getMAE(records):
    return sum([abs(rui-pui) for u,i,rui,pui in records])/float(len(records))

##     getAllUserRating(): 获取所有用户的预测评分，存放到fileResult中
##
## 参数:fileTrain,fileTest 是训练文件和对应的测试文件，fileResult为结果文件
##     similarity是相似度度量方法，默认是皮尔森。
##==================================================================
def getAllUserRating(fileTrain='ml-100k/u1.base', fileTest='ml-100k/u1.test',k=20, similarity=sim_pearson):
    traindata = loadMovieLensTrain(fileTrain)  # 加载训练集
    testdata = loadMovieLensTest(fileTest)  # 加载测试集
    inAllnum = 0
    records=[]
    for userid in testdata:  # test集中每个项目
        for item in testdata[userid]:  # 对于test集合中每一个项目用base数据集,CF预测评分
            rating = getRating(traindata, userid, item, k, similarity)  # 基于训练集预测用户评分(用户数目<=K)
            records.append([userid,item,testdata[userid][item],rating])
            inAllnum = inAllnum + 1
    SaveRecords(records)
    return records

def SaveRecords(records):
    file = open('records_ItemBasedCF.txt', 'a')
    file.write("%s\n" % ("------------------------------------------------------"))
    for u, i, rui, pui in records:
        file.write('%s\t%s\t%s\t%s\n' % (u, i, rui, pui))
    file.close()

############    主程序   ##############
if __name__ == "__main__":
    print("\n--------------基于ItemBased-KNN的推荐系统 运行中... -----------\n")
    trainfile='ml-100k/u1.base'
    testfile='ml-100k/u1.test'
    print("%3s %20s %20s %20s" % ('K', "RMSE","MAE","耗时"))
    starttime = datetime.datetime.now()
    for k in [10, 25, 50, 75, 100, 125, 150]:
        r=getAllUserRating(trainfile, testfile, k, sim_pearson)
        rmse=getRMSE(r)
        mae=getMAE(r)
        print("%3d %19.3f %19.3f %17ss" % (k, rmse, mae, (datetime.datetime.now() - starttime).seconds))