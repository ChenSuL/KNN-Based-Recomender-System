# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Name:        UserBasedCF.py
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
def sim_pearson(prefer, person1, person2, shrinkage = 100):
    sim = {}
    # 查找双方都评价过的项
    for item in prefer[person1]:
        if item in prefer[person2]:
            sim[item] = 1  # 将相同项添加到字典sim中
    # 元素个数
    n = len(sim)
    if len(sim) == 0:
        return 0

    # 用户偏好平均值
    mu1 = sum([prefer[person1][item] for item in sim])/n
    mu2 = sum([prefer[person2][item] for item in sim])/n

    # 求平方和
    sum1Sq = sum([pow(prefer[person1][item] - mu1, 2) for item in sim])
    sum2Sq = sum([pow(prefer[person2][item] - mu2, 2) for item in sim])

    # 求分子 ∑(Xi-mu1)(Yi-mu2)
    num1 = sum((prefer[person1][item] - mu1) * (prefer[person2][item] - mu2) for item in sim)
    
	#　求分母
    num2 = sqrt(sum1Sq) * sqrt(sum2Sq)
	
    if num2 == 0:  ### 如果分母为0，本处将返回0.
        return 0

    result = num1 / num2
	
    if(shrinkage > 0):	#考虑到数据的稀疏性　进行压缩
        result *= n / (n + shrinkage + 0.0)
	
    return result


### 计算cosine相关度
def sim_cosine(prefer, person1, person2, shrinkage = 100):
    sim = {}
    # 查找双方都评价过的项
    for item in prefer[person1]:
        if item in prefer[person2]:
            sim[item] = 1  # 将相同项添加到字典sim中
    # 元素个数
    n = len(sim)
    if len(sim) == 0:
        return 0
    
    #求分子
    num1 = sum(prefer[person1][item] * prefer[person2][item] for item in sim)
    
    #求分母
    sum1Sq = sum([pow(prefer[person1][item], 2) for item in sim])
    sum2Sq = sum([pow(prefer[person2][item], 2) for item in sim])
    num2 = sqrt(sum1Sq) * sqrt(sum2Sq)
    
    result = num1 / num2
    
    if(shrinkage > 0):	#考虑到数据的稀疏性　进行压缩
    	result *= n / (n + shrinkage + 0.0)
    
    return result


### 计算jaccard相关度
def sim_jaccard(prefer, person1, person2, shrinkage = 100):
    sim = {}
    # 查找双方都评价过的项
    for item in prefer[person1]:
        if item in prefer[person2]:
            sim[item] = 1  # 将相同项添加到字典sim中
    # 元素个数
    n = len(sim)
    if len(sim) == 0:
        return 0
    
    #求分子，评分过item交集
    num1 = n

    #求分母 评分过item并集
    num2 = len(prefer[person1]) + len(prefer[person2]) - n
    
    result = num1 / num2
    
    if(shrinkage > 0):	#考虑到数据的稀疏性　进行压缩
    	result *= n / (n + shrinkage + 0.0)
    
    return result


### 获取对item评分的K个最相似用户（K默认20）
def topKMatches(prefer, person, itemId, k=20, sim=sim_pearson):
    userSet = []
    scores = []
    users = []
    # 找出所有prefer中评价过Item的用户,存入userSet
    for user in prefer:
        if itemId in prefer[user]:
            userSet.append(user)
    # 计算相似性
    scores = [(sim(prefer, person, other), other) for other in userSet if other != person]

    # 按相似度排序
    scores.sort()
    scores.reverse()

    if len(scores) <= k:  # 如果小于k，只选择这些做推荐。
        for item in scores:
            users.append(item[1])  # 提取每项的userId
        return users
    else:  # 如果>k,截取k个用户
        kscore = scores[0:k]
        for item in kscore:
            users.append(item[1])  # 提取每项的userId
        return users  # 返回K个最相似用户的ID

### 计算用户的平均评分
def getAverage(prefer, userId):
    count = 0
    sum = 0
    for item in prefer[userId]:
        sum = sum + prefer[userId][item]
        count = count + 1
    return sum / count

### 平均加权策略，预测userId对itemId的评分
def getRating(prefer1, userId, itemId, knumber=20, similarity=sim_pearson):
    sim = 0.0
    averageOther = 0.0
    jiaquanAverage = 0.0
    simSums = 0.0
    # 获取K近邻用户(评过分的用户集)
    users = topKMatches(prefer1, userId, itemId, k=knumber, sim=similarity)

    # 获取userId 的平均值
    averageOfUser = getAverage(prefer1, userId)

    # 计算每个用户的加权，预测
    for other in users:
        sim = similarity(prefer1, userId, other)  # 计算比较其他用户的相似度
        averageOther = getAverage(prefer1, other)  # 该用户的平均分
        # 累加
        simSums += abs(sim)  # 取绝对值
        jiaquanAverage += (prefer1[other][itemId] - averageOther) * sim  # 累加，一些值为负

    # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
    if simSums == 0:
        return averageOfUser
    else:
        return (averageOfUser + jiaquanAverage / simSums)


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

def loadfile(filename):
    ''' load a file, return a generator. '''
    fp = open(filename, 'r')
    for i, line in enumerate(fp):
        yield line.strip('\r\n')
#       if i % 100000 == 0:
#            print >> sys.stderr, 'loading %s(%s)' % (filename, i)
    fp.close()
    #print >> sys.stderr, 'load %s succ' % filename

def getAllUserRating(trainfile, testfile, k=20, similarity=sim_pearson):
    traindata = loadMovieLensTrain(trainfile)  # 加载训练集
    testdata = loadMovieLensTest(testfile)  # 加载测试集
    inAllnum = 0
    records=[]
    for userid in testdata:  # test集中每个用户
        for item in testdata[userid]:  # 对于test集合中每一个项目用base数据集,CF预测评分
            rating = getRating(traindata, userid, item, k, similarity)  # 基于训练集预测用户评分(用户数目<=K)
            records.append([userid,item,testdata[userid][item],rating])
            inAllnum = inAllnum + 1
    SaveRecords(records)
    return records

def SaveRecords(records):
    file = open('records_UserBasedCF.txt', 'a')
    file.write("%s\n" % ("------------------------------------------------------"))
    for u, i, rui, pui in records:
        file.write('%s\t%s\t%s\t%s\n' % (u, i, rui, pui))
    file.close()

############    主程序   ##############
if __name__ == "__main__":
    trainfile = 'ml-100k/u2.base'
    testfile = 'ml-100k/u2.test'
    ratingfile = 'ml-1m/ratings.dat'
    print("\n--------------基于UserBased-KNN的推荐系统 运行中... -----------\n")
    starttime = datetime.datetime.now()
    print("%3s %20s %20s %20s" % ('K', "RMSE","MAE","耗时"))
    for k in [10, 25, 50, 75, 100, 125, 150]:
        r=getAllUserRating(trainfile, testfile, k, sim_pearson)
        rmse=getRMSE(r)
        mae=getMAE(r)
        print("%3d %19.3f %19.3f %17ss" % (k, rmse, mae, (datetime.datetime.now() - starttime).seconds))
