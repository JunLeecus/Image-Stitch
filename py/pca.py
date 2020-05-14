#-*- coding: UTF-8 -*-
from numpy import *
import numpy
def pca(X,CRate=128):
    #矩阵X每行是一个样本
     #对样本矩阵进行中心化样本矩阵
     meanValue=mean(X,axis=0)#计算每列均值
     X=X-meanValue#每个维度元素减去对应维度均值
     #协方差矩阵
     C=cov(X,rowvar=0)
     #特征值，特征向量
     eigvalue,eigvector=linalg.eig(mat(C))#特征值，特征向量
     #根据贡献率，来决定取多少个特征向量构成变换矩阵
     sumEigValue=sum(eigvalue)#所有特征值之和
     sortedeigvalue= numpy.sort(eigvalue)[::-1]    #对特征值从大到小排序
     for i in range(sortedeigvalue.size):
        j=i+1
        rate=sum(eigvalue[0:j])/sumEigValue
        if rate>CRate:
            break
     #取前j个列向量构成变换矩阵
     indexVec=numpy.argsort(-eigvalue)    #对covEigenVal从大到小排序，返回索引
     nLargestIndex=indexVec[:j] #取出最大的特征值的索引
     T=eigvector[:,nLargestIndex] #取出最大的特征值对应的特征向量
     newX=numpy.dot(X,T)#将X矩阵降维得到newX
     return newX,T,meanValue#返回降维后矩阵newX，变换矩阵T，每列的均值构成的数组返回降维后矩阵newX，变换矩阵T，每列的均值构成的数组