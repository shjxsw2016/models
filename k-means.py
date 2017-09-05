# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.misc
from scipy.misc import imsave
import cPickle as pickle

# 定义Struct结构
class ScreenData(object):
	def __init__(self, data, name):
		self.data = data
		self.name = name
	def __repr__(self):
		return self.name


# calculate Euclidean distance
def euclDistance(vec1, vec2):
	return sum(np.square(vec2-vec1))

# init centroids with random samples  
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = np.zeros((k, dim))
	s = set()
	# 随机初始化k个聚类中心
	for i in range(k):
		while True:
			index = int(np.random.uniform(0, numSamples))
			if index not in s:
				s.add(index)
				break
		centroids[i, :] = dataSet[index, :]  
	return centroids 


def k_means(dataSet, k):
	numSamples = dataSet.shape[0]
	clusterAssment = np.mat(np.zeros((numSamples, 2))) 
	for i in xrange(numSamples):  
		clusterAssment[i, 0] = -1 
	clusterChanged = True  

	# step 1：init centroids
	centroids = initCentroids(dataSet, k)
	count=0
	while clusterChanged:
		count = count+1
		clusterChanged=False
		## for each sample
		for i in xrange(numSamples):
			minDist = 100000.0
			minIndex = 0

			## for each centroid
			## step 2: find the centroid who is closest
			for j in range(k):
				distance = euclDistance(centroids[j,:], dataSet[i,:])
				if distance<minDist:
					minDist = distance
					minIndex = j

			## step 3: 更新样本点与中心点的分配关系
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :]=minIndex, minDist
			else:
				clusterAssment[i,1]=minDist

		## step 4: updat centroids
		for j in range(k):
			pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  
			centroids[j,:] = np.mean(pointsInCluster, axis=0)
	print(count)
	return centroids, clusterAssment

def make_dir(filepath):
	if os.path.isdir(filepath):
		pass
	else:
		os.mkdir(filepath)

def showCluster(dataSet, k, centroids, clusterAssment, filename):
	numSamples, dim = dataSet.shape
	path1 = './class1/'
	path2 = './class2/'
	path3 = './class3/'
	path4 = './class4/'
	make_dir(path1)
	make_dir(path2)
	make_dir(path3)
	make_dir(path4)
	for i in xrange(numSamples):
		markIndex = int(clusterAssment[i, 0])
		img = Image.open('./data/ngdata64/'+filename[i])
		if markIndex == 0:
			imsave(path1+filename[i],img)
		if markIndex == 1:
			imsave(path2+filename[i],img)
		if markIndex == 2:
			imsave(path3+filename[i],img)
		if markIndex == 3:
			imsave(path4+filename[i],img)
	# f1 = open('class1.txt','w')
	# f2 = open('class2.txt','w')
	# f3 = open('class3.txt','w')
	# f4 = open('class4.txt','w')
	# for i in xrange(numSamples):
	# 	markIndex = int(clusterAssment[i, 0])
	# 	if markIndex == 0:
	# 		f1.write(filename[i])
	# 		f1.write('\r\n')
	# 	if markIndex == 1:
	# 		f2.write(filename[i])
	# 		f2.write('\r\n')
	# 	if markIndex == 2:
	# 		f3.write(filename[i])
	# 		f3.write('\r\n')
	# 	if markIndex == 3:
	# 		f4.write(filename[i])
	# 		f4.write('\r\n')
	# f1.close()
	# f2.close()
	# f3.close()
	# f4.close()


# 载入本地文件
read_pkl = open('./data/screen_center.pkl','rb')
screen_vec = pickle.load(read_pkl)
filename   = pickle.load(read_pkl)

print(screen_vec[0].shape)
numSamples= len(screen_vec)

dim = len(screen_vec[0])
dataSet = np.reshape(screen_vec, [numSamples, dim])
#print(dataSet[0])
# 把本地文件导入结构中
"""
fulldata = []
for idx in range(len(screen_vec)):
	sd = ScreenData(screen_vec[idx], filename[idx])
	fulldata.append(sd)
"""

k=2
centroids, clusterAssment = k_means(dataSet, k)
showCluster(dataSet, k, centroids, clusterAssment, filename)