# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:20:19 2020

@author: jhzhang
"""
import numpy as np
import vtk
import math
from numpy import *

readerSTL = vtk.vtkSTLReader()
readerSTL.SetFileName("l4r.stl")
readerSTL.Update()
vertebra = readerSTL.GetOutput()
vertebraMapper = vtk.vtkPolyDataMapper() 
vertebraMapper.SetInputData(vertebra)         # maps polygonal data to graphics primitives
vertebraActor = vtk.vtkLODActor() 
vertebraActor.SetMapper(vertebraMapper)
vertebraActor.GetProperty().EdgeVisibilityOn()
vertebraActor.GetProperty().SetLineWidth(0.3)

a=np.loadtxt('surface_0.txt')  #加载对称面
amid=int(((a.shape)[0])/2)
amax=(a.shape)[0]-1
#3点确定对称面
point_position1 = tuple((a[0][0], a[0][1], a[0][2]))
point_position2 = tuple((a[amid][0], a[amid][1], a[amid][2]))
point_position3 = tuple((a[amax][0], a[amax][1], a[amax][2]))
Sysplane = vtk.vtkPlaneSource()            
Sysplane.SetOrigin(point_position1)
Sysplane.SetPoint1(point_position2)
Sysplane.SetPoint2(point_position3)                
SysMapper = vtk.vtkPolyDataMapper()
SysMapper.SetInputConnection(Sysplane.GetOutputPort())
SysActor = vtk.vtkActor()
SysActor.SetMapper(SysMapper)
SysActor.GetProperty().SetColor(1,0,0)

#放大对称平面
p1 = Sysplane.GetPoint1()
p2 = Sysplane.GetPoint2()
cx = (p1[0]+p2[0])/2
cy = (p1[1]+p2[1])/2
cz = (p1[2]+p2[2])/2
TransSysfilter = vtk.vtkTransformPolyDataFilter()
TransSysfilter.SetInputConnection(Sysplane.GetOutputPort())
TransSys = vtk.vtkTransform()
TransSys.Translate(cx,cy,cz)
TransSys.Scale(5,5,5)
TransSys.Translate(-cx,-cy,-cz)
TransSysfilter.SetTransform(TransSys)
TransSysfilter.Update() 

#%%
#获得对称面和椎体的交线（先三角化平面）    
TriSysfilter = vtk.vtkTriangleFilter()
TriSysfilter.SetInputConnection(TransSysfilter.GetOutputPort())
intersectSys = vtk.vtkIntersectionPolyDataFilter()
intersectSys.SetInputData(0,vertebra)
intersectSys.SetInputConnection( 1, TriSysfilter.GetOutputPort())
intersectSys.Update()
t = intersectSys.GetOutput()
nCells = t.GetNumberOfCells()
nPoints = t.GetNumberOfPoints()
print(" nCells:", nCells)
print(" nPoints:", nPoints)

p = [0, 0, 0]
pq = []
for i in range(t.GetNumberOfPoints()):
    t.GetPoint(i, p)
    # print(p)
    pi=[p[0],p[1],p[2]]
    pq.append(pi)
np.savetxt("pq.txt",pq, fmt="%.18f", delimiter=",")
#%%
#二维点聚类
# 加载数据
def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []              # 文件的最后一个字段是类别标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float, curLine))    # 将每个元素转成float类型
        dataMat.append(fltLine)
    return dataMat

# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离

# 构建聚簇中心，取k个(此例中k=4)随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    '''
    :param dataSet:  没有lable的数据集  (本例中是二维数据)
    :param k:  分为几个簇
    :param distMeans:    计算距离的函数
    :param createCent:   获取k个随机质心的函数
    :return: centroids： 最终确定的 k个 质心
            clusterAssment:  该样本属于哪类  及  到该类质心距离
    '''
    m = shape(dataSet)[0]   #m=样本数量
    clusterAssment = mat(zeros((m,2)))
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离，
    centroids = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
        # print centroids
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment
# --------------------测试----------------------------------------------------
# 用测试数据及测试kmeans算法

datMat = mat(loadDataSet('pq.txt'))
    # print (min(datMat[:,0]))
    # print (max(datMat[:,1]))
print (randCent(datMat,2))
myCentroids,clustAssing = kMeans(datMat,2)
print (myCentroids)
print (clustAssing,len(clustAssing))
p_point=[]
q_point=[]
for i in range(0,len(clustAssing)):
        # print(i)
     if clustAssing[i,0]==1.0:
        p_point.append(datMat[i])
            # p_point=np.array(p_point)
     elif clustAssing[i,0]==0.0:
        q_point.append(datMat[i])
            # q_point=np.array(q_point)        
     else:
        print('发生错误')
np.save(file='p_point.npy',arr=p_point)
np.save(file='q_point.npy',arr=q_point)

#%%
intersectSysMapper  = vtk.vtkPolyDataMapper()
intersectSysMapper.SetInputData(t)
intersectSysActor = vtk.vtkActor()
intersectSysActor.SetMapper( intersectSysMapper )
intersectSysActor.GetProperty().SetColor(1.0, 0.0, 0.0)

pn=np.load('p_point.npy')
qn=np.load('q_point.npy')
p=np.mat(pn)
q=np.mat(qn)

dmm = 1e+10
cq  = [0,0,0]
cp  = [0,0,0]
cpq = [0,0,0]
qindex = 0
pindex = 0

pi=(p.shape)[0]
qi=(q.shape)[0]

for i in range(pi):
    dm = 1e+10
    px = p[i,0] 
    py = p[i,1] 
    pz = p[i,2]
    for j in range(qi):
        qx = q[j,0] 
        qy = q[j,1] 
        qz = q[j,2]
        d = math.sqrt((px-qx)*(px-qx) + (py-qy)*(py-qy) + (pz-qz)*(pz-qz))
        if d < dm:
            dm = d
            cq = [qx,qy,qz]
            qindex = j
    if dm < dmm:
        dmm = dm
        cp = [px,py,pz]
        cpq = cq
        pindex = i
        pqindex = qindex

mx = (cp[0]+cpq[0])/2.0;
my = (cp[1]+cpq[1])/2.0;
mz = (cp[2]+cpq[2])/2.0;

#绘制PQ
lineSource = vtk.vtkLineSource()
points = vtk.vtkPoints()
p = cpq
q = cp
m = [mx,my,mz]
points.InsertNextPoint(p)
points.InsertNextPoint(q)
lineSource.SetPoints(points)
linemapper = vtk.vtkPolyDataMapper()
linemapper.SetInputConnection(lineSource.GetOutputPort())
lineactor = vtk.vtkActor()
lineactor.SetMapper(linemapper)
lineactor.GetProperty().SetColor(1.0,1.0,0.0)

#%%
#Coroplane平面
coro=np.loadtxt('coropoint.txt')
Coroplane = vtk.vtkPlaneSource()

x1=mx+10
y1=mx+10
x2=mx-10
y2=mx-10 
z1=(((p[0]-q[0])*(x1-m[0])+(p[1]-q[1])*(y1-m[1]))/(q[2]-p[2]))+mz  
z2=(((p[0]-q[0])*(x2-m[0])+(p[1]-q[1])*(y2-m[1]))/(q[2]-p[2]))+mz
         
Coroplane.SetOrigin(tuple(m))      
Coroplane.SetPoint1(tuple((x1,y1,z1)))     
Coroplane.SetPoint2(tuple((x2,y2,z2)))
CoroMapper = vtk.vtkPolyDataMapper()
CoroMapper.SetInputConnection(Coroplane.GetOutputPort())
CoroActor = vtk.vtkActor()
CoroActor.SetMapper(CoroMapper)
CoroActor.GetProperty().SetColor(0,1,0) 
     

#放大Coroplane平面
pc1 = Coroplane.GetPoint1()
pc2 = Coroplane.GetPoint2()
ccx = (pc1[0]+pc2[0])/2
ccy = (pc1[1]+pc2[1])/2
ccz = (pc1[2]+pc2[2])/2
TransCorofilter = vtk.vtkTransformPolyDataFilter()
TransCorofilter.SetInputConnection(Coroplane.GetOutputPort())
TransCoro = vtk.vtkTransform()
TransCoro.Translate(ccx,ccy,ccz)
TransCoro.Scale(5,5,5)
TransCoro.Translate(-ccx,-ccy,-ccz)
TransCorofilter.SetTransform(TransCoro)
TriCorofilter = vtk.vtkTriangleFilter()
TriCorofilter.SetInputConnection(TransCorofilter.GetOutputPort())
intersectCoro = vtk.vtkIntersectionPolyDataFilter()
intersectCoro.SetInputData(0,vertebra)
intersectCoro.SetInputConnection( 1, TriCorofilter.GetOutputPort())
intersectCoro.Update()
tc = intersectCoro.GetOutput()
intersectCoroMapper  = vtk.vtkPolyDataMapper()
intersectCoroMapper.SetInputData(tc)
intersectCoroActor = vtk.vtkActor()
intersectCoroActor.SetMapper( intersectCoroMapper )
intersectCoroActor.GetProperty().SetColor(0.0, 1.0, 0.0)

#%%
ren = vtk.vtkRenderer()
ren.SetBackground(0.5, 0.5, 0.5);
ren.AddActor(vertebraActor);

ren.AddActor(intersectSysActor);
# ren.AddActor(SysActor);

ren.AddActor(lineactor);

ren.AddActor(intersectCoroActor);
# ren.AddActor(CoroActor);

win = vtk.vtkRenderWindow()
win.SetSize(600, 600)
win.AddRenderer(ren)
inren = vtk.vtkRenderWindowInteractor()
inren.SetRenderWindow(win)
inren.Initialize()
win.Render()
inren.Start()
      

    
