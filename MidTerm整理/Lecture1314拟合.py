import numpy as np
import matplotlib.pyplot as plt

import math
import scipy
from sympy import *

from scipy import optimize
from scipy.linalg import lu, lu_factor, lu_solve, det

# Jacobi矩阵, SOR
from numpy import array, zeros, diag, diagflat, dot



class FittingCurve:

    def __init__(self):
        pass

    def computeAdjacencyMatrix(self,A,b,compute_error=True):
        pass


    # Normal Equation 求解, 返回最小二乘解和rsme
    def NormalEquation(self,A,b):

        # x = (A^T*A)^(-1)*(A^T*y)
        # 1. 获取一个Normal Equation的解
        x = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,b))

        # 2. 计算RMSE
        # 2.1 b 是实际的解, Ax = b_hat 是最小二乘解
        rsme = np.sqrt(np.mean(np.square(np.dot(A,x)-b)))
        print("Solution: {}".format(x))
        print("RMSE: {}".format(rsme))
        return x, rsme


    # 多项式拟合,手写， 一般是order小于点的个数，会欠拟合
    def polynomialModelFitting(self,datapoints, order):
        # a+bx+cx^2 的 order = 2

        x_points = np.array(list(map(lambda x:x[0],datapoints)))
        y_points = np.array(list(map(lambda x:x[1],datapoints)))

        # 形状因为包括了0阶，1阶，2阶，所以有三列
        A = np.zeros((x_points.shape[0],order+1))

        for i in range(order+1):
            A[:,i] = x_points**i

        x,rsme = self.NormalEquation(A,y_points)
        print(x,rsme)


    # 用于fit exponential
    def exponentModelFitting(self,func,):
        # y = c1e^(c2t)
        pass


    # Numpy实现功能
    def polynomialFittingPackage(self,f,datapoints):
        # https://www.itranslater.com/qa/details/2325702720658867200
        x = np.array(list(map(lambda x: x[0], datapoints)))
        y = np.array(list(map(lambda x: x[1], datapoints)))

        # 为了拟合y = A + B log x，只需对y（log x）拟合y
        # x_hat = np.polyfit(np.log(x), y, 1)
        # print(x_hat)


        # 使用scipy.optimize.curve_fit来拟合, 不需要进行坐标轴的转换, 函数接收三个参数t,a,b,  t是自变量，之后的参数都是系数
        result = optimize.curve_fit(f,x,y)

        # 画出原始数据点
        plt.plot(x,y,"o",label="datapoints")


        # 画出拟合后的图像
        x_points = np.linspace(x[0],x[-1],1000)
        y_hat = f(x_points,result[0][0],result[0][1])
        plt.plot(x_points,y_hat,"g",label="fitted")

        # print(result)
        plt.legend()
        plt.show()



if __name__ == "__main__":
    fitting = FittingCurve()


    # 最小二乘法
    # A = np.array([[1,1],[1,-1],[1,1]])
    # b = np.array([2,1,3])
    # fitting.NormalEquation(A,b)


    # Line Fitting
    datapoints = [(-1,1),(0,0),(1,0),(2,-2)]
    # 想要fit一个二阶项
    fitting.polynomialModelFitting(datapoints,2)


    f = lambda t,a,b: a*np.exp(b*t)
    # x = np.array([10, 19, 30, 35, 51])
    # y = np.array([1, 7, 20, 50, 79])

    # datapoints3 = [*zip(list(x),list(y))]
    # datapoints2 = [*zip([1,2,4,8,12,15,19,23,27,29,30,32,33],[2250,2500,5000,29000,120000,275000,1180000,3100000,7500000,24000000,42000000,220000000,410000000])]
    # # print(datapoints2)
    # fitting.polynomialFittingPackage(f,datapoints2)





