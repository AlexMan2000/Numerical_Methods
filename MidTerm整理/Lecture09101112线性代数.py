import numpy as np
import matplotlib.pyplot as plt

import math
import scipy
from sympy import *


from scipy.linalg import lu, lu_factor, lu_solve, det

# Jacobi矩阵, SOR
from numpy import array, zeros, diag, diagflat, dot


class LinearAlgebra:

    def __init__(self):
        self.TOL = 1e-10

    # 求解
    # 高斯消元法
    def guassianElimination(self, A, b,square=True):
        # 可以用于非方阵，但是可能有多解
        b = b[:,np.newaxis]
        if square:
            a = np.hstack((A, b))  # augmented matrix, self-written
            n = A.shape[0]
            x = np.zeros(n)
            for i in range(n):
                if a[i][i] == 0.0:
                    print('Divide by zero detected!')
                # Compare the ratio of adjacent two rows
                for j in range(i + 1, n):
                    ratio = a[j][i] / a[i][i]

                    for k in range(n + 1):  # since we have augmented matrix
                        a[j][k] = a[j][k] - ratio * a[i][k]

            # last index's solution
            x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]

            for i in range(n - 2, -1, -1):
                x[i] = a[i][n]

                # 需要减掉之前算过的每一个解
                for j in range(i + 1, n):
                    # a这里就是系数
                    x[i] = x[i] - a[i][j] * x[j]

                # a[i][i]就是(i,i)元素的值
                x[i] = x[i] / a[i][i]

            print(x)
            return x
        else:
            return 0


    # 一般用于方阵
    def LUFactorization(self,A,b,option="auto"):
        # 一般用于方阵
        # https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html
        if option == "auto":
            LU, p = lu_factor(A)
            # scipy的包
            x = lu_solve((LU, p), b)
            print(x)
        elif option == "self":
            # https://www.freesion.com/article/8791842738/
            """
            并非所有矩阵都能进行LU分解，能够LU分解的矩阵需要满足以下三个条件：
            矩阵是方阵（LU分解主要是针对方阵）；
            矩阵是可逆的，也就是该矩阵是满秩矩阵，每一行都是独立向量；
            消元过程中没有0主元出现，也就是消元过程中不能出现行交换的初等变换。
            """
            def LU(A):
                '''
                生成值全位0的U矩阵，和单位矩阵L
                '''
                L = np.eye(len(A))
                U = np.zeros(np.shape(A))
                for r in range(1, len(A)):  # 求U的第一行和L的第一列
                    U[0, r - 1] = A[0, r - 1]
                    L[r, 0] = A[r, 0] / A[0, 0]
                U[0, -1] = A[0, -1]
                for r in range(1, len(A)):  # 先求U再求L
                    for i in range(r, len(A)):
                        delta = 0
                        for k in range(0, r):  # 求∑(???∗???)
                            delta += L[r, k] * U[k, i]
                        U[r, i] = A[r, i] - delta

                        for i in range(r + 1, len(A)):  # 求L矩阵
                            theta = 0
                            for k in range(0, r):  # 求∑(?i?∗??r)
                                theta += L[i, k] * U[k, r]
                            L[i, r] = (A[i, r] - theta) / U[r, r]
                return L, U

            def my_LUsolve(A, b):
                L, U = LU(A)  # 得到L和U
                '''print("L={}".format(L))
                print("U={}".format(U))'''
                # 求解线性方程LY=b
                n = len(A)
                y = np.zeros((n, 1))
                b = np.array(b).reshape(n, 1)  # 把b列表格式变成向量格式
                for i in range(len(A)):
                    t = 0
                    for j in range(i):
                        t += L[i][j] * y[j][0]
                    y[i][0] = b[i][0] - t
                # print("y={}".format(y))
                # 求解UX=Y
                X = np.zeros((n, 1))
                for i in range(len(A) - 1, -1, -1):
                    t = 0
                    for j in range(i + 1, len(A)):
                        t += U[i][j] * X[j][0]
                    t = y[i][0] - t
                    if t != 0 and U[i][i] == 0:
                        return 0
                    X[i] = t / U[i][i]
                # print("X={}".format(X))
                return X

            X = my_LUsolve(A, b)
            print(X.squeeze())
            return X

        # https: // johnfoster.pge.utexas.edu / numerical - methods - book / LinearAlgebra_LU.html
        # n = A.shape[0]
        # L = np.eye(n,dtype=np.double)
        # U = A.copy()
        #
        # # L = [[1,2,3],
        # #      [4,5,6],
        # #      [7,8,9]]
        #
        # for i in range(n):
        #     # 就是行倍数， 这里获取的是0维张量, array([1,2,3])
        #     factor = U[i+1:,i] / U[i,i]
        #
        #     # 批量赋值, L[i+1:,i]就是array([1,2])
        #     L[i+1:,i] = factor
        #
        #     # 对下面的所有行做相减操作，广播机制, factor[:,np.newaxis]就是array([[1],[2]])
        #     U[i+1:] -= factor[:,np.newaxis]


    # 一般用于方阵
    def cramer(self,A,b):
        # 只能用于方阵
        assert A.shape[0] == A.shape[1], "Need square coefficient matrix for linear equation system"
        n = A.shape[0]
        x = np.zeros(n)
        for i in range(A.shape[0]):
            acopy = A.copy()
            acopy[:, i] = b
            x[i] = det(acopy) / det(A)

        print(x)
        return x





    # 稀疏矩阵求解，迭代法
    def JacobiMethod(self,A,b,N=25,x0=None):
        # 简称 DR 分解, D 是一个对角矩阵，R是剩下的元素
        """Solves the equation Ax=b via the Jacobi iterative method.
        https://www.quantstart.com/articles/Jacobi-Method-in-Python-and-NumPy/"""
        # Create an initial guess if needed
        if x0 is None:
            x = zeros(len(A[0]))
        else:
            x = x0

        # Create a vector of the diagonal elements of A
        # and subtract them from A
        D = diag(A)
        R = A - diagflat(D)

        # Iterate for N times, 矩阵相除就是乘以逆
        for i in range(N):
            x = (b - dot(R, x)) / D

        print(x)
        return x


    def GuassSeidelMethod(self,A,b,N=25,x0=None):
        """
        https://www.geeksforgeeks.org/gauss-seidel-method/
        :param A:
        :param b:
        :param N:
        :param x:
        :return:
        """
        if x0 is None:
            x = zeros(len(A[0]))
        else:
            x = x0

        solution_by_step = []
        for i in range(N):
            # Finding length of a(3)
            n = len(A)
            # for loop for 3 times as to calculate x, y , z
            for j in range(0, n):
                # temp variable d to store b[j]
                d = b[j]

                # to calculate respective xi, yi, zi
                for i in range(0, n):
                    if (j != i):
                        d -= A[j][i] * x[i]
                # updating the value of our solution
                x[j] = d / A[j][j]

            solution_by_step.append(x)
            # returning our updated solution

        print(solution_by_step[-1])
        return solution_by_step[-1]


    def SOR(self,A,b,N=25,w=1.5):
        # https://stackoverflow.com/questions/52951533/following-code-i-have-made-an-sor-iteration-in-python
        x0 = np.zeros(len(A[0]))

        if (w <= 0 or w >= 2):
            print('w should be inside (0, 2)')
            step = -1
            x = float('nan')
            return
        n = b.shape

        x= x0

        """
        When multiplying a sparsa matrix by a array you should not use: np.dot(np.array(a), x)) but a.dot(x)
        """
        for step in range(1, N):
            for i in range(n[0]):
                new_values_sum = np.dot(A[i, :i], x[:i])
                old_values_sum = np.dot(A[i, i + 1:], x0[i + 1:])
                x[i] = (b[i] - (old_values_sum + new_values_sum)) / A[i, i]
                x[i] = np.dot(x[i], w) + np.dot(x0[i], (1 - w))
            if (np.linalg.norm(np.dot(A, x) - b) < self.TOL):
                break
            x0 = x

        print(x)
        return x





    # 正定矩阵
    # Methods for symmetric positive-definite matrices
    def CholeskyFactorization(self,A):
        """Performs a Cholesky decomposition of A, which must
            be a symmetric and positive definite matrix. The function
            returns the lower variant triangular matrix, L."""
        """
        https://www.quantstart.com/articles/Cholesky-Decomposition-in-Python-and-NumPy/
        """

        if not self.determineSymmetricPositive(A):
            print("Not symmetric positive definite!")
        n = len(A)

        # Create zero matrix for L
        L = np.zeros((n,n))

        # Perform the Cholesky decomposition
        for i in range(n):
            for k in range(i + 1):
                temp_sum = np.sum([L[i][j] * L[k][j] for j in range(k)],keepdims=True)

                # For diagonal elements
                if i == k:
                    L[i][k] = np.sqrt(A[i][i] - temp_sum)
                else:
                    L[i][k] = (1.0 / L[k][k] * (A[i][k] - temp_sum))
        print(L.T)
        return L.T


    # 只能对 对称正定系数矩阵求解
    def ConjugateGradient(self,A,x,b,N):
        # N 次迭代
        # 需要一个initial guess
        """
        https://towardsdatascience.com/complete-step-by-step-conjugate-gradient-algorithm-from-scratch-202c07fb52a8
        https://stackoverflow.com/questions/53665433/conjugate-gradient-implementation-python
        :return:
        """
        if not self.determineSymmetricPositive(A):
            print("Not symmetric positive definite!")
        r = b - A.dot(x)
        p = r.copy()
        for i in range(N):
            Ap = A.dot(p)
            alpha = np.dot(p, r) / np.dot(p, Ap)
            x = x + alpha * p
            r = b - A.dot(x)
            if np.sqrt(np.sum((r ** 2))) < self.TOL:
                print('Itr:', i)
                break
            else:
                beta = -np.dot(r, Ap) / np.dot(p, Ap)
                p = r + beta * p

        print(x)
        return x


    # Power Method计算eigenvalues
    def computeEigenValue(self,A,x0=None,N=10,TOL = 1e-10,option="power"):
        """
        Rationale: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter15.02-The-Power-Method.html
        Code:
        :param A:
        :param option:
        :return:
        """
        print(option)

        if option=="power":
            assert x0 is not None


            n = A.shape[0]
            x = x0

            # Power Method Implementation
            lambda_old = 1.0
            condition = True
            step = 1
            while condition:
                # Multiplying a and x
                x = np.matmul(A, x)

                # Finding new Eigen value and Eigen vector
                lambda_new = max(abs(x))

                x = x / lambda_new



                # Displaying Eigen value and Eigen Vector
                print('\nSTEP %d' % (step))
                print('----------')
                print('Eigen Value = %0.4f' % (lambda_new))
                print('Eigen Vector: ')
                for i in range(n):
                    print('%0.3f\t' % (x[i]))

                # Checking maximum iteration
                step = step + 1
                if step > N:
                    print('Not convergent in given maximum iteration!')
                    break

                # Calculating error
                error = abs(lambda_new - lambda_old)
                print('errror=' + str(error))
                lambda_old = lambda_new
                condition = error > TOL

        elif option=="auto":
            eigval,eigvec = np.linalg.eig(A)
            print("Dominant eigenvalue:{}".format(max(eigval)))
            print("Dominant eigenvector:{}".format(eigvec[:,np.argmax(eigval)]))


    # 是不是所有的特征值都大于零
    def determineSymmetricPositive(self,A):
        return np.all(np.linalg.eigvals(A)>0) and np.all(A.T==A)





if __name__ == "__main__":
    linear = LinearAlgebra()

    # A = np.array([[1,2,-1],[2,1,-2],[-3,1,1]])
    # b = np.array([[1],[2],[3]])
    # A = np.array([[3, 1, -1], [2, 4, 1], [-1, 2, 5]])
    # b = np.array([4,1,1])
    # A = np.array([[3, -1, 1], [-1, 3, -1], [1, -1, 3]])
    # b = np.array([4,1,1])
    #
    # linear.guassianElimination(A, b)
    # linear.cramer(A,b)
    # linear.LUFactorization(A,b,option="auto")
    #
    #
    #
    # linear.JacobiMethod(A,b,N=250)
    # linear.GuassSeidelMethod(A,b,N=250)
    # linear.SOR(A,b,N=250)
    #
    # A = np.array([[4, -2, 2],[-2,2,-4],[2,-4,11]])
    # linear.CholeskyFactorization(A)

    # Test1
    # A = np.array([[4,1],[1,3]])
    # b = np.array([1,2])
    # initial_guess = np.array([2,1])
    # linear.ConjugateGradient(A,initial_guess,b,3)


    # A = np.array([[2,2],[2,5]])
    # print(linear.determineSymmetricPositive(A))

    # A = np.array([[2,-12],[1,-5]])
    # x0 = np.array([1,1])

    A = np.array([[1, 3], [2, 2]])
    x0 = np.array([-5, 5])
    linear.computeEigenValue(A,option="auto")
    linear.computeEigenValue(A,x0, option="power")

