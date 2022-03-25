# Basics
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

# Used for function definition, integration
from sympy import *

# Used for fitting curves
from scipy import optimize

# LU, DET
from scipy.linalg import lu, lu_factor, lu_solve, det

# Jacobi, SOR
from numpy import array, zeros, diag, diagflat, dot



class Solution:

    def __init__(self):
        pass


    # Problem1 helper functions

    # Problem2 helper functions
    def construct_A(self,datapoints,order):
        # k + c2x = lny-lnx

        res_list = []
        for pair in datapoints:
            x = pair[0]
            tmp_list = []
            for t in range(order, -1, -1):
                tmp_list.append(x ** t)
            res_list.append(tmp_list)

        return np.array(res_list)


    def construct_b(self,datapoints,option="polynomial"):
        b_res = []
        for pair in datapoints:
            y = pair[1]
            x = pair[0]
            if option == "exponential":
                b_res.append(math.log(y) - math.log(x))
            elif option == "polynomial":
                b_res.append(y)

        return np.array(b_res)


    def normalEquation(self, A, b):
        # x = (A^T*A)^(-1)*(A^T*y)

        # 1. Obtain Normal Equation Solution
        x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))

        # 2. Compute RMSE
        # 2.1 b is the true value, Ax = b_hat is the LS estimation
        rsme = np.sqrt(np.mean(np.square(np.dot(A, x) - b)))
        print("Solution: {}".format(x))
        print("RMSE: {}".format(rsme))
        return x, rsme


    def polynomialFittingPackage(self, f, datapoints):
        x = np.array(list(map(lambda x: x[0], datapoints)))
        y = np.array(list(map(lambda x: x[1], datapoints)))

        # scipy.optimize.curve_fit
        result = optimize.curve_fit(f, x, y)
        print(result)


        plt.plot(x, y, "o", label="datapoints")

        x_points = np.linspace(x[0], x[-1], 1000)
        y_hat = f(x_points, result[0][0], result[0][1])
        plt.plot(x_points, y_hat, "g", label="fitted")


        plt.legend()
        plt.show()
        return


    # Problem3 helper functions
    def formatPoly(self,order,coeff):
        """
        Format the expression for the poly interpolation
        :param order:
        :param coeff:
        :return: independent_variable, function, formatted_function
        """
        order_list = [i for i in range(order,-1,-1)]
        x = symbols('x')
        f = 0
        for order,coeff in zip(order_list,coeff):
            f += coeff*(x**order)

        return x,f,str(f)

    def plotInterpolation(self,data_point,func,x0):
        x_list = [pair[0] for pair in data_point]
        y_list = [pair[1] for pair in data_point]
        x_poly = np.linspace(data_point[0][0],data_point[-1][0],60)
        y_poly = [float(func.evalf(subs={x0:x})) for x in x_poly]
        plt.plot(x_list,y_list,"o",label="interpolation points")
        plt.plot(x_poly,y_poly,label="poly_interpolation")
        plt.legend()
        plt.show()




    def problem1(self):
        pass



    def problem2(self,datapoints,f=None,option="normal"):
        assert datapoints is not None

        if option == "normal":

            # Adjust the x axis to lie in domain
            datapoints = [(1, 0.227407),
                          (2, 0.535561),
                          (3, 0.355028),
                          (4, 0.135292),
                          (5, 0.034924)]

            x_data = np.array(list(map(lambda x:x[0],datapoints)))
            y_data = np.array(list(map(lambda x:x[1],datapoints)))


            # 1. Construct AX=b with order = 1
            A = self.construct_A(datapoints,1)
            b = self.construct_b(datapoints,"exponential")

            # 2 and 3 Use Normal Equation to solve the linear system
            x,rsme = self.normalEquation(A,b)

            k, c2 = x
            c1 = np.exp(k)

            f = lambda t, a, b: a * t * np.exp(b * t)

            # 画出原始数据点
            plt.plot(x_data,y_data , "o", label="datapoints")


            # 画出拟合后的图像
            x_points = np.linspace(x_data[0], y_data[-1], 100)
            y_hat = f(x_points, c1, c2)
            plt.plot(x_points, y_hat, "g", label="fitted")

            plt.legend()
            plt.show()

        else:
            assert f is not None
            self.polynomialFittingPackage(f,datapoints)






    def problem3(self,datapoints):
        A = self.construct_A(datapoints,4)
        b = self.construct_b(datapoints,"polynomial")

        # Coefficient
        coeff = np.linalg.solve(A, b)
        print(coeff)

        # Plot the graph
        # x0 is the independent variables, func is the interpolation function
        x0, func, formatted_poly = self.formatPoly(4,coeff)
        self.plotInterpolation(datapoints, func, x0)

        print(formatted_poly)





if __name__ == "__main__":
    midterm = Solution()

    # Problem 2
    datapoints = [(-2,0.227407),
                  (-1,0.535561),
                  (0,0.355028),
                  (1,0.135292),
                  (2,0.034924)]

    f = lambda t, a, b: a * t * np.exp(b * t)
    midterm.problem2(datapoints,f,"normal")



    # Problem 3
    # midterm.problem3(datapoints)