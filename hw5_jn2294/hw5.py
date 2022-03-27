# Basics
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import misc
import sympy


# Used for fitting curves
from scipy import optimize

# LU, DET
from scipy.linalg import lu, lu_factor, lu_solve, det

# Jacobi, SOR
from numpy import array, zeros, diag, diagflat, dot


class HW5:
    def __init__(self,mu=0,sigma=1,func=None):
        self.mu = mu
        self.sigma= sigma
        if func is None:
            self.func = lambda x, mu=0, sigma=1: (1 / (np.sqrt(2 * np.pi) * sigma)) * (
                    np.exp((-1 / 2) * np.power((np.log(x) - mu)/sigma,2)) / x)
        else:
            self.func = func
        self.x_points = np.linspace(0.1,5,100)
        self.y_points = self.func(self.x_points,self.mu,self.sigma)


    @staticmethod
    def construct_A(data_point, order):
        """
        Construct the coefficient matrix
        :param data_point: data in (x,y) point pair
        :param order: The order of the polynomial
        :return: numpy matrix
        """
        res_list = []
        for pair in data_point:
            x = pair[0]
            tmp_list = []
            for t in range(order, -1, -1):
                tmp_list.append(x ** t)
            res_list.append(tmp_list)

        return np.array(res_list)


    @staticmethod
    def construct_b(data_point,option="polynomial"):
        """
        Construct the vector b
        :param data_point: raw point data
        :param option:determine to use polynomial or exponential
        :return: numpy vector
        """
        b_res = []
        for pair in data_point:
            y = pair[1]
            x = pair[0]
            if option == "exponential":
                b_res.append(math.log(y) - math.log(x))
            elif option == "polynomial":
                b_res.append(y)

        return np.array(b_res)




    # Helper function for Chebyshev Interpolation point calculation
    def chebyshevInterpolation(self,a,b,n):
        """
        Chebyshev Interpolation calculation
        :param a: Start Point
        :param b: End Point
        :param n: The number of points that we want to calculate
        :return:
        """
        x_list = [(b+a)/2+((b-a)/2)*np.cos((2*i-1)*np.pi/(2*n)) for i in range(1,n+1)]
        x_list.reverse()
        return x_list


    def formatPolyFunction(self,order=None,coeff=None,mode="lambda"):
        """
        Format the expression for the poly interpolation
        :param order: The order of the polynomial function
        :param coeff: The coefficient of the poly function
        :return: independent_variable, function, formatted_function
        """
        if mode =="lambda":
            def outer(x,order_list,coeff):
                res = 0
                for order,coeff in zip(order_list,coeff):
                    res += coeff*(np.power(x,order))
                return res

            # return a lambda function
            return outer
        else:
            order_list = [i for i in range(order, -1, -1)]
            x = sympy.symbols('x')
            f = 0
            for order, coeff in zip(order_list, coeff):
                f += coeff * (x ** order)

            return x, f, str(f)

    def plotInterpolation(self, data_point, func, x0):
        x_list = [pair[0] for pair in data_point]
        y_list = [pair[1] for pair in data_point]
        x_poly = np.linspace(data_point[0][0], data_point[-1][0], 100)
        y_poly = [float(func.evalf(subs={x0: x})) for x in x_poly]
        plt.plot(x_list, y_list, "ro",label="original")
        plt.plot(x_poly, y_poly, label="poly_interpolation")
        plt.legend()
        plt.show()


    def polynomialErrorFunctionPlot(self,x_inter_points,func=None):
        """
        This function defines the error function plot for the interpolation
        :param x_inter_points: the interpolation points
        :return:
        """
        if func is None:
            myfunc = self.func
        else:
            myfunc = func


        # Order of derivative
        order = len(x_inter_points)
        multiple = 1 / (math.factorial(order))

        # 4.2.2 Define error function
        x = sympy.symbols("x")
        errorfunc = 1

        for i in range(order):
            errorfunc *= (x - x_inter_points[i])
            errorfunc *= multiple

        # Here we find c that maximizes the f^(6)(c)
        upper_boundary = -float("inf")
        upper_boundaries = []
        for inter_point in x_inter_points:
            upper_boundaries.append((inter_point,misc.derivative(myfunc, x0=inter_point, dx=0.5e-2, n=6,args=(0,1),order=7)))
            upper_boundary = max(misc.derivative(myfunc, x0=inter_point, dx=1e-4, n=6,args=(0,1),order=7),upper_boundary)
        errorfunc *= upper_boundary

        return x,errorfunc,upper_boundaries


    # Normal Equation Solver, return Least Square Solution and RSME
    def normalEquation(self, A, b):
        # x = (A^T*A)^(-1)*(A^T*b)
        # 1. Solve the Normal Equation
        x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))

        # 2. Compute RSME
        # 2.1 b is the actual solution, Ax = b_hat is the least square solution
        rsme = np.sqrt(np.mean(np.square(np.dot(A, x) - b)))
        return x, rsme


    # Polynomial fitting
    def polynomialModelFitting(self, datapoints, order):
        A = self.construct_A(datapoints,order)
        b = self.construct_b(datapoints,"polynomial")

        x, rsme = self.normalEquation(A, b)
        return x,rsme


    # Problem's main functions
    def problem1(self,start=0.1,end=5,num_points=6,data_points=None):

        # 1. Determine interpolation points

        if data_points is not None:
            data_point = data_points
            x_inter = np.array(list(map(lambda x:x[0],data_point)))
            y_inter = self.func(x_inter, self.mu, self.sigma)
        else:
            x_inter = np.linspace(start, end, num_points)
            y_inter = self.func(x_inter, self.mu, self.sigma)
            data_point = [*zip(list(x_inter),list(y_inter))]

        # 1.1 Since we have 6 points, the 5 order polynomial is what we need for unique solution
        order = num_points-1
        order_list = [i for i in range(order, -1, -1)]

        # 1.2 Construct linear system
        A = self.construct_A(data_point,order)
        b = self.construct_b(data_point)


        # 2. Solution for co-efficient for polynomial interpolation
        coeff = np.linalg.solve(A,b)
        print(coeff)

        # 2.1 The polynomial interpolation function
        interpolated_function = self.formatPolyFunction()



        # 3. Try to draw the full function plot
        x_inter_full = np.linspace(start,end,1000)
        y_inter_full = interpolated_function(x_inter_full,order_list,coeff)


        # 4. Draw the interpolation plot and the error plot
        fig,ax = plt.subplots(2,1)

        # 4.1 Draw interpolation point
        ax[0].plot(x_inter,y_inter,"ro",label="interpolation points")

        # 4.2 Draw the interpolated function
        ax[0].plot(x_inter_full,y_inter_full,label="interpolated function")


        # 4.3 Draw interpolation error plot
        # f(x) - P(x) = (x-x1)(x-x2)(x-x3)(x-x4)(x-x5)(x-x6)/6!*f^(6)(c)
        # 4.2.1 Defining functions
        x_sympy,func_sympy_error,upper_boundaries = self.polynomialErrorFunctionPlot(x_inter)
        error_plot_y = []

        # 4.2.2 Draw Error Plot
        x_points = np.linspace(start, end, 100)

        for error_x in x_points:
            error_plot_y.append(float(func_sympy_error.evalf(subs={x_sympy: error_x})))

        ax[1].plot(x_points, error_plot_y, label="Interpolation Error Plot")

        # 4.3 Compute error upper boundary
        for upper_boundary in upper_boundaries:
            print("Upper boundaries at x={0} is {1}".format(upper_boundary[0],upper_boundary[1]))


        # 5. Adjust the axes
        ax[0].set_title("Interpolation Plot")
        ax[1].set_title("Interpolation Error Plot")
        ax[0].set_xticks(x_inter)
        ax[0].legend()
        ax[1].legend()

        plt.subplots_adjust(wspace=1, hspace=1)
        plt.show()


    def problem2(self,start=0.1,end=5,num_points=6):
        x_cheb = self.chebyshevInterpolation(start,end,num_points)
        x_cheb_vec = np.array(x_cheb)

        y_cheb_vec = self.func(x_cheb_vec,0,1)
        data_points = [*zip(list(x_cheb_vec),list(y_cheb_vec))]


        # Reuse Problem 1, Plot the error, plot the interpolation results
        self.problem1(x_cheb[0],x_cheb[-1],len(x_cheb),data_points)


    def problem3(self,start=0.1,end=5,num_points=6):
        x_inter = np.linspace(start, end, num_points)
        y_inter = self.func(x_inter, self.mu, self.sigma)
        data_points = [*zip(list(x_inter), list(y_inter))]

        order = 3
        x,rsme = self.polynomialModelFitting(data_points, order = order)

        # order_list = [i for i in range(order, -1, -1)]
        x0, func, formatted_func = self.formatPolyFunction(order=3,coeff = x,mode="sympy")


        # 3. Try to draw the full function plot
        self.plotInterpolation(data_points, func, x0)
        print("Coefficient:{}".format(x))
        print("RSME:{}".format(rsme))
        return x,rsme


    def problem4(self,start=0.1,end=5,num_points=6):
        x_inter = np.linspace(start, end, num_points)
        y_inter = self.func(x_inter, self.mu, self.sigma)
        datapoints = [*zip(list(x_inter), list(y_inter))]

        # 1. Construct AX=b with order = 1
        A = self.construct_A(datapoints, 1)
        b = self.construct_b(datapoints, "exponential")

        # 2 Use Normal Equation to solve the linear system
        x, rsme = self.normalEquation(A, b)

        c2, k = x
        c1 = np.exp(k)

        print("c1:{0},c2:{1},rsme:{2}".format(c1,c2,rsme))

        f = lambda t, a, b: a * t * np.exp(b * t)

        # 画出原始数据点
        plt.plot(x_inter, y_inter, "ro", label="datapoints")

        # 画出拟合后的图像
        x_points = np.linspace(x_inter[0], x_inter[-1], 100)
        y_hat = f(x_points, c1, c2)
        plt.plot(x_points, y_hat, label="power_law")

        plt.legend()
        plt.show()



if __name__ == "__main__":
    hw5 = HW5()

    # hw5.problem1()

    # hw5.problem2()

    hw5.problem3()

    hw5.problem4()