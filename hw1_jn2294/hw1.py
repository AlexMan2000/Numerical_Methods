import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy import linalg


class HW1:
    """
    Pre-defined functions
    """
    x = symbols('x')
    default_func = cos(x)-x**3


    def __init__(self,func=None,tol=1e-10, error=0.5e-4, true = 0.865474):
        """

        :param func: The function object defined using sympy library.
        :param tol: For those methods that involves rescaling intervals(i.e bisectionMethod).
        :param error: The absolute value between obtained root and true root.
        :param true: True root.
        """
        self.func = self.default_func if not func else func
        self.TOL = tol
        self.error = error
        self.true = true


    # 1.1
    def bisectionMethod(self,start,end):
        """
        Implementation for bisection method
        :param start: Left endpoint
        :param end:  Right endpoint
        :return: None
        """
        print("1.1 Performing bisection method!")
        iteration = 0
        while (end-start)/2 > self.TOL:
            iteration += 1
            mid = (start+end)/2
            if np.abs(mid -  float(self.true)) < self.error:
                self.printHelper(iteration,1, mid, np.abs(mid - float(self.true)), iteration * 10, 10)
                return mid

            if self.f(self.func,start)*self.f(self.func,mid) < 0:
                end = mid
            else:
                start = mid


    # 1.2
    def newtonMethod(self, max_iter = 10000, initial = 0.3):
        """
        Implementation for newton method
        :param max_iter: To prevent the possible divergence in the method
        :param initial: x0
        :return: None
        """

        print("1.2 Performing newton method !")

        # g'(x)
        first_diff_func = diff(self.func, self.x)


        iteration = 0
        t = initial
        while iteration < max_iter:
            iteration+=1
            if(np.abs(t-self.true)) < self.error:
                self.printHelper(iteration,1,t,np.abs(t-self.true),iteration*10,10)
                break

            # x_{i+1} = x_{i} - g(x)/g'(x)
            t = t - self.f(self.func,t)/self.f(first_diff_func,t)


    # 1.3
    def secantMethod(self,x0,x1,max_iter=10000):
        """
        Implementation for secant method
        :param x0: initial root
        :param x1: initial root
        :param max_iter: To prevent the possible divergence in the method
        :return: None
        """
        print("1.3 Performing secant method !")

        iteration = 0
        x0 = x0
        x1 = x1
        while iteration < max_iter:
            iteration+=1
            if(np.abs(x0-self.true)) < self.error:
                self.printHelper(iteration,1,x0,np.abs(x0-self.true),iteration*15,15)
                break

            # x_{i+1} = x_{i} - ( (x_{i]-x_{i-1})f(xi) / (f(xi)-f(x_{i-1})) )
            x0 = x1 - (x1-x0)*self.f(self.func,x1)/(self.f(self.func,x1)-self.f(self.func,x0))


    # 1.4
    def fixedPointMethod(self,max_iter=200,initial=0,fixed_point = 0.60352):
        """
        The implementation of fixedPointMethod
        :param max_iter: To prevent the possible divergence in the method
        :param initial: x0
        :param fixed_point: fc
        :return: None
        """
        print("1.4 Performing fixed point method !")

        iteration = 0
        x = initial

        x1 = symbols('x')
        alternative_func = real_root(cos(x1)-x1,3)

        x2 = symbols('x')
        alternative_func2 = cos(x2)/(x2**2+1)


        while iteration<max_iter:
            iteration+=1
            if(np.abs(x-fixed_point)) < self.error:
                self.printHelper(iteration,0,x,np.abs(x-fixed_point),iteration*3,3)
                return
            x = self.f(self.func,x)

        print("x_{n} = cos(x_{n-1})-x_{n-1}^3 failed. Maximum iteration reached, the fixed point method diverges !\n ")


        x = initial
        iteration = 0
        while iteration<max_iter:
            iteration+=1
            if(np.abs(x-fixed_point)) < self.error:
                self.printHelper(iteration,0,x,np.abs(x-fixed_point),iteration*4,4)
                return

            x = alternative_func.evalf(subs={x1:x})

        print("x_{n} = (cos(x_{n-1})-x_{n-1})^(1/3) failed. Maximum iteration reached, the fixed point method diverges !\n")


        x = initial
        iteration = 0
        while iteration < max_iter:
            iteration += 1
            if (np.abs(x - fixed_point)) < self.error:
                print(
                    "x_{n} = cos(x_{n-1})/(x_{n-1}^2+1) converged!\n")
                self.printHelper(iteration, 0, x, np.abs(x - fixed_point), iteration * 5, 5)
                return

            x = alternative_func2.evalf(subs={x1: x})

        print(
            "x_{n} = cos(x_{n-1})/(x_{n-1}^2+1) failed. Maximum iteration reached, the fixed point method diverges !\n")

    # 1.5
    def interpolation(self,data_point, order):
        """
        Implementation of polynomial interpolation
        :param data_point: The key-value pair form of data point
        :param order: The order of the polynomial interpolation
        :return: None
        """

        print("1.5 Performing interpolation !")

        A = self.construct_A(data_point,order)
        b = self.construct_b(data_point)
        x = linalg.solve(A,b)
        x0,func,formatted_func = self.formatPoly(order,x)
        print("The polynomial interpolation takes the form of:\n{}".format(formatted_func))

        self.plotInterpolation(data_point,func,x0)


    def plotInterpolation(self,data_point,func,x0):
        x_list = [pair[0] for pair in data_point]
        y_list = [pair[1] for pair in data_point]
        x_poly = np.linspace(1,5,60)
        y_poly = [float(func.evalf(subs={x0:x})) for x in x_poly]
        plt.plot(x_list,y_list,label="original")
        plt.plot(x_poly,y_poly,label="poly_interpolation")
        plt.legend()
        plt.show()


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


    @staticmethod
    def construct_A(data_point,order):
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
                tmp_list.append(x**t)
            res_list.append(tmp_list)

        return np.array(res_list)


    @staticmethod
    def construct_b(data_point):
        """
        Construct the vector b
        :param data_point: raw point data
        :return: numpy vector
        """
        return np.array([pair[1] for pair in data_point])


    def printHelper(self,iteration,option,root,error,ops_num,ops_each):
        """
        Help format the print.
        """
        print("Total number of iteration: {0};\nThe {1} found: {2};\nFinal Absolute Error: {3};\n"
              "The estimated number of floating operation: {4}, {5} operations(excluding comparisons) for each"
              " iteration.\n############################################\n"
              .format(iteration,"root" if option==1 else "fixed point",root,error,ops_num,ops_each))


    def f(self,func,value):
        """
        Calculate the value of a given function at specified x
        :param func: sympy function object
        :param value: value of independent variable
        :return: None
        """
        return float(func.evalf(subs={self.x: value}))


if __name__ =="__main__":
    # Test
    hw1 = HW1()

    # Test 1.1
    # hw1.bisectionMethod(0,1)

    # Test 1.2
    # hw1.newtonMethod()
    #
    # # Test 1.3
    # hw1.secantMethod(0,1)
    #
    # # Test 1.4
    # hw1.fixedPointMethod()
    #
    # # Test 1.5
    data_point = [(1,930.00),(2,918.40),(3,937.41),(4,829.10),(5,846.35)]
    hw1.interpolation(data_point,4)