import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg,integrate
from sympy import *

import time

class HW2():

    x = symbols('x')
    func1 = sin(x**5) + x**3
    func2 = 5*x**4*cos(x**5) + 3*x**2

    start = 1.5
    end= 2.0


    def __init__(self):
        self.first_diff = diff(self.func1, self.x)
        # self.first_diff = self.func2

    # 2.1 Central Difference Method
    def cdm(self,h,x0,speed_test=False,dis_error = False):
        """
        :param h: the step size/ solution
        :param x0: center point
        :param speed_test: whether to take speed test, switch off to exclude the effect of print statement
        :param dis_error: switch off for speed test to exclude the effect of print estimation error
        :return:
        """

        res = (self.f(x0+h) - self.f(x0-h))/(2*h)

        if not speed_test:
            print("f'({0}) under h={1} is {2}".format(x0,h,res))

            if dis_error:
                correct = self.f_diff(x0)
                print("The correct result is {3}, the error is {4}"
                      .format(x0, h, res,correct,np.abs(correct - res)))


    # 2.2 Forward Difference Method
    def fdm(self,h,x0,speed_test=False,dis_error = False):
        """
        :param h: the step size/ resolution
        :param x0: center point
        :param speed_test: whether to take speed test, switch off to exclude the effect of print statement
        :param dis_error: switch off for speed test to exclude the effect of print estimation error
        :return:
        """

        res = (self.f(x0 + h) - self.f(x0)) /  h

        if not speed_test:
            print("f'({0}) under h={1} is {2}".format(x0, h, res))

            if dis_error:
                correct = self.f_diff(x0)
                print("The correct result is {3}, the error is {4}"
                      .format(x0, h, res,correct,np.abs(correct - res)))

    # 2.3 Midpoint Rule
    def midpoint(self,n, speed_test=False,dis_error = False):
        """
        :param n: Number of intervals
        :param speed_test: whether to take speed test, switch off to exclude the effect of print statement
        :param dis_error: switch off for speed test to exclude the effect of print estimation error
        :return:
        """
        delta_x = (self.end-self.start)/n
        sep_points = np.linspace(self.start,self.end,n+1)
        v1 = np.array(sep_points[:-1])
        v2 = np.array(sep_points[1:])
        mid_points = (v1+v2)/2
        result = delta_x*sum(map(lambda x:self.f_diff(x),mid_points))

        if not speed_test:
            print("The integration of the f'(x) under n={0} on [{1},{2}] is {3}".format(n,1.5,2.0,result))

            if dis_error:
                correct = 4.21009627762213
                print("The correct result is {0}, the error is {1}"
                      .format( correct, np.abs(correct - result)))

    # 2.4 Simpson's 1/3 Rule
    def simpson(self,n,speed_test=False,dis_error = False):
        """
        :param n: Number of intervals
        :param speed_test: whether to take speed test, switch off to exclude the effect of print statement
        :param dis_error: switch off for speed test to exclude the effect of print estimation error
        :return:
        """
        delta_x = (self.end-self.start)/n
        points = np.linspace(self.start,self.end,n+1)
        if n > 2:
            coeff_vec = np.array([1]+[4,2]*((n-2)//2)+[4]+[1],dtype=float)
        else:
            coeff_vec = np.array([1,4,1],dtype=float)
        function_value = np.array([self.f_diff(point) for point in points],dtype=float)
        result = delta_x*coeff_vec.dot(function_value)/3

        if not speed_test:
            print("The integration of f'(x) under n={0} on [{1},{2}] is {3}".format(n,self.start,self.end,result))

            if dis_error:
                correct = 4.21009627762213
                print("The correct result is {0}, the error is {1}"
                      .format( correct, np.abs(correct - result)))

    # 2.5 Gaussian Quadrature
    def gaussian(self,n,speed_test=False,dis_error = False):
        """
        :param n: Number of points
        :param speed_test: whether to take speed test, switch off to exclude the effect of print statement
        :param dis_error: switch off for speed test to exclude the effect of print estimation error
        :return:
        """
        # x = 1.75+0.25xd, where xd ~ [-1,1]
        a = (self.end+self.start)/2  # 1.75
        b = (self.end-self.start)/2  # 0.25
        if n == 3:
            I = float((5 / 9 * self.f_diff(a + b * sqrt(3 / 5)) +
                        5 / 9 * self.f_diff(a - b * sqrt(3 / 5)) +
                        8 / 9 * self.f_diff(a + b * 0)) * b)
        elif n == 4:
            I = float(((18 + sqrt(30)) / 36 * self.f_diff(a + b * sqrt(3 / 7 - 2 / 7 * sqrt(6 / 5))) +
                    (18 + sqrt(30)) / 36 * self.f_diff(a - b * sqrt(3 / 7 - 2 / 7 * sqrt(6 / 5))) +
                    (18 - sqrt(30)) / 36 * self.f_diff(a + b * sqrt(3 / 7 + 2 / 7 * sqrt(6 / 5))) +
                    (18 - sqrt(30)) / 36 * self.f_diff(a - b * sqrt(3 / 7 + 2 / 7 * sqrt(6 / 5)))) * b)

        elif n == 5:
            I = float(((322 + 13 * sqrt(70)) / 900 * self.f_diff(a + 1 / 3 * b * sqrt(5 - 2 * sqrt(10 / 7))) +
                        (322 + 13 * sqrt(70)) / 900 * self.f_diff(a - 1 / 3 * b * sqrt(5 - 2 * sqrt(10 / 7))) +
                        (322 - 13 * sqrt(70)) / 900 * self.f_diff(a + 1 / 3 * b * sqrt(5 + 2 * sqrt(10 / 7))) +
                        (322 - 13 * sqrt(70)) / 900 * self.f_diff(a - 1 / 3 * b * sqrt(5 + 2 * sqrt(10 / 7))) +
                        128 / 225 * self.f_diff(a + b * 0)) * b)
        else:
            I = 0

        if not speed_test:
            print("The integration of f'(x) under n={0} on [{1},{2}] is {3}".format(n,self.start,self.end,I))

            if dis_error:
                correct = 4.21009627762213
                print("The correct result is {0}, the error is {1}"
                      .format( correct, np.abs(correct - I)))

    def f(self,value):
        """
        Calculate the value of a given function at specified x
        :param func: sympy function object
        :param value: value of independent variable
        :return: None
        """
        return float(self.func1.evalf(subs={self.x: value}))

    def f_diff(self,value):
        return float(self.first_diff.evalf(subs={self.x: value}))


    def test(self):
        print("Testing Central Difference Method !")
        self.cdm(10e-3, 1.5,dis_error=True)
        self.cdm(10e-4, 1.5,dis_error=True)
        self.cdm(10e-3, 1.7,dis_error=True)
        self.cdm(10e-4, 1.7,dis_error=True)
        print("####################################")

        print("Testing Forward Difference Method !")
        self.fdm(10e-3, 1.5,dis_error=True)
        self.fdm(10e-4, 1.5,dis_error=True)
        self.fdm(10e-3, 1.7,dis_error=True)
        self.fdm(10e-4, 1.7,dis_error=True)
        print("####################################")

        print("Testing Midpoint Rule !")
        self.midpoint(10,dis_error=True)
        self.midpoint(10**2,dis_error=True)
        self.midpoint(10**3,dis_error=True)
        print("####################################")

        print("Testing Simpson's 1/3 Rule !")
        self.simpson(10,dis_error=True)
        self.simpson(10**2,dis_error=True)
        self.simpson(10**3,dis_error=True)
        print("####################################")

        print("Testing Gaussian Quadrature !")
        self.gaussian(3,dis_error=True)
        self.gaussian(4,dis_error=True)
        self.gaussian(5,dis_error=True)
        print("####################################")


    # Speed test for computing differentiation
    def speed_test1(self):
        print("Speed testing central difference method!")
        times = []

        for i in range(200):
            start = time.time()
            self.cdm(10e-3, 1.5,speed_test=True)
            self.cdm(10e-4, 1.5,speed_test=True)
            self.cdm(10e-3, 1.7,speed_test=True)
            self.cdm(10e-4, 1.7,speed_test=True)
            times.append(time.time()-start)

        print("Central Difference Method takes {0} on average".format(np.mean(times)))

        print("Speed testing forward difference method!")
        times = []
        for i in range(200):
            start = time.time()
            self.fdm(10e-3, 1.5,speed_test=True)
            self.fdm(10e-4, 1.5,speed_test=True)
            self.fdm(10e-3, 1.7,speed_test=True)
            self.fdm(10e-4, 1.7,speed_test=True)
            times.append(time.time() - start)

        print("Forward Difference Method takes {0} on average".format(np.mean(times)))


    #Speed test for computing integration
    def speed_test2(self):
        print("Testing Midpoint Rule")
        times = []
        for i in range(200):
            start = time.time()
            self.midpoint(10,speed_test=True)
            self.midpoint(10 ** 2,speed_test=True)
            self.midpoint(10 ** 3,speed_test=True)
            times.append(time.time()-start)
        print("Time taken on average: {}".format(np.mean(times)))


        print("Testing Simpson Rule")
        times = []
        for i in range(200):
            start = time.time()
            self.simpson(10 ,speed_test=True)
            self.simpson(10 ** 2 ,speed_test=True)
            self.simpson(10 ** 3 ,speed_test=True)
            times.append(time.time() - start)
        print("Time taken on average: {}".format(np.mean(times)))



        print("Testing Guassian Quadrature")
        times = []
        for i in range(200):
            start = time.time()
            self.gaussian(3,speed_test=True)
            self.gaussian(4,speed_test=True)
            self.gaussian(5,speed_test=True)
            times.append(time.time() - start)
        print("Time taken on average: {}".format(np.mean(times)))


if __name__ =="__main__":

    hw2 = HW2()

    # hw2.test()
    #
    # hw2.speed_test1()
    #
    # hw2.speed_test2()

    f = lambda x:5*x**4*np.cos(x**5) + 3*x**2
    print(scipy.integrate.quadrature(f,1.5,2.0))