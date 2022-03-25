import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()

from sympy import *


class NumericalMethod:
    x = symbols("x")

    def __init__(self):
        self.right = 0.865474
        self.TOL = 0.1e-10
        self.error = 0.5e-4

        self.newtonMemo = {}

    def generate_x(self):
        return self.x

    def bisectionMethod(self,start=0, end = 1,TOL=0.1e-10,max_iteration = 500):

        # function definition

        func = cos(self.x) - self.x**3

        a0 = start
        b0 = end
        iteration = 0

        while (b0-a0)/2 > self.TOL and iteration <= max_iteration:
            iteration += 1
            mid = (a0+b0)/2
            error = np.abs(mid - float(self.right))
            if error < 0.5e-4:
                self.printHelper(iteration,error,mid)
                return mid

            if self.f(func,mid)*self.f(func,b0) < 0:
                a0 = mid
            else:
                b0 = mid

        return "max_iteration_reached"


    def newtonMethod(self,max_iteration = 10000,initial = 0.3,explore=False):

        func = cos(self.x) - self.x ** 3
        first_diff = diff(func,self.x)

        iteration = 0
        x0 = initial
        while iteration <= max_iteration:
            iteration += 1
            x0 -= float(self.f(func,x0))/float(self.f(first_diff,x0))
            error = float(x0-self.right)
            if float(x0-self.right) < self.error:
                self.printHelper(iteration,error,x0)
                if explore:
                    return iteration
                else:
                    return x0

        print("max_iteration_reached")


    def exploreNewton(self):
        initial_choice = np.linspace(0.1,1,100)
        iteration_step_list = []
        for initial in initial_choice:
            iteration_step_list.append(self.newtonMethod(initial = initial,explore=True))

        plt.plot(initial_choice,iteration_step_list)
        plt.show()

    # 正确解法
    def secantMethod(self,max_iteration=10000,ini0 = 0, ini1 = 1):
        func = cos(self.x) - self.x**3

        x0 = ini0
        x1 = ini1
        iteration = 0

        while iteration <= max_iteration:
            iteration += 1
            error = np.abs(x0-self.right)
            # 由于x0最先被计算出来，因此我们只要6次就可以得到结果
            if error < self.error:
                self.printHelper(iteration,error,x1)
                return x0

            # 动态规划的时间优化方法，作业中是错误的方法
            if iteration % 2 == 0:
                x1 = x0 - (x0 - x1)*(self.f(func,x0)/float(self.f(func,x0)-self.f(func,x1)))
            else:
                x0 = x1 - (x1 - x0) * (self.f(func, x1) / float(self.f(func, x1) - self.f(func, x0)))


        return "max_iteration_reached"


    # 需要对函数做一定的变换才能收敛
    def fixedPoint(self,max_iteration=10000,init=0,answer = 0.60352):
        # sympy使用real_root求幂
        func_list = [cos(self.x) - self.x**3, real_root(cos(self.x)-self.x,3), cos(self.x)/(self.x**2+1)]


        for func in func_list:
            print("Trying function:{}".format(func))
            iteration = 0
            x1 = init
            while iteration <= max_iteration:
                iteration += 1
                x1 = self.f(func,x1)
                error = np.abs(x1-answer)
                if error < self.error:
                    print("Root found for function {}".format(func))
                    self.printHelper(iteration,error,x1)
                    return x1

            print("max_iteration_reached for func {}".format(func))


    # Choose the interpolate the points and plot out the desired graph
    def interpolation(self,data_points,mode="poly",):
        # Python 函数闭包
        # https: // blog.csdn.net / register_2 / article / details / 80225970
        x_points = list(map(lambda x:x[0],data_points))
        y_points = list(map(lambda x: x[1], data_points))
        print(x_points)
        print(y_points)
        num_points = len(x_points)
        memo = {}

        # 给多项式造轮子, anx^(n-1)+a(n-1)*x^(n-2)+...+a0*x^0
        def constructA(x_list):
            num_points = len(x_list)
            vstack_rows = []
            for i in range(len(x_list)):
                stacked_matrix = np.array([x_list[i]**j for j in range(num_points-1,-1,-1)])
                vstack_rows.append(stacked_matrix)
            A = np.vstack(vstack_rows)
            return A

            # 使用numpy的广播机制

        def constructB(y_true):
            return np.array(y_true).reshape(len(y_true),1)

        def newtonDividedDifference():
            n = len(x_points)
            res = newtonDividedDifferenceHelper(n,0,n-1,memo)
            memo[(0,n-1)] = res
            return memo

        def newtonDividedDifferenceHelper(n_length,start_x_index,end_x_index,memo):
            # 就是一个f[xi,...,xj]的长度
            if n_length == 1:
                res = y_points[start_x_index]
                memo[(start_x_index,end_x_index)] = res
                return res

            if memo.get((start_x_index,end_x_index),None) is not None:
                return memo[(start_x_index,end_x_index)]


            res = (newtonDividedDifferenceHelper(n_length - 1, start_x_index + 1, end_x_index,memo)-
            newtonDividedDifferenceHelper(n_length-1,start_x_index,end_x_index-1,memo))\
                  /(x_points[end_x_index]-x_points[start_x_index])

            memo[(start_x_index,end_x_index)] = res

            return res


        # 用于插值
        # x_list = np.linspace(1,num_points,num_points)
        # if func:
        #     y_true = [self.f(func,x) for x in x_list]


        # 用于画原图
        plt.scatter(x_points,y_points,label="original",color="red")


        # num_points个点，我们需要num_points-1次的多项式
        if mode == "poly":
            A = constructA(x_points)
            b = constructB(y_points)

            # 返回一个(n,)的向量
            coeff = np.linalg.solve(A,b)
            poly_equation = 0
            for i,order in enumerate(range(num_points-1,-1,-1)):
                poly_equation += coeff[i][0]*self.x**order

            # 画图
            x_poly = np.linspace(1,max(x_points),10000)
            y_poly = [self.f(poly_equation,x) for x in x_poly]
            #
            # total_error = reduced(lambda x,y:x+y, [np.abs(y_true[i] - y_poly[i]) for i in range(len(x_poly))])
            # print(total_error)

            plt.plot(x_poly,y_poly,label="poly")
            plt.legend()
            plt.show()

        # 两两连线即可
        elif mode == "linear":
            plt.plot(x_points,y_points,label="linear")

        elif mode == "lagrange":
            # 是另外一种poly interpolation的表达形式，插值结果是一样的
            lagrangeEquation = 0
            for i in range(num_points):
                multiple = 1
                for j in range(num_points):
                    pass
                lagrangeEquation += y_points[i]

        elif mode == "newton":
            memo = newtonDividedDifference()
            newtonEquation = memo[(0,0)]
            print(memo)
            for i in range(1,num_points):
                multiple = 1
                for j in range(0,i):
                    multiple*=(self.x-x_points[j])
                newtonEquation += memo[(0,i)]*multiple

            print(newtonEquation)


            x_newton = np.linspace(1,max(x_points),10000)
            y_newton = [self.f(newtonEquation,x) for x in x_newton]

            plt.plot(x_newton,y_newton,label="newton")


        plt.legend()
        plt.show()


    def printHelper(self,iteration,error,rootValue):
        print("Iteration spent:{0}\nError: {1}\nRoot found: {2}".format(iteration,error,rootValue))

    def f(self,func,value):
        return float(func.evalf(subs={self.x:value}))


if __name__ == "__main__":
    practice = NumericalMethod()

    # Lecture 1&2 Root Finding

    # BisectionMethod
    # practice.bisectionMethod()


    # NewtonMethod
    # practice.newtonMethod()
    # practice.exploreNewton()


    # SecantMethod
    # practice.secantMethod()


    # fixedPoint
    # practice.fixedPoint()



    # Lecture 3
    # Interpolation
    # graph_x = practice.generate_x()
    # func = graph_x**2
    data_point = [(1, 930.00), (2, 918.40), (3, 937.41), (4, 829.10), (5, 846.35)]
    practice.interpolation(data_points=data_point,mode="newton")


