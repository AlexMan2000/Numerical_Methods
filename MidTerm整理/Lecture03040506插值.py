import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy import interpolate,misc
import math

class InterpolateError:
    # Reference: https: // zhuanlan.zhihu.com / p / 136700122

    def __init__(self):
        pass


    # 一维插值
    def interpolate1d(self):
        x = np.linspace(0, 10, 11)
        y = np.sin(x)
        # y = np.sin(x) + np.random.normal(loc=0, scale=1, size=len(x)) * 0.2

        xnew = np.linspace(0, 10, 101)

        plt.plot(x, y, 'ro')
        list1 = ['linear', 'nearest','quadratic','cubic']
        for kind in list1:
            f = interpolate.interp1d(x, y, kind=kind)
            # f是一个函数，用这个函数就可以找插值点的函数值了：
            ynew = f(xnew)
            plt.plot(xnew, ynew, label=kind)

        plt.legend(loc='lower right')
        plt.show()


    # 画出王error plot
    def interpolationError(self,x_inter,n,true_func):
        # x_inter是插值点
        # true_func 是真实的函数
        # P(x)是插值得出的函数

        # Scipy 求导 https: // docs.scipy.org / doc / scipy / reference / generated / scipy.misc.derivative.html
        # Sympy 求导 https://blog.csdn.net/weixin_42646103/article/details/107850332
        a = x_inter[0]
        b = x_inter[1]



        # 画原函数图, 需要所有点
        x_origin = np.linspace(a,b,100)
        y_origin = true_func(x_origin)



        x_inter = np.linspace(a,b,n)
        x_cheb = np.array(self.chebyshevInterpolation(a,b,n))

        fig,ax=plt.subplots(3,1)

        ax[0].plot(x_origin,y_origin,label="original_function")
        ax[0].set_title("Original Function")



        # 算出插值的y值，需要插值点
        y_inter = true_func(x_inter)
        y_cheb = true_func(x_cheb)


        # 插值函数形状,需要所有点
        kernel_list = ["linear","cubic"]
        x_points = x_origin
        x_points_cheb = np.linspace(x_cheb[0],x_cheb[1],100)

        # 对每种不同的差值方式遍历
        for kernel in kernel_list:
            # 使用scipy求解出插值函数, P(x)
            # https://zhuanlan.zhihu.com/p/136700122

            interfunc_uniform = interpolate.interp1d(x_inter,y_inter,kernel)
            interfunc_cheb = interpolate.interp1d(x_cheb,y_cheb,kernel)


            y_points_uniform = interfunc_uniform(x_points)
            y_points_cheb = interfunc_cheb(x_points_cheb)

            # Defining functions, 计算插值函数
            multiple = 1/(math.factorial(len(x_inter)))



            # 求error, 使用sympy求解error
            x = symbols("x")
            errorfunc_uniform = 1

            for i in range(len(x_inter)):
                errorfunc_uniform*=(x-x_inter[i])
            errorfunc_uniform*=multiple
            # 暂时取中间的点
            derivative_multiple = misc.derivative(true_func, x0=x_inter[len(x_inter) // 2], dx=1e-5, n=2)
            errorfunc_uniform *= derivative_multiple



            # Cheb
            x_cheb_sym = symbols("x1")
            error_func_cheb = 1
            for i in range(len(x_cheb)):
                error_func_cheb *= (x_cheb_sym - x_cheb[i])
                error_func_cheb *= multiple

            # 暂时取中间的点
            derivative_multiple_cheb = misc.derivative(true_func, x0=x_cheb[len(x_cheb) // 2], dx=1e-5, n=2)
            error_func_cheb *= derivative_multiple_cheb


            error_plot_uniform = []
            error_plot_cheb = []

            # 使用scipy求导, 要注意这里的dx不能取的太小，否则会出现溢出
            for error_x in x_points:
                # Error 表达式
                # print(errorfunc)

                # 打印Error的值
                error_plot_uniform.append(float(errorfunc_uniform.evalf(subs={x:error_x})))

            # cheb
            for error_x in x_points_cheb:
                error_plot_cheb.append(float(error_func_cheb.evalf(subs={x_cheb_sym:error_x})))


            # 画出error plot
            ax[1].set_title("Error Plot")
            ax[2].set_title("Cheb")
            ax[1].plot(x_points,error_plot_uniform,label=kernel)
            ax[2].plot(x_points,error_plot_cheb,label=kernel+"cheb")
            ax[1].legend()
            ax[2].legend()


            # 画插值函数图
            ax[0].plot(x_points,y_points_uniform,label=kernel)


        ax[0].plot(x_inter,y_inter,'ro',label="interpolation point")
        ax[0].legend()

        # 调整子图间距
        # https://blog.csdn.net/weixin_46192930/article/details/106979206
        plt.subplots_adjust(wspace=1,hspace=1)
        plt.show()


    # 这个使用的是非均匀的插值算法, 求出非均匀的插值点
    def chebyshevInterpolation(self,a,b,n):
        x_list = [(b+a)/2+((b-a)/2)*np.cos((2*i-1)*np.pi/(2*n)) for i in range(1,n+1)]
        x_list.reverse()
        return x_list


    def spline(self,datapoints):
        # 导包
        x = np.array(list(map(lambda x: x[0],datapoints)))
        y = np.array(list(map(lambda x: x[1],datapoints)))
        # s = 0 是完全拟合，s = 1是部分拟合
        temp = interpolate.splrep(x,y,s=0)

        # To draw all the points
        xnew = np.arange(0, np.pi ** 2, np.pi / 100)
        ynew = interpolate.splev(xnew, temp, der=0)

        plt.figure()

        plt.plot(x,y,"*",color="red")

        # linear
        plt.plot(x, y, 'b',color="red")

        # Cubic
        plt.plot(xnew, ynew,color="blue")


        plt.plot(xnew, np.cos(xnew),color="yellow")

        plt.legend(['interpolation','Linear', 'Cubic Spline', 'True'])
        # plt.axis设置横轴的始末和纵轴的始末
        plt.axis([-0.1, 6.5, -1.1, 1.1])
        plt.title('Cubic-spline Interpolation in Python')
        plt.show()




if __name__ == "__main__":
    interpolateError = InterpolateError()
    # interpolateError.interpolate1d()

    a = -1
    b = 1
    n = 10
    range_ = (a,b)
    f = lambda x: np.sin(x)
    interpolateError.interpolationError(range_,n,f)


    # print(interpolateError.chebyshevInterpolation(a,b,n))

    # x = np.arange(0,10)
    # y = np.cos(x**3)
    # datapoints = [*zip(x,y)]
    # interpolateError.spline(datapoints)