# Basics
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.fft import fft
from scipy.linalg import dft
import sympy
import pandas as pd


class HW6:

    def __init__(self):
        # Compute the w for DFT
        self.w = lambda n: np.exp(-1j*2*np.pi/n)
        self.csv_path = "./EURUSD=X.csv"


    def formatInterpolation(self,a,b,c,d):
        # Utilize the DFT THM
        n = a.shape[0]
        t = sympy.symbols("t")
        if n % 2 == 0:
            coeff = []
            for k in range(1,n//2):
                coeff.extend([a[k],-b[k]])

            period_func = []

            for k in range(1,n//2):
                period_func.extend([sympy.cos(2*sympy.pi*k*(t-c)/(d-c)),sympy.sin(2*sympy.pi*k*(t-c)/(d-c))])

            middle = sum([coeff[i]* period_func[i] for i in range(len(coeff))])

            func = (1/np.sqrt(n)) * (a[0] + 2*middle + a[n//2]*sympy.cos(n*sympy.pi*(t-c)/(d-c)))
        else:
            t = sympy.symbols("t")

            coeff = []
            for k in range(1, n):
                coeff.extend([a[k], -b[k]])

            period_func = []

            for k in range(1, n):
                period_func.extend(
                    [sympy.cos(2 * sympy.pi * k * (t - c) / (d - c)), sympy.sin(2 * sympy.pi * k * (t - c) / (d - c))])

            rest = sum([coeff[i] * period_func[i] for i in range(len(coeff))])

            func = a[0] + rest

        return t,func


    def plotInterpolation(self,t_inter,x_inter,t_full,x_full,mode="normal",datetimes=None):
        print(mode)
        print(datetimes)
        fig,ax = plt.subplots(2,1)

        plt.subplots_adjust(hspace=0.5)

        ax[0].set_title("Imagine")
        ax[0].set_xlabel("t axis")
        ax[0].plot(t_inter,x_inter)
        ax[0].set_xticks(t_inter)

        ax[1].set_title("Interpolation")
        ax[1].set_xlabel("t axis")
        ax[1].plot(t_inter,x_inter,"go",label="interpolated points")
        ax[1].plot(t_full,x_full,label="interpolation function")

        if mode == "datetime":
            assert datetimes is not None
            # ax[1].set_xticks(datetimes)
            ax[1].set_xticks(np.arange(0,12))
            ax[1].set_xticklabels(datetimes,rotation=20)
        elif mode == "normal":
            ax[1].set_xticks(t_inter)

        ax[1].legend(loc=(0.7,1))

        plt.show()


    def performDFT(self,x,n,mode="self"):
        if mode == "self":
            DFT = np.zeros((n,n),dtype = np.complex)

            for i in range(n):
                for j in range(n):
                    DFT[i, j] = self.w_ft ** (i * j)

            DFT = (1 / np.sqrt(n)) * DFT
            DFT_x =  np.dot(DFT, x)

        else:
            DFT = dft(n,scale="sqrtn")
            DFT_x = DFT @ x


        return DFT_x, DFT


    def performFFT(self,x,n,mode="recursive"):
        if mode == "recursive":
            Fn = (1/np.sqrt(n)) * self.computeFn(n)
        elif mode == "scipy":
            Fn_x  = fft(x,norm="ortho")
            return Fn_x,None
        else:
            Fn = 0
        print(Fn)
        return Fn @ x, Fn


    def computeFn(self,n):
        """
        We will use the matrix form of multiplication
        :param n:
        :param x:
        :return:
        """
        if n == 1:
            return np.array([[1]])

        # 1. Get the D matrix of size n = 2^m/2
        Dn_2 = np.diag(np.power(self.w_ft,np.arange(0,n//2)))

        # 2. Get the transpose matrix to align the odd and even terms
        Pn_2 = np.zeros((n, n))
        Pn_2[np.arange(0, n // 2), np.arange(0, n, 2)] = 1
        Pn_2[np.arange(n // 2, n), np.arange(1, n, 2)] = 1

        # 3. Get the identity matrix of size n
        In_2 = np.eye(n // 2)

        # 4. Compute the Fn
        Fn_2 = self.computeFn(n//2)

        Fn = np.block([[In_2,Dn_2],[In_2,-Dn_2]]) @ np.block([[Fn_2,np.zeros((n//2,n//2))],[np.zeros((n//2,n//2)),Fn_2]]) @ Pn_2

        return Fn


    def evaluateFunction(self,func,variable,value):
        return float(func.evalf(subs={variable:value}))


    def problem1(self,periodic_points,start=0,end=1,n=8,mode="normal",datetimes=None):
        """
        Interpolate between [start,end)
        :param periodic_points:
        :param start:
        :param end:
        :return:
        """
        # 1. Compute number of points
        dim = len(periodic_points)

        # The x, Ready for DFT
        f = np.array(periodic_points)

        w = self.w(n)
        self.w_ft = w

        F, DFT = self.performDFT(f,n)

        # Calculate the real part of the vector
        a = np.real(F)

        # Calculate the imaginary part of the vector
        b = np.imag(F)

        var,interpolation_func = self.formatInterpolation(a,b,start,end)
        # print("Interpolation Function:{}".format(interpolation_func))

        t_inter = np.linspace(start,end,n+1)[:-1]
        x_inter = list(f)

        t_full = list(np.linspace(start,end,100))
        x_full = [self.evaluateFunction(interpolation_func,var,val) for val in t_full]


        if mode == "datetime":
            assert datetimes is not None

        self.plotInterpolation(t_inter,x_inter,t_full,x_full,mode,datetimes)


    def problem2(self,start = -2,end=2,N=1024):
        ts = (end-start) / N  # step
        sr = 1/ts  # sampling rate
        T0 = 0.1984
        f = lambda t: np.exp(-np.abs(t) / T0)
        dt = 4 / N
        t = np.arange(-2, 2, dt)
        x = f(t)


        # Using DFT
        # w = self.w(N)
        # self.w_ft = w
        # X,DFT = self.performDFT(x,N)
        # X = 1/(np.sqrt(N))*X

        n = np.arange(N)
        k = n.reshape((N, 1))

        # DFT Transformation
        e = np.exp(-2j * np.pi * k * n / N)
        # DFT Transformed
        X = np.dot(e, x)

        # plt
        frq = k / 4  # two sides frequency range
        X = np.abs(X) / sr
        # get w1,f1
        w1 = np.vstack((-frq[N // 2 - 1::-1], frq[:N // 2])) * np.pi * 2
        f1 = np.hstack((X[N // 2 - 1::-1], X[:N // 2]))


        # Compute FFT
        xf = np.fft.fft(x) / N
        # xf = np.fft.fft(xf) / N  # fft computing and normalization
        w2 = w1
        f2 = abs(xf)

        # plot the result in frequency domain
        fig,ax = plt.subplots(2,1)
        ax[1].plot(w1, f1, 'o-.', label='DFT', color='red')
        ax[1].plot(w2, f2, 'o--', label='FFT', color='blue', alpha=0.5)
        ww = np.linspace(-20, 20, 10000)
        ax[0].plot(ww, 2 * T0 / (1 + (ww * T0) ** 2), label='$f(\omega)$', color='black')
        plt.legend()
        plt.xlabel('$\omega$')
        plt.xlim([-20, 20])
        plt.ylabel('$f(\omega)$')
        plt.show()


    def problem3(self):
        data = pd.read_csv(self.csv_path)
        data["Date"] = pd.to_datetime(data.Date)
        data.set_index("Date",inplace=True)

        # The last data entry every month(Adjusted Close Price)
        sampled_data = data[["Adj Close"]].resample("M").last()[:-1]


        # Plot the dataset
        # plt.plot(sampled_data)
        # plt.xticks(sampled_data.index,rotation=45)
        # plt.show()


        # Perform trigonometric interpolation function
        periodic_points = sampled_data["Adj Close"].to_numpy()
        n = len(periodic_points)


        # datetime string
        datetime_strings = pd.Series(pd.to_datetime(sampled_data.index.values))
        datetime_strings = list(datetime_strings.apply(lambda x: x.strftime("%Y-%m-%d")).to_numpy())

        self.problem1(periodic_points,0,12,n,mode="datetime",datetimes=datetime_strings)


    def speedTest(self):
        # In Jupyter lab
        pass


if __name__ == "__main__":
    hw6 = HW6()


    # Time series data
    # data_points = [1,-1]*4
    # hw6.problem1(data_points)

    hw6.problem2(N=1024)

    # hw6.problem3()














