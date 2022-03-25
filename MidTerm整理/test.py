from sympy import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import math


if __name__ == "__main__":
    x = symbols('x')
    # f = cos(x)-x**3
    # f2 = real_root((cos(x)-x),3)
    #
    # t = 0
    # iteration = 0
    # max_iter = 100
    # while iteration < max_iter:
    #     print(t)
    #     iteration += 1
    #     if (np.abs(t - 0.65302)) < 0.5e-4:
    #         print(t)
    #     t = f2.evalf(subs={x:t})

    # print(root(32,5))

    # t = 0
    # iteration = 0
    # while iteration < 100:
    #     print(t)
    #     iteration += 1
    #     t = f2.evalf(subs={x:t})

    import time

    start = time.time()
    time.sleep(2)


    print(time.time()-start)