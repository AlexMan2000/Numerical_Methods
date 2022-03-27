from sympy import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import math


if __name__ == "__main__":
    A = np.array([[-1,2],[2,-6]])
    eig,eigvec = np.linalg.eig(A)
    print(eig)
    print(eigvec)