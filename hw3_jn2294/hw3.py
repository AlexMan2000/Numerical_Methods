import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg,integrate
from sympy import *
from scipy.linalg import lu, lu_factor, lu_solve, det



import time
import seaborn as sns
sns.set()

class HW3:
    def __init__(self):
        pass


    # Problem 3.1
    def heartEquation(self,option,nsim=0):
        """
        Use Monte Carlo to compute the area
        :param option:
        :param nsim:
        :return:
        """
        if option == "monte":
            assert nsim != 0
            xlim = [-1.5, 1.5]
            ylim = [-1.5, 2.5]

            xrange = xlim[1] - xlim[0]
            yrange = ylim[1] - ylim[0]

            # Generate random points within range
            x = np.random.random(nsim) * xrange + xlim[0]
            y = np.random.random(nsim) * yrange + ylim[0]

            def heartboundary(x,y):
                return x**2 + (y-np.sqrt(np.abs(x)))**2 - 2

            def circleboundary(x,y):
                return x**2 + (y-(2*y*np.sqrt(np.abs(x)))/(2*y - 2*np.sqrt(2)))**2 - (np.sqrt(2)-(2*y*np.sqrt(np.abs(x)))/(2*y - 2*np.sqrt(2)))**2

            # Utilize a table to record the points coordinates
            inside_coordinates = np.zeros((nsim,2))
            inside_points = 0
            for i in range(nsim):
                in_out = heartboundary(x[i],y[i])
                if in_out <= 0 :
                    inside_points += 1
                    inside_coordinates[inside_points-1,:] = np.array([x[i],y[i]])

            # Circle Simulation
            # inside_circles = np.zeros((nsim,2))
            # inside_circle_points = 0
            # for i in range(nsim):
            #     in_out = circleboundary(x[i], y[i])
            #     if in_out <= 0:
            #         inside_circle_points += 1
            #         inside_circles[inside_circle_points - 1, :] = np.array([x[i], y[i]])


            fig = plt.figure(figsize=(4.5, 4), dpi=200)
            ax = plt.axes()

            ax.scatter(inside_coordinates[0:inside_points, 0]
                       , inside_coordinates[0:inside_points, 1]
                       , color="red"
                       , s=0.5
                       )
            # ax.scatter(inside_circles[0:inside_circle_points,0]
            #            , inside_circles[0:inside_circle_points,1]
            #            , color = "blue"
            #            , s= 0.5)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.tight_layout()
            plt.savefig("./program_imgs/heart.png")
            plt.show()

            print("area = {}".format(xrange * yrange * inside_points / nsim))

        elif option == "newton":
            pass
        elif option == "GQ":
            pass


    # Problem 3.2
    def integration(self,option,loop=False,nsim=1000):
        """
        :param option:
        :param nsim: The number of point to simulate
        :return:
        """
        f = lambda x: np.sin(x) / x
        correct = scipy.integrate.quadrature(f,0,10)
        # Guassian Quadrature
        if option == "GQ":
            f = lambda x: np.sin(x)/x
            # I have modified the source code to make it output the order of the Guassian Quadrature
            result = scipy.integrate.quadrature(f,0,10) # tol=1.49e-8
            print("The numerical solution is {}\n".format(result[0]),"The err is {}\n".format(result[1]),"The first n resulting in err less than tol is {}".format(result[2]))
            return result

        # Monte Carlo
        elif option == "monte":
            assert nsim!=0,"Must specify how many simulation for monte carlo"
            xlim = [0,10]
            ylim = [-2/(3*np.pi),1]  # lim sinx /x =1

            xrange = xlim[1] - xlim[0]
            yrange = ylim[1] - ylim[0]


            x = np.random.random(nsim)*xrange + xlim[0]
            y = np.random.random(nsim)*yrange + ylim[0]


            def function_boudary(x):
                return np.sin(x)/x

            points = np.zeros((nsim,3))
            n_points = 0
            pos_points = 0
            neg_points = 0
            other_points = 0
            for i in range(nsim):
                n_points += 1
                # Negative points
                if function_boudary(x[i]) <= y[i] <= 0:
                    points[n_points-1,:] = np.array([x[i],y[i],-1])
                    neg_points += 1
                # Positive points
                elif 0 <= y[i] <= function_boudary(x[i]):
                    points[n_points-1,:] = np.array([x[i],y[i],1])
                    pos_points += 1
                # Other points
                else:
                    points[n_points-1,:] = np.array([x[i],y[i],0])
                    other_points += 1

            if not loop:
                fig = plt.figure(figsize=(4.5, 4), dpi=200)
                ax = plt.axes()

                # Draw the positive area points
                ax.scatter(points[np.where(points[:,2]==1)][:,0]
                           , points[np.where(points[:,2]==1)][:,1]
                           , color="blue"
                           , s=0.5
                           )


                # Draw the negative area points
                ax.scatter(points[np.where(points[:, 2]==-1) ][:,0]
                           ,points[np.where(points[:, 2]==-1) ][:,1]
                           , color="red"
                           , s=0.5
                           )


                # Draw other area points
                ax.scatter(points[np.where(points[:, 2]==0)][:,0]
                           , points[np.where(points[:, 2]==0)][:,1]
                           , color="yellow"
                           , s=0.5
                           )

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                plt.tight_layout()
                plt.savefig("./program_imgs/integration.png")
                plt.show()

                # print("The numerical solution is: {}\n".format(xrange * yrange * (pos_points-neg_points) / nsim),
                #       "The err is {}".format(np.abs(xrange * yrange * (pos_points-neg_points) / nsim - correct[0])))

            return xrange * yrange * (pos_points-neg_points) / nsim


    # Problem 3.3.1
    def randomGeneration(self,option,n=2**10):
        matrix = np.zeros((n,n))
        if option == "uniform":
            matrix = np.random.uniform(-1,1,n*n).reshape(n,n)
        elif option == "normal":
            matrix = np.random.normal(0,1,n*n).reshape(n,n)
        return matrix

    # Problem 3.3.2 - 3.3.4
    def matrix_multiplication(self,m1,m2,option):
        """
        Perform matrix multiplication, providing two options, naive and Strassen Algorithm
        :param m1:
        :param m2:
        :return:
        """
        self.num_multiplications = 0
        self.num_additions = 0

        # Helper functions for matrix properties check and recursions
        def dim_check(m1,m2)->int:
            dim1 = m1.shape[1]
            dim2 = m2.shape[0]
            return  dim1 == dim2

        def judge_square(matrix)->int:
            return matrix.shape[0] == matrix.shape[1]

        def judge_order(matrix)->int:
            """
            Judge whether the matrix is of order of power of 2
            :param matrix: Given that the matrix is a square matrix
            :return:
            """
            n = matrix.shape[0]

            return 1 if (n & (n-1)==0) else 0

        def segment_matrix(matrix)->tuple:
            n = matrix.shape[0]
            s11 = matrix[:n//2,:n//2]
            s12 = matrix[:n//2,n//2:]
            s21 = matrix[n//2:,:n//2]
            s22 = matrix[n//2:,n//2:]
            return s11,s12,s21,s22

        def strass_helper(A,B)->np.ndarray:
            assert A.shape == B.shape

            # Terminating condition, the matrix of 1 x 1
            if A.shape[0] == 1:
                result = A.dot(B)
                self.num_multiplications += 1

                return np.array([[result]])

            n = A.shape[0]
            result_matrix = np.zeros(A.shape)

            a11,a12,a21,a22 = segment_matrix(A)
            b11,b12,b21,b22 = segment_matrix(B)

            m1 = strass_helper(a11+a22,b11+b22)
            m2 = strass_helper(a21+a22,b11)
            m3 = strass_helper(a11,b12-b22)
            m4 = strass_helper(a22,b21-b11)
            m5 = strass_helper(a11+a12,b22)
            m6 = strass_helper(a21-a11,b11+b12)
            m7 = strass_helper(a12-a22,b21+b22)

            self.num_additions += a11.shape[0]**2*10

            c11 = m1 + m4 - m5 + m7
            c12 = m3 + m5
            c21 = m2 + m4
            c22 = m1 - m2 + m3 + m6

            self.num_additions += a11.shape[0]**2*8

            result_matrix[:n//2,:n//2] = c11
            result_matrix[:n//2,n//2:] = c12
            result_matrix[n//2:,:n//2] = c21
            result_matrix[n//2:,n//2:] = c22

            return result_matrix


        # main procedure
        # Relies on no thirty party package
        if option == "naive":
            assert dim_check(m1,m2)==True, "Dimension mismatch, cannot perform multiplication operation"

            # Shape of output matrix
            row = m1.shape[0]
            column = m2.shape[1]
            result_matrix = np.zeros((row,column))
            dim = m1.shape[0]

            start = time.time()
            for i in range(row):
                for j in range(column):
                    for k in range(dim):
                        # Vanilla version
                        result_matrix[i][j] += m1[i,k]*m2[k,j]

                        # Vecotrized Version
                        # result_matrix[i][j] = m1[i].dot(m2[j])

                        self.num_multiplications += dim
                        self.num_additions += (dim - 1)

            print("Total number of multiplications:{}\n".format(self.num_multiplications))
            print("Total number of additions:{}\n".format(self.num_additions))
            print("Total time spent:{}\n".format(time.time()-start))

            return result_matrix

        elif option == "Strassen":
            assert dim_check(m1,m2) ==True, "Dimension mismatch, cannot perform multiplication operation"
            assert judge_square(m1) == True, "Must be square matrix"
            assert judge_square(m2) == True, "Must be square matrix"
            assert judge_order(m1) == True, "Must be order of power of two"
            assert judge_order(m2) == True, "Must be order of power of two"

            start = time.time()
            result_matrix = strass_helper(m1,m2)

            print("Total number of multiplications:{}\n".format(self.num_multiplications))
            print("Total number of additions:{}\n".format(self.num_additions))
            print("Total time spent:{}\n".format(time.time()-start))

            return result_matrix

    def official(self,m1,m2):

        import numpy as np

        def split(matrix):
            """
            Splits a given matrix into quarters.
            Input: nxn matrix
            Output: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d
            """
            row, col = matrix.shape
            row2, col2 = row // 2, col // 2
            return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

        def strassen(x, y):
            """
            Computes matrix product by divide and conquer approach, recursively.
            Input: nxn matrices x and y
            Output: nxn matrix, product of x and y
            """

            # Base case when size of matrices is 1x1
            if len(x) == 1:
                return x * y

            # Splitting the matrices into quadrants. This will be done recursively
            # until the base case is reached.
            a, b, c, d = split(x)
            e, f, g, h = split(y)

            # Computing the 7 products, recursively (p1, p2...p7)
            p1 = strassen(a, f - h)
            p2 = strassen(a + b, h)
            p3 = strassen(c + d, e)
            p4 = strassen(d, g - e)
            p5 = strassen(a + d, e + h)
            p6 = strassen(b - d, g + h)
            p7 = strassen(a - c, e + f)

            # Computing the values of the 4 quadrants of the final matrix c
            c11 = p5 + p4 - p2 + p6
            c12 = p1 + p2
            c21 = p3 + p4
            c22 = p1 + p5 - p3 - p7

            # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
            c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

            return c

        start = time.time()
        result_matrix = strassen(m1,m2)
        print(result_matrix)
        print("Total time spent:{}\n".format(time.time()-start))

    # Problem 3.4
    def solveLinear(self, A, b,option):
        if option == "official":
            x = linalg.solve(A, b)
        elif option == "Cramer":
            assert A.shape[0] ==A.shape[1], "Need square coefficient matrix for linear equation system"
            n = A.shape[0]
            x = np.zeros(n)
            for i in range(A.shape[0]):
                acopy = A.copy()
                acopy[:,i] = b
                x[i] = det(acopy)/det(A)
        elif option == "LU":
            res = lu(A)  # P L U
            LU,p = lu_factor(A)
            x = lu_solve((LU,p),b)
        elif option == "G":
            n = A.shape[0]
            b = b.reshape((n,1))
            a = np.hstack((A,b))  # augmented matrix, self-written
            x = np.zeros(n)
            for i in range(n):
                if a[i][i] == 0.0:
                    print('Divide by zero detected!')
                # Compare the ratio of adjacent two rows
                for j in range(i + 1, n):
                    ratio = a[j][i] / a[i][i]

                    for k in range(n + 1):
                        a[j][k] = a[j][k] - ratio * a[i][k]

            x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]

            for i in range(n - 2, -1, -1):
                x[i] = a[i][n]

                for j in range(i + 1, n):
                    x[i] = x[i] - a[i][j] * x[j]

                x[i] = x[i] / a[i][i]
        else:
            x = 0
        return x


    def test_random_matrix_multiplication(self, n=2 ** 10):
        A = self.randomGeneration(option="uniform",n=n)
        B = self.randomGeneration(option="uniform",n=n)

        # print(self.official(A,B))
        # print(self.matrix_multiplication(A, B, "naive"))
        print("######################################################")
        print(self.matrix_multiplication(A, B, "Strassen"))

    def test_linear_equation(self, n):
        A = self.randomGeneration(option="normal", n=n)
        b = np.ones(n)
        print("Generated Matrix:", A)
        print("Generated Vector:", b)
        print("Start solving")
        print("###############################")

        x = self.solveLinear(A,b,"official")
        print("Official",x)
        x = self.solveLinear(A,b,"LU")
        print("LU",x)
        x = self.solveLinear(A,b,"Cramer")
        print("Cramer",x)
        x = self.solveLinear(A,b,"G")
        print("Guassian",x)

    def speed_test_linearalg(self,n,iteration=200):
        A = self.randomGeneration(option="normal", n=n)
        b = np.ones(n)
        print("Generated Matrix:", A)
        print("Generated Vector:", b)
        print("Start speed testing")
        print("###############################")

        time_map = {"official":0,"LU":0,"Cramer":0,"G":0}


        start = time.time()
        times = []
        for i in range(iteration):
            self.solveLinear(A,b,"official")
            times.append(time.time()-start)

        time_map["official"] = np.mean(times)

        start = time.time()
        times = []
        for i in range(iteration):
            self.solveLinear(A, b, "LU")
            times.append(time.time() - start)

        time_map["LU"] = np.mean(times)

        start = time.time()
        times = []
        for i in range(iteration):
            self.solveLinear(A, b, "Cramer")
            times.append(time.time() - start)

        time_map["Cramer"] = np.mean(times)


        start = time.time()
        times = []
        for i in range(iteration):
            self.solveLinear(A, b, "G")
            times.append(time.time() - start)

        time_map["G"] = np.mean(times)


        return time_map


if __name__ == "__main__":
    hw3 = HW3()

    # hw3.heartEquation("monte",nsim=100000)
    # result_list = []
    # for i in range(100):
    #     result_list.append(hw3.integration("monte",loop=True,nsim=20000))
    #
    # print(np.mean(result_list))

    # t = hw3.integration("GQ")
    # print(np.abs(t[0]-1.6579773814530576))
    # hw3.test_random_matrix_multiplication(2**8)

    # hw3.test_linear_equation(10)

    # time_map = hw3.speed_test_linearalg(100)
    # print(time_map)
    # hw3.test_linear_equation(100)


