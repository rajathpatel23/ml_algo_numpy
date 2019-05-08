#gradient descent algorithm
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import pandas as pd
import math


def cost_function(z,a,b):
	A = (((a-z[0])**2) + (b*(z[1]-(z[0]**2)**2)))
	return(A)

def gradient_1(z,a,b):
	d_f_1 = (2*z[0] - 2*a + (4*b*((z[0]**3) - z[1]*z[0])))
	return(d_f_1)

def gradient_2(z,a,b):
	d_f_2 = (2*b*(z[1]-(z[0]**2)))
	return(d_f_2)

def descent(d_F,z,alpha):
	z = z - alpha*d_F
	return(z)

def main():
	Num_iteration =100
	a = 25
	b = 100
	z = np.zeros((2,1))
	gradient_list_1 = []
	gradient_list_2 = []
	P_1 = []
	P_2 = []
	z = np.random.normal(0.5,0.9,(2,1))
	print(z)
	print(z.shape)
	F = np.zeros((Num_iteration, 1))
	E = ((1*(10**-8)))
	NT = 5
	precision = 0.0000100e+06 
	alpha = np.zeros((2,1))
	for i in range(Num_iteration):
		P_1.append(z.item(0))
		P_2.append(z.item(1))
		F[i] = cost_function(z,a,b)
		if abs(float(F[i] - F[i-1]))<= float(precision):
			#print("The convergence condition at cost function value"+ str(F[i]) + "and parameter values" + str(z[0])+','+ str(z[1])+ "at iteration "+ str(i))
			print(F[:i])
			plt.plot(F[:i])
			plt.ylabel('Cost_function')
			plt.xlabel('Number of Iteration')
			plt.title("Convergence and Cost function variation graph")
			print("The convergence condition at cost function value"+ str(F[i]) + "and parameter values" + str(z[0])+','+ str(z[1])+ "at iteration "+ str(i))
			plt.show()
			exit()
		d_f = np.zeros((2,1))
		d_f[0] = gradient_1(z, a, b)
		d_f[1]= gradient_2(z, a, b)
		gradient_list_1.append((d_f.item(0)**2))
		gradient_list_2.append((d_f.item(1)**2))
		G1 = sum(gradient_list_1)
		G2 = sum(gradient_list_2)
		alpha[0] = NT/(math.sqrt(E+G1))
		alpha[1] = NT/(math.sqrt(E+G2))
		z = descent(d_f,z,alpha)
	print(F)
	#print(P_1)
	#print(P_2)
	P_1 = np.array((P_1))
	P_2 = np.array((P_2))
	v1 = P_1.transpose()
	v2 = P_2.transpose()
	plt.plot(F)
	plt.xlabel('Number of iteration')
	plt.ylabel('Cost Function')
	plt.title("Convergence and Cost function variation graph")
	plt.show()


if __name__ == '__main__':
	main()



