import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import cmath as cm
from math import pi

# TODO implmentare il filo come una lista
# di punti da usare per l'integrale di linea
class rk4:

	A_old = [0+0*1j, 0+0*1j]
	r = np.zeros(2)
	k = 0 + 0*1j
	line_points = []

	def __init__(self, k, points):
		self.k = k
		self.line_points = points

	# y = A(r), non dipende da esso
	# r1 punto da evaluare, è uno scalare: rappresenta
	# la coordinata y di integrazione
	def f(self, _r1):
		r1 = self.line_points[_r1]
		return self.Ji(r1) * self.G(self.r, r1[0])

	def Ji(self, r1):
		return r1[2]
		#if abs(r1) < 10:	# filo lungo 20
		#	return np.array([1 + 0*1j, 0 + 0*1j])
		#else:
		#	return np.array([0 + 0*1j, 0 + 0*1j])

	def G(self, r, r1):
		vr1 = r1#np.array([0, r1]) # <<<
		rr1 = la.norm(r - vr1) + 0.00001
		N = cm.exp(-1j * self.k * rr1)
		D = 4 * pi * rr1

		return N / D

	def step(self, i):
		yn = self.A_old

		k1 = self.f(i)
		k2 = self.f(i + 1)#h/2)
		k4 = self.f(i + 2)#h)

		h = self.line_points[i][1] + self.line_points[i+1][1]
		yn1 = yn + (h / 4) * (k1 + 2*k2 + k4)

		self.A_old = yn1

	def compute(self, x, y):

		self.A_old = [0+0*1j, 0+0*1j]
		its = int(len(self.line_points) / 2) - 1
		#print(its)

		self.r = np.array([x, y])
		for i in range(0, its):
			self.step(i * 2)

		if(self.A_old[0] > 10):
			self.A_old[0] = 0+0*1j

	def plot(self):
		fig = plt.figure()
		plt.plot(self.t, self.y)
		plt.show()
		