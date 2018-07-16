from math import pi
from math import sin
from math import cos
import numpy as np
import numpy.linalg as la

'''
Linea parametrica
'''
class line:

	'''
	Restituisce il punto in cui passa la linea
	in funzione del parametro t
	
	/param t Parametro che definisce la funzione
	/return Punto intersecato dalla linea
	'''
	def r(self, t):
		return np.array([0, t - 10])

	'''
	Il numero ottimale di punti con cui dividere
	la funzione.

	/return Quanti punti usare per chiamare r.
	'''
	def count(self):
		return 200

	def step(self):
		return self.max() / self.count()

	'''
	Il valore massimo assumibile da t, ricordo che
	0 <= t <= max.

	/return Il valore massimo di t.
	'''
	def max(self):
		return 20

	def points(self):
		p0 = p1 = self.r(0)

		points = [[np.zeros(2), 0, np.zeros(2)]] * self.count()
		points[0] = [p1, -1, np.array([1, 0])]

		for i in range(1, self.count()):
			p1 = self.r(i * self.step())
			points[i - 1][1] = la.norm(p1 - p0)
			points[i] = [p1, -1, np.array([1, 0])]

			i += 1
			p0 = p1

		return points
