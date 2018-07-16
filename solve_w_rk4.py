# Project
from line import line
from rk4 import rk4
# Matplotlib
from matplotlib import pyplot as plt
# Numpy
import numpy as np

def solve(l, Xl, Yl, px, py):
	Z = np.zeros([l, l])
	#phase = np.zeros([l, l])
	ln = line()
	line_points = ln.points()
	
	XX = [line_points[i][0][0] for i in range(0, len(line_points))]
	YY = [line_points[i][0][1] for i in range(0, len(line_points))]
	LL = [line_points[i][1] for i in range(0, len(line_points))]

	# Plot dell'antenna
	plt.plot(YY, XX)

	intg = rk4(1 + 0*1j, line_points)
	for x in Xl:
		for y in Yl:

			intg.compute(x / 2, y / 2)

			Ax = intg.A_old[0]
			Ay = intg.A_old[1]	# NULLO

			Z[int(x + px)][int(y + py)] = abs(Ax)#cm.phase(Ax))
	
	return Z
