# PyCUDA
import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.driver as drv
from pycuda.compiler import SourceModule
# Numpy
from matplotlib import pyplot as plt
import numpy.linalg as la
import numpy as np
# Std
import os

def compute(volume, offset):
	bsize = (32, 32, 1)
	gsize = (int(volume[0] / bsize[0]), int(volume[1] / bsize[1]), int(volume[2]))

	DEFINES = '\n#define SCALE ' + str(1) + \
		'\n#define WIDTH ' + str(volume[0]) + \
		'\n#define HEIGHT ' + str(volume[1]) + \
		'\n#define bwidth ' + str(bsize[0]) + \
		'\n#define bheight ' + str(bsize[1]) + \
		'\n#define offx ' + str(-offset[0]) + \
		'\n#define offy ' + str(-offset[1]) + \
		'\n#define offz ' + str(-offset[2]) + \
		'\n#define DEPTH ' + str(volume[2]) + \
		'\n#define bdepth ' + str(1) + '\n' # inutile
	
	# Non optimal method
	path = os.path.split(__file__)
	cufile = open(path[0] + '/kernel.cu', 'r')
	code = cufile.read()
	code = code.replace('%DEFINES%', DEFINES)
	mod = SourceModule(code, "nvcc",
		include_dirs=["/usr/local/cuda/include"],
		no_extern_c=True)
	compute = mod.get_function("compute")
	
	# Array di uscita
	dest = np.zeros(volume[0] * volume[1] * volume[2]).astype(np.float32)

	# Il numero di elementi, infinitamente piccoli,
	# che compongono l'antenna
	pieces = 100

	# Lista dei vettori di corrente impressa che fluisce nell'antenna,
	# 2 = la corrente è difinita da due vettori: posizione e valore
	# 4 = ogni vettore è composto da 4 elementi, usiamo float4 per
	# migliore allineamento in memoria.
	# TODO NON USATO
	#geometry = np.zeros(2 * pieces * 4).astype(np.float32)
	
	compute(drv.Out(dest), block=bsize, grid=gsize)
	context.synchronize()

	return dest
