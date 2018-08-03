# Project
from gpu import compute
import solve_w_rk4
# Matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Numpy
import numpy as np
import numpy.linalg as la
# Skimage
from skimage import measure
# Std
import sys
import cmath as cm

fig = plt.figure()
ax = fig.gca(projection='3d')
np.set_printoptions(threshold=np.inf)

def main(argv):
	use_gpu = False
	iso_surf = (0.04,)
	volume = None

	use_gpu = '--use-gpu' in argv
	# Prevents errors or warnings when using the default settings
	if use_gpu:
		volume = np.array([32, 32, 32])
	else:
		volume = np.array([32, 32, 1])

	if '--volume' in argv:
		arg_idx = argv.index('--volume')
		volume = np.array(eval('(' + argv[arg_idx + 1] + ')'))
	if not use_gpu and volume[2] > 1:
		print("The CPU solver does not support 3D computations. The Z dimension is ignored.")

	if '--iso-surf' in argv and not use_gpu:
		arg_idx = argv.index('--iso-surf')
		iso_surf = np.array(eval('(' + argv[arg_idx + 1] + ')'))
	elif use_gpu:
		print("The isosurface computation is not supported when using the CPU solver.")

	offset = volume / 2

	if use_gpu:
		Z = compute(volume, offset)
		Zcube = Z.reshape(volume)
		for s in iso_surf:
			isoplot(Zcube, s, offset)
		plt.show()

	else:
		Xl = np.arange(-offset[0], offset[0])
		Yl = np.arange(-offset[1], offset[1])
		X, Y = np.meshgrid(Xl, Yl)
		Z = solve_w_rk4.solve(volume[0], Xl, Yl, offset[0], offset[1])
		surf = ax.plot_surface(X, Y, Z, cmap='hot', linewidth=0, antialiased=True)
		plt.show()

def isoplot(volume, val, offset):
	verts, faces, normals, values = measure.marching_cubes_lewiner(volume, level=val, step_size=5)
	verts = verts - offset
	mesh = Poly3DCollection(verts[faces])
	mesh.set_edgecolor('k')
	ax.add_collection3d(mesh)

if __name__ == '__main__':
	main(sys.argv)
