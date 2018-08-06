#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <math_functions.h>

//%DEFINES%

#define ay -1
#define by 1

#define ax 1
#define bx 2

#define ny 20
#define nx 20

#define hy (1.0f*by-ay)/ny
#define hx (1.0f*bx-ax)/nx

extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl sqrtf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl expf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl abs(float) __THROW;
extern __device__ __device_builtin__ void sincosf(float x, float *sptr, float *cptr) __THROW;

__device__ float faa(float x, float y)
{
	return x*x*y + x*y*y;
}

extern "C"
__global__ void compute(float* out_points)
{
	float s = 0;

	#pragma unroll 1
	for(int i = 0; i <= ny; i++)
	{
		float p;

		if(i == 0 || i == ny)
			p = 1;
		else if(i & 1)
			p = 4;
		else
			p = 2;

		#pragma unroll 1
		for(int j = 0; j <= nx; j++)
		{
			float q;

			if(j == 0 || j == nx)
				q = 1;
			else if(j & 1)
				q = 4;
			else
				q = 2;
			float x = ax + j*hx;
			float y = ay + i*hy;
			s += p*q * faa(x,y);
		}
	}
	float I = hx * hy / 9.0f * s;
	out_points[0] = I;
}
