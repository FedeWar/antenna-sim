#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <math_functions.h>

%DEFINES%

extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl sqrtf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl expf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl abs(float) __THROW;
extern __device__ __device_builtin__ void sincosf(float x, float *sptr, float *cptr) __THROW;

__device__ cuComplex operator*(float2 v, float a)
{
	return make_float2(v.x * a, v.y * a);
}

/*
__device__ cuComplex cmul(const cuComplex& c, float a)
{
	return make_cuComplex(c.x * a, c.y * a);
}

__device__ cuComplex cdot(const cuComplex& c1, const cuComplex& c2)
{
	return make_cuComplex(
		c1.x * c2.x - c1.y * c2.y,
		c1.x * c2.y + c1.y * c2.x);
}
*/

__device__ float2& operator+=(float2& a, const float2& b)
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

__device__ float3& operator+=(float3& a, const float3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

struct complex2
{
	__device__ complex2(float xr, float xi, float yr, float yi) :
		x(make_cuComplex(xr, xi)),
		y(make_cuComplex(yr, yi))
	{}

	__device__ complex2(cuComplex _x, cuComplex _y) :
		x(_x),
		y(_y)
	{}
	
	__device__ complex2 operator+(const complex2& c) const
	{
		return complex2(cuCaddf(x, c.x), cuCaddf(y, c.y));
	}

	__device__ complex2 operator*(const float a) const
	{
		return complex2(x * a, y * a);
	}

	__device__ complex2& operator*=(const float a)
	{
		x.x *= a;
		x.y *= a;
		y.x *= a;
		y.y *= a;
		return *this;
	}

	__device__ complex2 operator*(const cuComplex& c) const
	{
		return complex2(cuCmulf(x, c), cuCmulf(y, c));
	}

	__device__ complex2& operator+=(const complex2& c)
	{
		x.x += c.x.x;
		x.y += c.x.y;
		y.x += c.y.x;
		y.y += c.y.y;

		return *this;
	}

	cuComplex x;
	cuComplex y;
};

__device__ float vabs(const float3& v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float vabs(const float2& v)
{
	return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ float3 sub(const float3& v1, const float3& v2)
{
	return make_float3(
		v1.x - v2.x,
		v1.y - v2.y,
		v1.z - v2.z);
}

__device__ float3 mul(const float3& v, float a)
{
	return make_float3(v.x * a, v.y * a, v.z * a);
}

// c = k * rr1
// exp(-1j * self.k * rr1)
// c viene moltiplicato per -j
__device__ cuComplex cexp(const cuComplex& c)
{
	float a = expf(-c.y);
	float r, i;

	sincosf(-c.x, &r, &i);

	return make_cuComplex(a * r, a * i);
}

__device__ float3 add(const float3& v1, const float3& v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ complex2 Ji(float3 r, float3 r1)
{
	// FIXME in teoria si possono levare
	// 1) Perché va implementato un modo più elastico
	// 2) L'integrazione è garantita tra [-10, +10)
	// filo lungo 20
	if(abs(r1.x) < 0.001 && abs(r1.y) < 10)
	{
		return complex2(1, 0, 0, 0);
	}
	else
	{
		return complex2(0, 0, 0, 0);
	}
}

__device__ cuComplex G(const float3& r, const float3& r1)
{
	// Fa in modo che il denominatore non si annulli
	const float epsilon = 0.000001f;
	const float pi = 3.1415927410125f;
	const cuComplex k = make_cuComplex(1, 0);

	const float rr1 = vabs(sub(r, r1)) + epsilon;
	const cuComplex N = cexp(k * rr1);
	const float D = 1.0f / (4 * pi * rr1);

	return N * D;
}

/*
 * La funzione da integrare lungo r1.
 *
 * /param r		Punto in cui calcolare l'integrale.
 * /param r1	Somma degli incrementi differenziali fino a ora.
 * /return		Valore della funzione.
 */
__device__ complex2 f(const float3& r, const float3& r1)
{
	return Ji(r, r1) * G(r, r1);
}

/*
 * Esegue un passo dell'algoritmo RK4.
 * 
 * /param r			Punto in cui calcolare l'integrale.
 * /param r1		Somma degli incrementi differenziali fino a ora.
 * /param A 		Valore per A trovato nel precedente step.
 * /param dr		Incremento differenziale.
 * /define abs_dr	Lunghezza dell'incremento differenziale per r1
 * /return Il valore di A incrementato di uno step.
 */
__device__ inline void step(const float3& r, const float3& r1, complex2& A, const float3& dr)
{
	#define abs_dr 0.1f
	complex2 k1 = f(r, r1);
	k1 += f(r, add(r1, mul(dr, 0.5f))) * 2.0f;	// k2
	k1 += f(r, add(r1, dr));	// k3

	A += k1 * (abs_dr / 4.0f);
}

extern "C"
__global__ void compute(float* out_points)
{
	const int x = blockIdx.x * bwidth + threadIdx.x;
	const int y = blockIdx.y * bheight + threadIdx.y;
	const int z = blockIdx.z;// * bdepth + thread.z
	const int point = (z * HEIGHT + y) * WIDTH + x;
	const float3 r = make_float3((x + offx) * SCALE, (y + offy) * SCALE, (z + offz) * SCALE);
	const float3 dr = make_float3(0, 0.1f, 0);
	const int its = 200;

	complex2 A(0, 0, 0, 0);
	float3 r1 = make_float3(0, -10, 0);

	for(int i = 0; i < its; i++)
	{
		step(r, r1, A, dr);
		r1 += dr;
	}

	out_points[point] = vabs(A.x);
}
