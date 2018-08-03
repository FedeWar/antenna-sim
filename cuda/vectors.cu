
__device__ cuComplex operator*(float2 v, float a)
{
	return make_float2(v.x * a, v.y * a);
}

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
