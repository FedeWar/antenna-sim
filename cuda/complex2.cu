
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
