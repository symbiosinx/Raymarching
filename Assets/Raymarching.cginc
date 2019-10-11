float4x4 _FrustrumCorners;
float4x4 _CameraInvViewMatrix;
float _MaxSteps;
float _MaxDist;
float _StepFactor;
float _ContactThreshold;
float4x4 _Positions;
float4x4 _Rotations;
float4x4 _Scales;
float _FractalScale;
float _FractalRotationX;
float _FractalRotationY;
float _FractalRotationZ;

struct appdata {
	float2 uv : TEXCOORD0;
	float4 vertex : POSITION;
};

struct v2f {
	float2 uv : TEXCOORD0;
	float3 viewDir : TEXCOORD1;
	float4 vertex : SV_POSITION;
};

v2f vert(appdata v) {
	v2f o;
	o.vertex = UnityObjectToClipPos(v.vertex);
	o.uv = v.uv;
	o.viewDir = _FrustrumCorners[o.uv.x+(o.uv.x ? 1-o.uv.y : (1-o.uv.y)*3)].xyz;
	o.viewDir = mul(_CameraInvViewMatrix, o.viewDir);
	return o;
}

struct ray {
	bool hit;
	float steps;
	float length;
	float3 origin;
	float3 direction;
};

struct light {
	float range;
	float intensity;
	float3 position;
	float3 color;
};


float3 rotate(float3 v, float3 a) {

	float3 c = cos(a);
	float3 s = sin(a);
	float3x3 mx = float3x3(1, 0, 0, 0, c.x, -s.x, 0, s.x, c.x);
	float3x3 my = float3x3(c.y, 0, s.y, 0, 1, 0, -s.y, 0, c.y);
	float3x3 mz = float3x3(c.z, -s.z, 0, s.z, c.z, 0, 0, 0, 1);
	return mul(mz, mul(my, mul(mx, v)));

}

float3 rotate(float3 v, float4 q) {

	float3 u = -q.xyz;
	return 2 * dot(u, v) * u + (q.w*q.w - dot(u, u)) * v + 2 * q.w * cross(u, v);
}

float sq(float x) {
	return x*x;
}

float3 sq(float3 x) {
	return x*x;
}

float remap(float x, float o1, float o2, float n1, float n2) {
	return (x - o1) / (o2 - o1) * (n2 - n1) + n1;
}

float mod(float x, float m) {
	float r = fmod(x, m);
	float o = r < 0 ? r + m : r;
	return o;
}
float3 mod(float3 x, float3 m) {
	float3 r = fmod(x, m);
	float3 o = float3(r.x < 0 ? r.x + m.x : r.x, r.y < 0 ? r.y + m.y : r.y, r.z < 0 ? r.z + m.z : r.z);
	return o;
}
float shmod(float x, float m) {
	return mod(x + m * .5, m) - m * .5;
}
float3 shmod(float3 x, float3 m) {
	return mod(x + m * .5, m) - m * .5;
}

float3 blend(float3 c1, float3 c2, float k) {
	return normalize(lerp(c1, c2, k))*lerp(length(c1), length(c2), k);
}

float smin(float a, float b, float k = .5) {
	float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
	return lerp(b, a, h) - k * h * (1.0 - h);
}

float4 smin(float4 a, float4 b, float k = .5) {
	float h = clamp(0.5 + 0.5 * (b.w - a.w) / k, 0.0, 1.0);
	return float4(blend(b.rgb, a.rgb, h), lerp(b.w, a.w, h) - k * h * (1.0 - h));
}

float smax(float a, float b, float k = .5) {
	float h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
	return lerp(-b, -a, h) - k * h * (1.0 - h);
}

float4 smax(float4 a, float4 b, float k = .5) {
	float h = clamp(0.5 + 0.5 * (a.w - b.w) / k, 0.0, 1.0);
	return float4(lerp(b.rgb, a.rgb, h), -lerp(-b.w, -a.w, h) - k * h * (1.0 - h));
}

float rand(float2 co){
	return frac(sin(dot(co.xy, float2(12.9898,78.233))) * 43758.5453);
}

float noise(float2 st) {
	float2 i = floor(st);
	float2 f = frac(st);

	// Four corners in 2D of a tile
	float a = rand(i);
	float b = rand(i + float2(1.0, 0.0));
	float c = rand(i + float2(0.0, 1.0));
	float d = rand(i + float2(1.0, 1.0));

	// Smooth Interpolation

	// Cubic Hermine Curve.  Same as SmoothStep()
	float2 u = f*f*(3.0-2.0*f);
	// u = smoothstep(0.,1.,f);

	// Mix 4 coorners percentages
	return lerp(a, b, u.x) +
	(c - a)* u.y * (1.0 - u.x) +
	(d - b) * u.x * u.y;
}

float luminance(float3 col) {
	//return col.x * 0.2126 + col.y * 0.7152 + col.z * 0.0722;
	return col.x* 0.2990 + col.y * 0.5870 + col.z * 0.1140;
}

void boxfold(inout float3 p, float b=1) {
	b *= .5;
	p = clamp(p, -b, b) * 2.0 - p;
}

void spherefold(inout float3 p, float R=1.0) {
	//float R = 1.0;
	float r = length(p);
	if (r<R) p = p*R*R/(r*r);
}



float4 sphere(float3 p, float r=1.0) {
	return float4(1, 1, 1, length(p) - r*.5);
}

float4 box(float3 p, float3 b = float3(1.0, 1.0, 1.0)) {
	b *= .5;
	float3 d = abs(p) - b;
	return float4(1, 1, 1, length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0));
}

float4 torus(float3 p, float2 t=float2(.5, .25)) {
	float2 q = float2(length(p.xy) - t.x, p.z);
	return float4(1, 1, 1, length(q) - t.y);
}

float4 cylinder(float3 p, float h=.5, float r=.5) {
	float2 d = abs(float2(length(p.xz),p.y)) - float2(h,r);
	return float4(1, 1, 1, min(max(d.x,d.y),0.0) + length(max(d,0.0)));
}

float4 mandelbulb(float3 p, float e=7, float iters=12, float bailout=10) {
	float3 z = p;
	float c = p;
	float dr = 1.0;
	float r = 0.0;
	float o = bailout;
	float o2 = bailout;
	float o3 = bailout;
	for (float i = 0; i < iters; i++) {
		r = length(z);
		o = min(o, length(z - float3(0, 0, 0)));
		o2 = min(o2, length(z - float3(0, 0, 1)));
		//o2 = max(o2, length(z - float3(0, 0, 0)));
		o3 = min(o3, length(z - float3(0, 0, 2)));
		if (r > bailout) break;

		// convert to polar coordinates
		float theta = acos(z.z / r);
		float phi = atan2(z.y, z.x);
		dr = pow(r, e - 1.0) * e * dr + 1.0;

		// scale and rotate the point
		float zr = pow(r, e);
		theta = theta * e;
		phi = phi * e;


		// convert back to cartesian coordinates
		z = zr * float3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
		z += p;

	}
	return float4(o, o2, o3, .5 * log(r) * r / dr);
}

float4 mandelbrot(float3 p, float e=2, float iters=12, float bailout=10) {
	float3 z = p;
	float c = p;
	float dr = 1.0;
	float r = 0.0;
	float o = bailout;
	float o2 = bailout;
	float o3 = bailout;
	for (float i = 0; i < iters; i++) {
		o = min(o, length(z - float3(0, 0, 0)));
		o2 = min(o2, length(z - float3(1, 0, 0)));
		o3 = min(o3, length(z - float3(2, 0, 0)));

		r = length(z);
		if (r > bailout) break;

		// convert to polar coordinates
		float theta = atan2(z.y, z.x);
		float phi = asin(z.z / r);
		dr = pow(r, e - 1.0) * e * dr + 1.0;

		// scale and rotate the point
		float zr = pow(r, e);
		theta = theta * e;
		phi = phi * e;


		// convert back to cartesian coordinates
		z = zr * float3(cos(theta) * cos(phi), sin(theta) * cos(phi), -sin(phi));
		z += p;

	}
	return float4(o, o2, o3, 0.5 * log(r) * r / dr);
}

float4 sierpinski(float3 p) {
	float x = p.x; float y = p.y; float z = p.z;
	float r = x * x + y * y + z * z;
	float scale = _FractalScale;
	float bailout = 20;
	float o = bailout;
	float o2 = bailout;
	float o3 = bailout;
	float3 c;
	for (float i = 0; i < 15 && r < bailout; i++) {
		//Folding... These are some of the symmetry planes of the tetrahedron
		if (x + y < 0) { float x1 = -y;y = -x;x = x1; }
		if (x + z < 0) { float x1 = -z;z = -x;x = x1; }
		if (y + z < 0) { float y1 = -z;z = -y;y = y1; }

		c = float3(x, y, z);
		c = rotate(c, float3(_FractalRotationX, _FractalRotationY, _FractalRotationZ));
		x = c.x; y = c.y; z = c.z;

		//Stretches about the point [1,1,1]*(scale-1)/scale; The "(scale-1)/scale" is here in order to keep the size of the fractal constant wrt scale
		x = scale * x - (scale - 1); //equivalent to: x=scale*(x-cx); where cx=(scale-1)/scale;
		y = scale * y - (scale - 1);
		z = scale * z - (scale - 1);
		r = x * x + y * y + z * z;
		o = min(o, length(float3(x, y, z) - float3(1, 0, 0)));
		o2 = min(o2, length(float3(x, y, z) - float3(0, 1, 0)));
		o3 = min(o3, length(float3(x, y, z) - float3(0, 0, 1)));
	}
	return float4(o, o2, o3, (sqrt(r) - 2) * pow(scale, -i)); //the estimated distance
}

float4 menger(float3 p) {
	int n,iters=12;float t;
	float x = p.x, y = p.y, z = p.z;
	float o = 50; float o2 = 50; float o3 = 50;
	for(n=0;n<iters;n++){
		x=abs(x);y=abs(y);z=abs(z);//fabs is just abs for floats
		if(x<y){t=x;x=y;y=t;}
		if(y<z){t=y;y=z;z=t;}
		if(x<y){t=x;x=y;y=t;}
		p = rotate(float3(x, y, z), float3(_FractalRotationX, _FractalRotationY, _FractalRotationZ));
		x = p.x; y = p.y; z = p.z;
		x=x*3.0-2.0;y=y*3.0-2.0;z=z*3.0-2.0;
		if(z<-1.0)z+=2.0;
		o = min(o, length(float3(x, y, z)   - float3(0, 0, 0)));
		o2 = min(o2, length(float3(x, y, z) - float3(0, .5, 0)));
		o3 = min(o3, length(float3(x, y, z) - float3(0, 0, .5)));
	}
	return float4(o, o2, o3, (sqrt(x*x+y*y+z*z)-1.5)*pow(3.0,-(float)iters));
}

float4 terrain(float2 x) {
	float2 p = x*0.003/250.0;
	float a = 0.0;
	float b = 1.0;
	float2 d = float2(0.0, 0.0);
	for( int i=0; i<3; i++ )
	{
		float2 n = noise(p);
		d += float2(n.y, 0.0);
		a += b*n.x/(1.0+dot(d,d));
		b *= 0.5;
		p = float2(.8, -.6)*p*2.0;
	}

	return float4(1, 1, 1, 250.0*120.0*a);
}



float4 scene(float3 p) {

	float3 p0 = (p-_Positions[0].xyz);
	float3 p1 = (p-_Positions[1].xyz);
	float3 p2 = (p-_Positions[2].xyz);
	float3 p3 = (p-_Positions[3].xyz);

	p0 = rotate(p0, _Rotations[0]);
	p1 = rotate(p1, _Rotations[1]);
	p2 = rotate(p2, _Rotations[2]);
	p3 = rotate(p3, _Rotations[3]);

	//_FractalRotationX = sin(_Time.y)*.5;
	//_FractalRotationY = sin(_Time.y*1.414)*.5;
	//_FractalRotationZ = sin(_Time.y*1.618)*.5;
	//return mandelbulb(rotate(p, float3(0, 0, _Time.y)), menger(rotate(p, float3(_Time.y*.25, _Time.y*1.4*.25, _Time.y*1.6*.25))).w+1.1);

	float4 o0 = box(p0, _Scales[0].xyz);
	float4 o1 = box(p1, _Scales[1].xyz);
	float4 o2 = box(p2, _Scales[2].xyz);
	float4 o3 = box(p3, _Scales[3].xyz);

	o0.rgb = float3(1, .25, .25);
	o1.rgb = float3(.25, 1, .25);
	o2.rgb = float3(.25, .25, 1);

	float blend = 1;

	return min(max(max(-o0, o1), -o2), o3);
}


ray raymarch(float3 ro, float3 rd, float depth=2000, float steps=2000) {
	depth = min(depth, _MaxDist);
	steps = min(steps, _MaxSteps);
	float dm = 0;
	bool hit;
	for (int i = 0; i < steps; i++) {
		float3 cp = ro + rd * dm;
		float dts = scene(cp).w;
		dm += abs(dts) * _StepFactor;
		_ContactThreshold = dm*.0025;
		if (abs(dts) < _ContactThreshold) {
			hit = true;
			break;
		}
		if (dm > depth) {
			break;
		}
	}
	ray r;
	r.hit = hit;
	r.steps = i;
	r.length = dm;
	r.origin = ro;
	r.direction = rd;
	return r;
}