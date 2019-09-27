Shader "Custom/Raymarching" {
    Properties {
		[HideInInspector] _MainTex("Main Texture", 2D) = "white" {}
		_LightPosition("Light Position", Vector) = (0, 10, 0, 0)
		_StepFactor("Step Factor", Range(0.05, 1)) = 1
		_MaxSteps("Max Steps", Range(0, 2000)) = 100
		_MaxDist("Max Distance", Range(0, 2000)) = 100
		_ContactThreshold("Contact Threshold", Range(0.00001, 0.1)) = 0.01
		_NormalSampleScale("Normal Sample Scale", Range(0.00001, 0.01)) = 0.01
		_Reflections("Number of Reflections", Range(0, 10)) = 1
		_RefractiveIndex("Refractive Index", Range(1, 2)) = 1.5
		_Color1("Color 1", Color) = (0, 0, 0, 1)
		_Color2("Color 2", Color) = (1, 1, 1, 1)
		_FractalScale("Fractal Scale", Range(1.2, 2)) = 1.5
		_FractalRotationX("Fractal Rotation X", Range(-.5, .5)) = 0
		_FractalRotationY("Fractal Rotation Y", Range(-.5, .5)) = 0
		_FractalRotationZ("Fractal Rotation Z", Range(-.5, .5)) = 0
		_Glossiness("Glossiness", Range(0, 1)) = 0
    }
    SubShader {
        Cull Off ZWrite On ZTest LEqual

        Pass {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

			sampler2D _MainTex;
			float3 _LightPosition;
			float _StepFactor;
			float _MaxSteps;
			float _MaxDist;
			float _ContactThreshold;
			float _NormalSampleScale;
			float _RefractiveIndex;
			int _Reflections;
			samplerCUBE _Skybox;
			float3 _Color1;
			float3 _Color2;
			float _FractalScale;
			float _FractalRotationX;
			float _FractalRotationY;
			float _FractalRotationZ;
			float _Glossiness;
			float4x4 _FrustrumCorners;
			float4x4 _CameraInvViewMatrix;
			sampler2D _CameraDepthTexture;

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


			float4 qmul(float4 v, float4 q) {
				return float4(
					v.x*q.x - v.y*q.y - v.z*q.z - v.w * q.w,
					v.x*q.y + v.y*q.x - v.z*q.w + v.w * q.z,
					v.x*q.z + v.y*q.w + v.z*q.x - v.w * q.y,
					v.x*q.w - v.y*q.z + v.z*q.y + v.w * q.x
				);
			}

			float3 rotate(float3 v, float3 a) {
				float3 c = cos(a);
				float3 s = sin(a);
				float3x3 mx = float3x3(1, 0, 0, 0, c.x, -s.x, 0, s.x, c.x);
				float3x3 my = float3x3(c.y, 0, s.y, 0, 1, 0, -s.y, 0, c.y);
				float3x3 mz = float3x3(c.z, -s.z, 0, s.z, c.z, 0, 0, 0, 1);
				return mul(mz, mul(my, mul(mx, v)));

				//float cy = cos(a.x * 0.5);
				//float sy = sin(a.x * 0.5);
				//float cp = cos(a.y * 0.5);
				//float sp = sin(a.y * 0.5);
				//float cr = cos(a.z * 0.5);
				//float sr = sin(a.z * 0.5);
				//float4 q = float4(
				//	cy * cp * cr + sy * sp * sr,
				//	cy * cp * sr - sy * sp * cr,
				//	sy * cp * sr + cy * sp * cr,
				//	sy * cp * cr - cy * sp * sr
				//);
				//return qmul(qmul(float4(v, 0), q), float4(q.x, -q.y, -q.z, -q.w)).xyz;
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

			float smin(float a, float b, float k = .5) {
				float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return lerp(b, a, h) - k * h * (1.0 - h);
			}

			float smax(float a, float b, float k = .5) {
				// Temporary code
				return -smin(-a, -b, k);
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
				return float4(length(p) - r*.5, 1, 1, 1);
			}

			float4 box(float3 p, float3 b = float3(1.0, 1.0, 1.0)) {
				float3 d = abs(p) - b;
				return float4(length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0), 1, 1, 1);
			}

			float4 torus(float3 p, float2 t=float2(1, .5)) {
				float2 q = float2(length(p.xy) - t.x, p.z);
				return float4(length(q) - t.y, 1, 1, 1);
			}

			float4 cylinder(float3 p, float h=1, float r=1) {
				float2 d = abs(float2(length(p.xz),p.y)) - float2(h,r);
				return float4(min(max(d.x,d.y),0.0) + length(max(d,0.0)), 1, 1, 1);
			}

			float4 mandelbulb(float3 p, float e=8, float iters=12, float bailout=10) {
				float3 z = p;
				float dr = 1.0;
				float r = 0.0;
				float o = bailout;
				float o2 = bailout;
				float o3 = bailout;
				for (float i = 0; i < iters; i++) {
					r = length(z);
					o = min(o, length(z - float3(0, 0, 0)));
					o2 = min(o2, length(z - float3(0, 0, .25)));
					o3 = min(o3, length(z - float3(0, 0, 16)));
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
				return float4(0.5 * log(r) * r / dr, o, o2, o3);
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
				return float4((sqrt(r) - 2) * pow(scale, -i), o, o2, o3); //the estimated distance
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
				return float4((sqrt(x*x+y*y+z*z)-1.5)*pow(3.0,-(float)iters), o, o2, o3);
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

				return float4(250.0*120.0*a, 1, 1, 1);
			}



			float4 sdscene(float3 p) {
				return mandelbulb(p, 2);
			}


			ray raymarch(float3 ro, float3 rd) {
				float dm = 0;
				bool hit;
				for (int i = 0; i < _MaxSteps; i++) {
					float3 cp = ro + rd * dm;
					float dts = sdscene(cp);
					dm += abs(dts) * _StepFactor;
					if (abs(dts) < _ContactThreshold) {
						hit = true;
						break;
					}
					if (dm > _MaxDist) {
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

			float3 getnormalraw(float3 p, float s = 0.) {
				float2 e = float2(max(s, _NormalSampleScale), 0);

				// Algorithm 1

				//return normalize(float3(
				//	sdscene(p + e.xyy),
				//	sdscene(p + e.yxy),
				//	sdscene(p + e.yyx)
				//) - sdscene(p));

				// Algorithm 2

				return (float3(
					sdscene(p + e.xyy).x - sdscene(p - e.xyy).x,
					sdscene(p + e.yxy).x - sdscene(p - e.yxy).x,
					sdscene(p + e.yyx).x - sdscene(p - e.yyx).x
					));

				// Algorithm 3

				//float2 k = float2(-1., 1.);
				//return normalize(
				//	k.xyy*sdscene(p+k.xyy*e.x) +
				//	k.yyx*sdscene(p+k.yyx*e.x) +
				//	k.yxy*sdscene(p+k.yxy*e.x) +
				//	k.xxx*sdscene(p+k.xxx*e.x)
				//);
			}

			float3 getnormal(float3 p, float s=0.) {
				return normalize(getnormalraw(p, s));
			}

			float getbrightnesshard(float3 p, float3 n, light l) {
				float dist = length(p - l.position);
				if (dist >= l.range) {
					return 0.0;
				}
				return lerp(clamp(dot(n, l.position-p), 0, 1) * l.intensity, 0.0, dist/l.range);
			}

			float getbrightness(float3 p, float3 n, light l) {
				float dist = length(p - l.position);
				if (dist >= l.range) {
					return 0.0;
				}
				float d = clamp(dot(n, l.position - p), -1, 1) * .5 + .5;
				d *= d;
				return lerp(d * l.intensity, 0.0, dist / l.range);
			}

			float getshadow(float3 p, float3 n) {
				float3 l = normalize(_LightPosition - p);
				float d = raymarch(p + n * _ContactThreshold * 2, l).length;
				if (d < length(_LightPosition - p)) {
					return 0;
				}
				return 1;
			}

			float getAO(float3 n) {
				return clamp(length(n/_NormalSampleScale), 0, 1);
			}

			float getscatter(ray r, light l) {
				// light to ray origin
				float3 q = r.origin - l.position;

				// coefficients
				float b = dot(r.direction, q);
				float c = dot(q, q);

				// evaluate integral
				float s = l.intensity / sqrt(c - b * b);
				return s * (atan((r.length + b) * s) - atan(b * s));
			}

			float3 refract2(float3 i, float3 n, float eta) {
				eta = 2.0f - eta;
				float cosi = dot(n, i);
				float3 o = (i * eta - n * (-cosi + eta * cosi));
				return o;
			}

			fixed4 frag(v2f i) : SV_Target{

				float4 tex = tex2D(_MainTex, i.uv);
				float3 view = i.viewDir;

				float depth = Linear01Depth(tex2D(_CameraDepthTexture, i.uv)) * _ProjectionParams.z;

				light l;
				l.range = 1000;
				l.intensity = 1;
				l.position = _LightPosition;
				//l.color = float3(1, .4, .2);
				l.color = float3(.8, .7, .6);

				fixed3 col;
				col = 1;

				float3 ro = _WorldSpaceCameraPos;
				float3 rd = view;
				float dist = 0.0;

				ray r = raymarch(ro, rd);
				if (r.length > depth) return tex;
				dist += r.length;
				float3 hitpoint = ro + rd * r.length;
				float3 rawnormal = getnormalraw(hitpoint);
				float3 normal = normalize(rawnormal);

				//for (int i = 0; i < 1 && r.hit; i++) {
				//	ro = hitpoint - normal * _ContactThreshold * 2;
				//	rd = sign(sdscene(hitpoint).x >= 0) ? refract(rd, normal, 1.1) : reflect(rd, normal);
				//	r = raymarch(ro, rd);
				//	dist += r.length;
				//	hitpoint = ro + rd * r.length;
				//	rawnormal = getnormalraw(hitpoint);
				//	normal = normalize(rawnormal);
				//	col = clamp(sdscene(hitpoint).yzw * .75, 0, 1);
				//}

				if (r.hit) rd = reflect(rd, normal);

				float3 sky = texCUBE(_Skybox, rd);

				col = r.hit ?
					lerp(sdscene(hitpoint).yzw, sky, _Glossiness * clamp((1-dot(normal, rd))*.5+.5, 0, 1))/2
				: tex;
				col += r.steps / 100 * _StepFactor;

				return fixed4(col, 1);

            }
            ENDCG
        }
    }
}
