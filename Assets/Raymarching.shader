Shader "Custom/Raymarching" {
    Properties {
		_AspectRatio("Aspect Ratio", Float) = 1.777778
		_FieldOfView("Field of View", Range(.1, 20)) = 1
		_Position("Camera Position", Vector) = (0, 1, -3, 0)
		_Rotation("Camera Rotation", Vector) = (0, 0, 0, 0)
		_LightPosition("Light Position", Vector) = (0, 10, 0, 0)
		_MaxSteps("Max Steps", Range(0, 2000)) = 100
		_MaxDist("Max Distance", Range(0, 2000)) = 100
		_ContactThreshold("Contact Threshold", Range(0.00001, 0.1)) = 0.01
		_NormalSampleScale("Normal Sample Scale", Range(0.00001, 0.1)) = 0.01
    }
    SubShader {
        Cull Off ZWrite On ZTest LEqual

        Pass {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v) {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

			float _AspectRatio;
			float _FieldOfView;
			float3 _Position;
			float2 _Rotation;
			float3 _LightPosition;
			float _MaxSteps;
			float _MaxDist;
			float _ContactThreshold;
			float _NormalSampleScale;

			struct ray {
				bool hit;
				float steps;
				float length;
			};

			float3 rotate(float3 v, float3 a) {
				float3 c = cos(a);
				float3 s = sin(a);
				float3x3 mx = float3x3(1, 0, 0, 0, c.x, -s.x, 0, s.x, c.x);
				float3x3 my = float3x3(c.y, 0, s.y, 0, 1, 0, -s.y, 0, c.y);
				float3x3 mz = float3x3(c.z, -s.z, 0, s.z, c.z, 0, 0, 0, 1);
				return mul(mz, mul(my, mul(mx, v)));
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

			float sdsphere(float3 p, float r=1.) {
				return length(p) - r*.5;
			}

			float sdtorus(float3 p, float2 t=float2(1, .5)) {
				float2 q = float2(length(p.xy) - t.x, p.z);
				return length(q) - t.y;
			}


			float2 sdmandelbulb(float3 p, float e=8, float iters=12, float bailout=10) {
				float3 z = p;
				float dr = 1.0;
				float r = 0.0;
				float o = bailout;
				for (float i = 0; i < iters; i++) {
					r = length(z);
					o = min(o, r);
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
				return float2(0.5 * log(r) * r / dr, o/iters);
			}

			float sdmandelbrot(float3 p, int iters=10, float bailout=10) {
				float3 c, c2;
				float r = 0;
				float dr = 1;
				for (int i = 0; i < iters; i++) {
					r = length(c);
					if (r > bailout) {
						break;
					}
					c = float3((c2.x - c2.y) * (1 - c2.z / (c2.x + c2.y)), 2 * c.x * c.y * (1 - c2.z / (c2.x + c2.y)), -2 * c.z * sqrt(c2.x + c2.y));
					dr = pow(r, 2 - 1.0) * 2 * dr + 1.0;
					c += p;
					c2 = c*c;
				}
				return .5 * log(r) * r / dr;
			}

			float sdsierpinski(float3 z) {
				float3 a1 = float3(1,1,1);
				float3 a2 = float3(-1,-1,1);
				float3 a3 = float3(1,-1,-1);
				float3 a4 = float3(-1,1,-1);
				float3 c;
				int n = 0;
				float dist, d;
				while (n < 20) {
					c = a1; dist = length(z-a1);
					d = length(z-a2); if (d < dist) { c = a2; dist=d; }
					d = length(z-a3); if (d < dist) { c = a3; dist=d; }
					d = length(z-a4); if (d < dist) { c = a4; dist=d; }
					z = 2.0*z-c*(2.0-1.0);
					n++;
				}
			return length(z) * pow(2.0, float(-n));
			}

			float sdmenger(float3 p){//this is our old friend menger
				int n,iters=12;float t;
				float x = p.x, y = p.y, z = p.z;
				for(n=0;n<iters;n++){
					x=abs(x);y=abs(y);z=abs(z);//fabs is just abs for floats
					if(x<y){t=x;x=y;y=t;}
					if(y<z){t=y;y=z;z=t;}
					if(x<y){t=x;x=y;y=t;}
					x=x*3.0-2.0;y=y*3.0-2.0;z=z*3.0-2.0;
					if(z<-1.0)z+=2.0;
				}
				return (sqrt(x*x+y*y+z*z)-1.5)*pow(3.0,-(float)iters);
			}

			float sdscene(float3 p) {
				//float s = smin(sdsphere(cpos, float3(1, 1, 5), 1), sdsphere(cpos, float3(-1, 1, 5), 1));
				//float t = sdtorus(cpos, float3(0, 0, 5), float2(2, .5));
				//float p = cpos.y;
				//return smin(p, t);
				//float3 cpos = _Position.xyz;
				//float3 p1 = shmod(p, (float3)(5));
				//return sdmandelbulb(p-float3(0, 0, 5), 7).x;
				//float2 r1 = float2(cos(1.2), sin(1.2));
				//float3x3 m1 = { r1.x, 0, r1.y,
				//			   0, 1, 0,
				//			   r1.y, 0, r1.x };
				//float2 r2 = float2(cos(2), sin(2));
				//float3x3 m2 = { 1, 0, 0, 
				//				0, r2.x, -r2.y,
				//				0, r2.y, r2.x };
				//float2 r3 = float2(cos(.5), sin(.5));
				//float3x3 m3 = { r3.x, -r3.y, 0, 
				//				r3.y, r3.x, 0,
				//				0, 0, 1 };
				//float3 p1 = shmod(mul(m1, p+float3(1, 1, 1)), float3(4, 4, 4));
				//float3 p2 = shmod(mul(m2, p), float3(4, 4, 4));
				//float3 p3 = shmod(mul(m3, p+float3(-1, -1, -2)), float3(5, 4, 3));
				//return smin(smin(sdsierpinski(p1), sdmenger(p2), 1), sdmandelbulb(p3, 8));
				//return sdmenger(p.x, p.y, p.z);
				//p = float3(p.x * cos(p.y * .0) - p.z * sin(p.y * .0), p.y, p.x * sin(p.y * .0) + p.z * cos(p.y * .0));
				//p = float3(p.y * cos(p.x * .0) - p.z * sin(p.x * .0), p.x, p.y * sin(p.x * .0) 1+ p.z * cos(p.x * .0));
				//p = rotate(p, float3(sin(p.y*.5+_Time.y), sin(p.z*.5+_Time.y), sin(p.x*.5+_Time.y)));
				//float3 p1 = rotate(shmod(p, 5), float3(0, _Time.y, 0));
				//return sdmandelbulb(p1, 9);
				return min(abs(rotate(p, float3(0, 0, 0)).y), smin(sdmandelbulb(rotate(p-float3(0, 2, 0), float3(UNITY_PI*.5, 0, 0)), 8), sdsphere(p-float3(1, 3, 0), 2)));

			}

			ray raymarch(float3 ro, float3 rd) {
				float dm = 0;
				bool hit;
				for (int i = 0; i < _MaxSteps; i++) {
					float3 cp = ro + rd * dm	;
					float dts = sdscene(cp);
					dm += dts;
					if (dts < _ContactThreshold) {
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
				return r;
			}

			float3 getnormal(float3 p) {
				float2 e = float2(_NormalSampleScale, 0);

				// Algorithm 1
				
				//return normalize(float3(
				//	sdscene(p + e.xyy),
				//	sdscene(p + e.yxy),
				//	sdscene(p + e.yyx)
				//) - sdscene(p));
				
				// Algorithm 2

				return normalize(float3(
					sdscene(p+e.xyy) - sdscene(p-e.xyy),
					sdscene(p+e.yxy) - sdscene(p-e.yxy),
					sdscene(p+e.yyx) - sdscene(p-e.yyx)
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

			float getlight(float3 p) {
				float3 n = getnormal(p);
				float3 l = normalize(_LightPosition - p);
				float dif = clamp(dot(n, l), 0, 1);
				return dif;
			}

			float getshadow(float3 p) {
				float3 n = getnormal(p);
				float3 l = normalize(_LightPosition - p);
				float d = raymarch(p + n * _ContactThreshold * 2, l).length;
				if (d < length(_LightPosition - p)) {
					return 0;
				}
				return 1;
			}

			fixed4 frag(v2f i) : SV_Target{

				fixed4 col;
				col.a = 1;

				float3 view = float3(-i.uv.x*_AspectRatio+_AspectRatio*.5, -i.uv.y+.5, 1./_FieldOfView);
				view = normalize(rotate(view, float3(_Rotation.yx, 0.)));


				float3 forward = rotate(float3(0., 0., 1 / _FieldOfView), float3(_Rotation.yx, 0.));

				float3 ro = _Position;
				float3 rd = normalize(view);
				
				ray r = raymarch(ro, rd);
				
				float3 hitpoint = ro + rd * r.length;
				float3 normal = getnormal(hitpoint);

				float3 light = normalize(_LightPosition - hitpoint);

				if (r.hit) {
					if (abs(hitpoint.y) < _ContactThreshold) {
						if (mod(hitpoint.x, 2) < 1 ^ mod(hitpoint.z, 2) < 1) {
							col.rgb = .2;
						} else {
							col.rgb = 1;
						}
					}
					else {
						if (dot(normal, -view) < .3) {
							col.rgb = 0;
						} else {
							col.rgb = dot(normal, light) > 0 ? .8 : .5;
							float3 ref = reflect(light, normal);
							float spec = smoothstep(pow(max(dot(view, ref), 0), 16), .1, 0);
							col += spec;
						}

						
					}
					col.rgb *= remap(getshadow(hitpoint), 0, 1, .5, 1);
					//col.rgb *= (1 - r.steps / 100);
				} else {
					col.rgb = 0;
				}



				//float dist = r.length;
				//float3 p = ro + rd * dist;

				//col.rgb = sdmandelbulb(p).y;

				//float dif = getlight(p);
				//col.rgb = dif;
				return col;

            }
            ENDCG
        }
    }
}
