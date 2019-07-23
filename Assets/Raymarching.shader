Shader "Custom/Raymarching" {
    Properties {
		_Position("Camera Position", Vector) = (0, 1, 0, 0)
		_MaxSteps("Max Steps", Range(0, 200)) = 100
		_MaxDist("Max Distance", Range(0, 200)) = 100
		_ContactThreshold("Contact Threshold", Range(0.00001, 0.1)) = 0.01
		_Size("Size", Range(.1, 20)) = 1
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

			float4 _Position;
			float _MaxSteps;
			float _MaxDist;
			float _ContactThreshold;
			float _Size;

			struct ray {
				bool hit;
				float steps;
				float length;
			};

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
				return mod(x, m) - m * .5;
			}
			float3 shmod(float3 x, float3 m) {
				return mod(x, m) - m * .5;
			}

			float smin(float a, float b, float k = 1) {
				float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return lerp(b, a, h) - k * h * (1.0 - h);
			}

			float sdsphere(float3 p, float r=1.0) {
				return length(p) - r;
			}

			float sdtorus(float3 cpos, float3 pos, float2 t) {
				float3 p = cpos - pos;
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

			float sdscene(float3 p) {
				//float s = smin(sdsphere(cpos, float3(1, 1, 5), 1), sdsphere(cpos, float3(-1, 1, 5), 1));
				//float t = sdtorus(cpos, float3(0, 0, 5), float2(2, .5));
				//float p = cpos.y;
				//return smin(p, t);
				//float3 cpos = _Position.xyz;
				//float3 p1 = shmod(p, (float3)(5));
				float mb = sdmandelbulb(p-float3(0, 0, 5), 7).x;
				return mb;
			}

			ray raymarch(float3 ro, float3 rd) {
				float dm = 0;
				bool hit;
				for (int i = 0; i < _MaxSteps; i++) {
					float3 cp = ro + rd * dm;
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

			float3 getnormal(float3 pos) {
				float d = sdscene(pos);
				float2 e = float2(.000001, 0);
				float3 n = d - float3(
					sdscene(pos - e.xyy),
					sdscene(pos - e.yxy),
					sdscene(pos - e.yyx));
				//if (length(n) == 0) { return n; }
				return normalize(n);
			}

			float getlight(float3 pos) {
				float3 lpos = /*_WorldSpaceLightPos0*/ float3(0, 5, 0);
				float3 n = getnormal(pos);
				float3 l = normalize(lpos - pos);
				float dif = clamp(dot(n, l), 0, 1);
				float d = raymarch(pos + n * _ContactThreshold * 2, l).length;
				if (d < length(lpos - pos)) {
					dif *= 0.1;
				}
				return dif;
			}

			fixed4 frag(v2f i) : SV_Target{

				fixed4 col;
				col.a = 1;

				float3 view = float3((-i.uv+.5)*_Size, 1);
				view = float3(view.x * cos(_Position.w) - view.z * sin(_Position.w), view.y, view.x * sin(_Position.w) + view.z * cos(_Position.w));

				float3 ro = _Position;
				float3 rd = normalize(view);
				
				ray r = raymarch(ro, rd);
				float steps = r.steps;

				//col.rgb = 1-(r.steps/100);
				//col.rgb = r.hit ? dot(getnormal(ro + rd * r.length), float3(0, 1, 0)+.5): 0;

				float dist = r.length;
				float3 p = ro + rd * dist;

				col.rgb = sdmandelbulb(p).y;

				//float dif = getlight(p);
				//col.rgb = dif;
				return col;

            }
            ENDCG
        }
    }
}
