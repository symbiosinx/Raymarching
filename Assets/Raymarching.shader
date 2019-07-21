Shader "Custom/Raymarching"
{
    Properties
    {
		_Position("Camera Position", Vector) = (0, 1, 0, 0)
		_MaxSteps("Max Steps", Range(0, 200)) = 100
		_MaxDist("Max Distance", Range(0, 200)) = 100
		_ContactThreshold("Contact Threshold", Range(0.00001, 0.1)) = 0.01
		_Size("Size", Range(.1, 20)) = 1
    }
    SubShader
    {
        Cull Off ZWrite On ZTest LEqual

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v)
            {
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

			float mod(float x, float m) {
				float r = fmod(x, m);
				return r < 0 ? r + m : r;
			}
			float3 mod(float3 x, float3 m) {
				float3 r = fmod(x, m);
				float3 o = float3(r.x < 0 ? r.x + m.x : r.x, r.y < 0 ? r.y + m.y : r.y, r.z < 0 ? r.z + m.z : r.z);
				return o;
			}

			float smin(float a, float b, float k = 1) {
				float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return lerp(b, a, h) - k * h * (1.0 - h);
			}

			float sdsphere(float3 cpos, float3 pos, float r) {
				float3 p = cpos - pos;
				return length(p) - r;
			}

			float sdtorus(float3 cpos, float3 pos, float2 t) {
				float3 p = cpos - pos;
				float2 q = float2(length(p.xy) - t.x, p.z);
				return length(q) - t.y;
			}

			float sdsphereinf(float3 cpos, float3 pos, float3 c) {
				float3 p = cpos - pos;
				float3 q = mod(p, c) - c*.5;
				return sdsphere(q, float3(0, 0, 0), 1);
			}

			float sdmandelbulb(float3 p, float3 e=7, int iter=10, float bailout=10) {
				float3 z = p;
				float dr = 1.0;
				float r = 0.0;
				for (int i = 0; i < iter; i++) {
					r = length(z);
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
				return 0.5 * log(r) * r / dr;
			}

			float sdscene(float3 cpos) {
				//float s = smin(sdsphere(cpos, float3(1, 1, 5), 1), sdsphere(cpos, float3(-1, 1, 5), 1));
				//float t = sdtorus(cpos, float3(0, 0, 5), float2(2, .5));
				//float p = cpos.y;
				//return smin(p, t);
				//float s = sdsphereinf(cpos, float3(0, 0, 0), float3(3, 6, 5));
				//return s;
				float mb = sdmandelbulb(cpos - float3(0, 0, 5));
				return mb;
			}

			float raymarch(float3 ro, float3 rd) {
				float dm = 0;
				for (int i = 0; i < _MaxSteps; i++) {
					float3 cp = ro + rd * dm;
					float dts = sdscene(cp);
					dm += dts;
					if (dm > _MaxDist || dts < _ContactThreshold) {
						break;
					}
				}
				return i;
			}

			float3 getnormal(float3 pos) {
				float d = sdscene(pos);
				float2 e = float2(.01, 0);
				float3 n = d - float3(
					sdscene(pos - e.xyy),
					sdscene(pos - e.yxy),
					sdscene(pos - e.yyx));
				//if (length(n) == 0) { return n; }
				return normalize(n);
			}

			float getlight(float3 pos) {
				float3 lpos = /*_WorldSpaceLightPos0*/ float3(0, 3, 0);
				float3 n = getnormal(pos);
				float3 l = normalize(lpos - pos);
				float dif = clamp(dot(n, l), 0, 1);
				//float d = raymarch(pos + n * _ContactThreshold * 2, l);
				//if (d < length(lpos - pos)) {
				//	dif *= 0.1;
				//}
				return dif;
			}

			fixed4 frag(v2f i) : SV_Target
			{

				float3 view = float3((-i.uv+.5)*_Size, 1);
				view = float3(view.x * cos(_Position.w) - view.z * sin(_Position.w), view.y, view.x * sin(_Position.w) + view.z * cos(_Position.w));

				float3 ro = _Position;
				float3 rd = normalize(view);
				float dist = raymarch(ro, rd);
				return float4(1-dist/100, 1-dist/100, 1-dist/100, 1);
				float3 p = ro + rd * dist;
				float dif = getlight(p);
				float3 col = float3(dif, dif, dif);
				return float4(col, 1);

            }
            ENDCG
        }
    }
}
