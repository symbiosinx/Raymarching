Shader "Hidden/Raymarching" {
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
        Cull Off ZWrite On ZTest Always

		Pass {
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#include "UnityCG.cginc"
			#include "Raymarching.cginc"
			
			sampler2D _MainTex;
			float3 _LightPosition;
			
	
			
			float _NormalSampleScale;
			float _RefractiveIndex;
			int _Reflections;
			samplerCUBE _Skybox;
			float3 _Color1;
			float3 _Color2;
			float _Glossiness;

			sampler2D _CameraDepthTexture;
			
			sampler2D _ShadowMap;
			float4x4 _ShadowMapVPMatrix;



			float3 getnormalraw(float3 p, float s = 0.) {
				float2 e = float2(max(s, _NormalSampleScale), 0);

				// Algorithm 1

				//return normalize(float3(
				//	scene(p + e.xyy),
				//	scene(p + e.yxy),
				//	scene(p + e.yyx)
				//) - scene(p));

				// Algorithm 2

				return (float3(
					scene(p + e.xyy).w - scene(p - e.xyy).w,
					scene(p + e.yxy).w - scene(p - e.yxy).w,
					scene(p + e.yyx).w - scene(p - e.yyx).w
					));

				// Algorithm 3

				//float2 k = float2(-1., 1.);
				//return normalize(
				//	k.xyy*scene(p+k.xyy*e.x) +
				//	k.yyx*scene(p+k.yyx*e.x) +
				//	k.yxy*scene(p+k.yxy*e.x) +
				//	k.xxx*scene(p+k.xxx*e.x)
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

			float getshadow(float3 p, float3 n, float k=16, float depth=2000, float steps=2000) {
				depth = min(depth, _MaxDist);
				steps = min(steps, _MaxSteps);
				float dm = 0;
				float res = 1;
				p += n * _ContactThreshold*2;
				for (int i = 0; i < steps; i++) {
					float3 cp = p + _WorldSpaceLightPos0 * dm;
					float dts = scene(cp).w;
					res = min(res, k*dts/dm);
					dm += abs(dts) * _StepFactor;
					_ContactThreshold = dm*.0025;
					if (abs(dts) < _ContactThreshold) {
						break;
					}
					if (dm > depth) {
						break;
					}
				}

				return res;
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

			fixed4 frag(v2f i) : SV_Target {

				float4 tex = tex2D(_MainTex, i.uv);
				float3 view = i.viewDir;

				float depth = Linear01Depth(tex2D(_CameraDepthTexture, i.uv)) * _ProjectionParams.z;
				bool ghit = depth != _ProjectionParams.z;

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

				ray r = raymarch(ro, rd, depth);

				dist += r.length;
				float3 hitpoint = ro + rd * r.length;
				float3 rawnormal = getnormalraw(hitpoint);
				float3 normal = normalize(rawnormal);

				//for (int i = 0; i < 1 && r.hit; i++) {
				//	ro = hitpoint - normal * _ContactThreshold * 2;
				//	rd = sign(scene(hitpoint).w >= 0) ? refract(rd, normal, 1.1) : reflect(rd, normal);
				//	r = raymarch(ro, rd);
				//	dist += r.length;
				//	hitpoint = ro + rd * r.length;
				//	rawnormal = getnormalraw(hitpoint);
				//	normal = normalize(rawnormal);
				//	col = clamp(scene(hitpoint).rgb * .75, 0, 1);
				//}

				//if (r.hit) rd = reflect(rd, normal);

				float3 sky = texCUBE(_Skybox, rd);

				float3 srgb = scene(hitpoint).rgb;

				float shadow = remap(getshadow(hitpoint, normal), 0, 1, .25, 1);

				col = r.hit ?
					//clamp(scene(hitpoint).r*2.5-1.75, 0, 1)
					srgb * (dot(normal, _WorldSpaceLightPos0)*.5+.5) * shadow * getAO(rawnormal)
					//shadow
					//normal*.5+.5
				: tex;	
				//col += r.steps / 100 * _StepFactor;


				//float la = 0;
				//for (float i = 0; i*.1 < min(5, r.length); i++) {
				//	float3 rp = ro+rd*i*.1;
				//	la += getshadow(rp, 0)*.05;
				//}
				//
				//col.rgb += la;

				return fixed4(col, 1);

			}
			ENDCG
		}

		GrabPass {
			"_ScreenTexture"
		}

		Pass {
			CGPROGRAM
			#pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
			
			sampler2D _CameraDepthTexture;
			sampler2D _ScreenTexture;

			struct appdata {
				float2 uv : TEXCOORD0;
				float4 vertex : POSITION;
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

			fixed4 frag(v2f i) : SV_Target {
				float4 tex = tex2D(_ScreenTexture, i.uv);
				return tex;
			}
			ENDCG
		}
    }
}
