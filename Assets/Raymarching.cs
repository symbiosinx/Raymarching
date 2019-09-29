using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
[ImageEffectAllowedInSceneView]
public class Raymarching : MonoBehaviour {


	public Material mat;
	public ReflectionProbe reflectionProbe;

	Camera cam;

	void Start() {
		cam = GetComponent<Camera>();
		cam.renderingPath = RenderingPath.DeferredShading;
		cam.depthTextureMode = cam.depthTextureMode | DepthTextureMode.Depth;
		//mat.SetVector("_AmbientLight", RenderSettings.ambientLight);
		mat.SetVector("_AmbientLight", new Vector4(0.396f, 0.478f, 0.592f, 1f));
		//mat.SetTexture("_Skybox", RenderSettings.customReflection);
		mat.SetTexture("_Skybox", reflectionProbe.bakedTexture);
	}

	Matrix4x4 GetFrustumCorners() {
    	float camFov = cam.fieldOfView;
    	float camAspect = cam.aspect;

    	Matrix4x4 frustumCorners = Matrix4x4.identity;

    	float fovWHalf = camFov * 0.5f;
		
    	float tan_fov = Mathf.Tan(fovWHalf * Mathf.Deg2Rad);

    	Vector3 toRight = Vector3.right * tan_fov * camAspect;
    	Vector3 toTop = Vector3.up * tan_fov;

    	Vector3 topLeft = (-Vector3.forward - toRight + toTop);
    	Vector3 topRight = (-Vector3.forward + toRight + toTop);
    	Vector3 bottomRight = (-Vector3.forward + toRight - toTop);
    	Vector3 bottomLeft = (-Vector3.forward - toRight - toTop);

    	frustumCorners.SetRow(0, topLeft);
    	frustumCorners.SetRow(1, topRight);
    	frustumCorners.SetRow(2, bottomRight);
    	frustumCorners.SetRow(3, bottomLeft);
		
    	return frustumCorners;
	}

	void OnRenderImage(RenderTexture src, RenderTexture dest) {
		mat.SetMatrix("_FrustrumCorners", GetFrustumCorners());
		mat.SetMatrix("_CameraInvViewMatrix", cam.cameraToWorldMatrix);
		Graphics.Blit(src, dest, mat);
	}
}
