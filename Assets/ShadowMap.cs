using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
[ImageEffectAllowedInSceneView]
[RequireComponent(typeof(Camera))]
public class ShadowMap : MonoBehaviour {

	public Material shadowMapMaterial;

	[HideInInspector] public RenderTexture shadowMap;
	[HideInInspector] public Camera cam;

	void Start() {
		cam = GetComponent<Camera>();
		cam.depthTextureMode = DepthTextureMode.None;
	}

    void Update() {
        
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

	public void Render() {
		RenderTexture r = RenderTexture.GetTemporary(100, 100);
		shadowMapMaterial.SetMatrix("_FrustrumCorners", GetFrustumCorners());
		shadowMapMaterial.SetMatrix("_CameraInvViewMatrix", cam.cameraToWorldMatrix);
		Graphics.Blit(r, shadowMap, shadowMapMaterial);
		RenderTexture.ReleaseTemporary(r);
	}

}
