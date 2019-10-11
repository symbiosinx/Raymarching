using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
[ImageEffectAllowedInSceneView]
public class ShadowMapDirectional : MonoBehaviour {
    
	[HideInInspector] public RenderTexture shadowMap;

	Vector3[] initialCorners = {
		new Vector3(-0.5f, 0.5f, 0f),
		new Vector3( 0.5f, 0.5f, 0f),
		new Vector3( 0.5f,-0.5f, 0f),
		new Vector3(-0.5f,-0.5f, 0f)
	};

	Vector3[] corners = {
		Vector3.zero,
		Vector3.zero,
		Vector3.zero,
		Vector3.zero
	};

    void Update() {
        for (int i = 0; i < 4; i++) {
			corners[i] = initialCorners[i];
			corners[i].x *= transform.lossyScale.x;
			corners[i].y *= transform.lossyScale.y;
			corners[i].z *= transform.lossyScale.z;
			corners[i] = transform.rotation * corners[i];
			corners[i] += transform.position;
		}
    }

	void OnDrawGizmos() {
		Gizmos.color = Color.yellow;
		Gizmos.DrawLine(corners[0], corners[1]);
		Gizmos.DrawLine(corners[1], corners[2]);
		Gizmos.DrawLine(corners[2], corners[3]);
		Gizmos.DrawLine(corners[3], corners[0]);
		foreach (Vector3 corner in corners) {
			Gizmos.DrawSphere(corner, 0.05f);
		}
		Gizmos.color = Color.cyan;
		Gizmos.DrawLine(transform.position, transform.position+transform.forward);
		
	}



}
