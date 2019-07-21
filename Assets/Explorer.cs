using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Explorer : MonoBehaviour {

	public Material mat;

	float speed = 5;
	float rotSpeed = 60;

	Vector3 pos = Vector3.up;
	float rot = 0;

	void Update() {

		Vector3 dir = Vector3.zero;

		if (Input.GetKey(KeyCode.Space)) {
			dir += Vector3.up;
		}
		if (Input.GetKey(KeyCode.LeftControl)) {
			dir -= Vector3.up;
		}
		if (Input.GetKey(KeyCode.W)) {
			dir += Vector3.forward;
		}
		if (Input.GetKey(KeyCode.S)) {
			dir -= Vector3.forward;
		}
		if (Input.GetKey(KeyCode.D)) {
			dir += Vector3.right;
		}
		if (Input.GetKey(KeyCode.A)) {
			dir -= Vector3.right;
		}
		if (Input.GetKey(KeyCode.RightArrow)) {
			rot += rotSpeed * Time.deltaTime;
		}
		if (Input.GetKey(KeyCode.LeftArrow)) {
			rot -= rotSpeed * Time.deltaTime;
		}

		pos += (Quaternion.Euler(0f, rot, 0f) * dir) * speed * Time.deltaTime;



		mat.SetVector("_Position", new Vector4(pos.x, pos.y, pos.z, -rot*Mathf.Deg2Rad));
	}
}
