using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Explorer : MonoBehaviour {

	public Transform plane;
	public Material mat;

	public float aspectRatio = 1;

	float speed = 5;
	float rotSpeed = 60;

	Vector3 pos = new Vector3(0f, 1f, -3f);
	Vector2 rot = Vector2.zero;

	void Start() {
		Cursor.visible = false;
	}

	void Update() {

		plane.localScale = new Vector3(aspectRatio, 1f, 1f);

		Vector3 dir = Vector3.zero;

		speed = 5f;

		if (Input.GetKey(KeyCode.LeftShift)) {
			speed = 0.5f;
		}
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
		if (Input.GetKey(KeyCode.UpArrow)) {
			rot.y += rotSpeed * Time.deltaTime;
		}
		if (Input.GetKey(KeyCode.DownArrow)) {
			rot.y -= rotSpeed * Time.deltaTime;
		}
		if (Input.GetKey(KeyCode.RightArrow)) {
			rot.x += rotSpeed * Time.deltaTime;
		}
		if (Input.GetKey(KeyCode.LeftArrow)) {
			rot.x -= rotSpeed * Time.deltaTime;
		}

		rot += new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));

		rot.y = Mathf.Clamp(rot.y, -90f, 90f);
		pos += (Quaternion.Euler(0f, rot.x, 0f) * dir) * speed * Time.deltaTime;

		mat.SetVector("_Position", new Vector4(pos.x, pos.y, pos.z, 0f));
		mat.SetVector("_Rotation", new Vector4(rot.x * Mathf.Deg2Rad, -rot.y * Mathf.Deg2Rad, 0f, 0f));
		mat.SetFloat("_AspectRatio", aspectRatio);
	}
}
