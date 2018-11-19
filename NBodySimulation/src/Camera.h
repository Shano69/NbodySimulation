#pragma once

#include <GL\glew.h>
#include <glm\glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm\gtc\matrix_transform.hpp>

#include<GLFW\glfw3.h>

class Camera
{
public:
	Camera();
	Camera(glm::vec3 startPosition = 
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3 startUp = glm::vec3(0.0f, 1.0f, 0.0f),
		GLfloat startYaw = -90.0f,
		GLfloat startPitch = 0.0f,
		GLfloat startMoveSpeed = 500.0f,
		GLfloat StartTurnSpeed = 0.3f);
	void keyControl(bool* keys, GLfloat deltaTime);
	void mouseControl(GLfloat xChange, GLfloat yChange);

	glm::mat4 calculateViewMatrix();

	glm::vec3 getPos() { return position; }
	void setPos(glm::vec3 pos) { position = pos; }

	~Camera();

private:
	glm::vec3 position;
	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec3 worldUp;

	GLfloat yaw;
	GLfloat pitch;

	GLfloat moveSpeed;
	GLfloat turnSpeed;

	void update();
};

