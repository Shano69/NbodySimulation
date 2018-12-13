#pragma once

//GLEW
#include <GL\glew.h>

//GLFW
#include <GLFW\glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <float.h>

//Project Parts
#include "Camera.h"
#include "Shader.h"
#include "Mesh.h"


class Application
{
public:
	Application(const char *windowTitle);
	~Application();

	//data for creating window
	static const GLuint WIDTH = 800;
	static const GLuint HEIGHT = 600;
	static int SCREEN_WIDTH, SCREEN_HEIGHT;

	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	inline GLfloat getBufferWidth() { return bufferWidth; }
	inline GLfloat getBufferHeight() { return bufferHeight; }

	inline 	GLFWwindow* getWindow() { return app_window; }

	bool getShouldClose() { return glfwWindowShouldClose(app_window); }

	bool* getKeys() { return keys; }
	GLfloat getXChange();
	GLfloat getYChange();

	void swapBuffer() { glfwSwapBuffers(app_window); }

	void showFPS();
	//camera
	static Camera camera;

	//Camera methods
	void moveCamera(GLfloat deltaTime);

	//get and set methods

	Shader getShader() { return app_shader; }
	void setShader(const Shader &shader)
	{
		app_shader = shader;
		app_shader.UseShader();
	}

	// call the window render 
	int initRender();

	//additional methods
	void clear();
	void draw(const Mesh &mesh);
	void display();
	void terminate() { glfwTerminate(); };

private:
	const char *app_windowTitle;
	// view and projection matrices
	glm::mat4 app_view = glm::mat4(1.0f);
	glm::mat4 app_projection = glm::mat4(1.0f);

	Shader app_shader;
	GLFWwindow* app_window;

	GLint width, height;
	GLint bufferWidth, bufferHeight;

public:


	static bool keys[1024];

	static GLfloat lastX;
	static GLfloat lastY;
	static GLfloat xChange;
	static GLfloat yChange;
	static bool mouseFirstMoved;

	void createCallbacks();
	static void handleKeys(GLFWwindow* window, int key, int code, int action, int mode);
	static void handleMouse(GLFWwindow* window, double xPos, double yPos);
};

