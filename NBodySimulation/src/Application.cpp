#include <memory>
#include <functional>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glm/ext.hpp"

#include "Application.h"


Camera Application::camera = Camera::Camera(glm::vec3(0.0f, 0.0f, 0.0f));

GLfloat Application::lastX = WIDTH / 2.0;
GLfloat Application::lastY = HEIGHT / 2.0;
GLfloat Application::xChange = 0.0f;
GLfloat Application::yChange = 0.0f;

bool Application::mouseFirstMoved = true;
bool Application::keys[1024];


Application::Application()
{
	app_shader = Shader();
}


Application::~Application()
{
}

GLfloat Application::getXChange()
{
	GLfloat theChange = xChange;
	xChange = 0.0f;
	return theChange;
}

GLfloat Application::getYChange()
{
	GLfloat theChange = yChange;
	yChange = 0.0f;
	return theChange;
}

void Application::moveCamera(GLfloat deltaTime)
{
	camera.keyControl(getKeys(), deltaTime);
	camera.mouseControl(getXChange(), deltaTime);
}

int Application::initRender()
{
	// Initialise GLFW
	if (!glfwInit())
	{
		printf("GLFW initialisation failed!");
		glfwTerminate();
		return 1;
	}

	// Setup GLFW window properties
	// OpenGL version
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// Core Profile
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// Allow Forward Compatbility
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	// Create the window
	app_window = glfwCreateWindow(WIDTH, HEIGHT, "Test Window", NULL, NULL);
	if (!app_window)
	{
		printf("GLFW window creation failed!");
		glfwTerminate();
		return 1;
	}

	// Get Buffer Size information

	glfwGetFramebufferSize(app_window, &bufferWidth, &bufferHeight);

	// Set context for GLEW to use
	glfwMakeContextCurrent(app_window);

	// Handle key + mouse input
	createCallbacks();
	//remove cursor
	glfwSetInputMode(app_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Allow modern extension features
	glewExperimental = GL_TRUE;

	if (glewInit() != GLEW_OK)
	{
		printf("GLEW initialisation failed!");
		glfwDestroyWindow(app_window);
		glfwTerminate();
		return 1;
	}

	glEnable(GL_DEPTH_TEST);

	// Setup Viewport size
	glViewport(0, 0, bufferWidth, bufferHeight);

	glEnable(GL_DEPTH_TEST);
	// Setup some OpenGL options
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	return 1;
}

void Application::clear()
{
	// Clear window
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Application::draw(const Mesh & mesh)
{
	app_view = camera.calculateViewMatrix();
	app_projection = glm::perspective(45.0f, (GLfloat)getBufferWidth() / getBufferHeight(), 0.1f, 100.0f);
	mesh.getShader().UseShader();
	// view and projection matrices

	// Get the uniform locations for MVP
	GLint modelLoc = glGetUniformLocation(mesh.getShader().shaderID, "model");
	GLint viewLoc = glGetUniformLocation(mesh.getShader().shaderID, "view");
	GLint projLoc = glGetUniformLocation(mesh.getShader().shaderID, "projection");
	GLint rotateLoc = glGetUniformLocation(mesh.getShader().shaderID, "rotate");


	// get the uniform locations for lighing
	GLint ambientLoc = glGetUniformLocation(mesh.getShader().shaderID, "ambient");
	GLint eyePositionLoc = glGetUniformLocation(mesh.getShader().shaderID, "eyePosition");

	// Pass the matrices to the shader
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(mesh.getModel()));
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(app_view));
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(app_projection));
	glUniformMatrix4fv(rotateLoc, 1, GL_FALSE, glm::value_ptr(mesh.getRotate()));


	// pass lighting data to shader
	glUniform4fv(ambientLoc, 1, glm::value_ptr(glm::vec4(0.0, 1.0, 1.0, 1.0)));
	glUniform3fv(eyePositionLoc, 1, glm::value_ptr(Application::camera.getPos()));

	glBindVertexArray(mesh.getVertexArrayObject());
	glDrawArrays(GL_TRIANGLES, 0, mesh.getNumIndices());
	glBindVertexArray(0);
}

void Application::display()
{
	glBindVertexArray(0);
	// Swap the buffers
	glfwSwapBuffers(app_window);
}

void Application::createCallbacks()
{
	glfwSetKeyCallback(app_window, handleKeys);
	glfwSetCursorPosCallback(app_window, handleMouse);
}

void Application::handleKeys(GLFWwindow * window, int key, int code, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

	if (key >= 0 && key < 1024)
	{
		if (action == GLFW_PRESS)
		{
			keys[key] = true;
		}
		else if (action == GLFW_RELEASE)
		{
			keys[key] = false;
		}
	}
}

void Application::handleMouse(GLFWwindow * window, double xPos, double yPos)
{
	if (Application::mouseFirstMoved)
	{
		Application::lastX = xPos;
		Application::lastY = yPos;
		Application::mouseFirstMoved = false;
	}

	double xOffset = xPos - Application::lastX;
	double yOffset = Application::lastY - yPos;

	Application::lastX = xPos;
	Application::lastY = yPos;

	Application::camera.mouseControl((GLfloat)xOffset, (GLfloat)yOffset);


}

