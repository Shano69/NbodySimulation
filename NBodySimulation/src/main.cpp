#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>

#include <GL\glew.h>
#include <GLFW\glfw3.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>


#include "Mesh.h"
#include "Shader.h"
#include "Application.h"


// Vertex Shader
static const char* vShader = "Shaders/physics.vert";

// Fragment Shader
static const char* fShader = "Shaders/physics.frag";


std::vector<Mesh*> meshList;
std::vector<Shader*> shaderList;

using namespace std;

int main() 
{

	//create app
	Application app = Application::Application();
	app.initRender();
	app.camera.setPos(glm::vec3(0.0f, 5.0f, 20.0f));

	Mesh plane(Mesh::CUBE);
	plane.setShader(Shader(vShader, fShader));
	plane.scale(glm::vec3(20.0f, 1.0f, 20.0f));
	plane.translate(glm::vec3(0.0f, 0.0f, 0.0f));

	meshList.push_back(&plane);

	const float dt = 0.003f;
	float accumulator = 0.0f;
	GLfloat currentTime = (GLfloat)glfwGetTime();


		// Loop until window closed
	while (!app.getShouldClose())
	{
		//New frame time
		GLfloat newTime = (GLfloat)glfwGetTime();
		GLfloat frameTime = newTime - currentTime;

		//*******************************************************************************************************************
		frameTime *= 2;
		currentTime = newTime;
		accumulator += frameTime;

		while (accumulator >= dt)
		{
			accumulator -= dt;
		}

		// Get + Handle user input events
		glfwPollEvents();

		//key handler
		app.camera.keyControl(app.getKeys(), dt);
		
		//moyse handler
		app.camera.mouseControl(app.getXChange(), app.getYChange());

		app.clear();

		//draw the bodies from the list
		for (auto &m : meshList)
		{
			app.draw(*m);
		}
		app.display();
	}

	return 0;
}