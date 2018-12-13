#pragma once

#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <omp.h>
#include <future>
#include <chrono>
#include <limits>


#include <GL\glew.h>
#include <GLFW\glfw3.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>



#include "Mesh.h"
#include "Body.h"
#include "Shader.h"
#include "Application.h"
#include "CudaNbody.cuh"

// Vertex Shader
static const char* vShader = "Shaders/physics.vert";

// Fragment Shader
static const char* fShader = "Shaders/physics.frag";

Cuda solver;

constexpr int BODIES = 1024*2 ;

std::vector<Body*> bodyList;
std::vector<glm::vec3> posList(BODIES);
std::vector<float> massList(BODIES);
std::vector<glm::vec3> gravs(BODIES);
std::vector<Shader*> shaderList;


using namespace std;
using namespace std::chrono;

int main()
{
	//stats file
	ofstream output;
	output.open("CUDA.csv");
	//create app
	Application app = Application::Application("NbodySim");
	app.initRender();
	app.camera.setPos(glm::vec3(0.0f, 5.0f, 1000.0f));

	// Seed with real random number if available
	random_device r1;
	// Create random number generator
	default_random_engine e1(r1());
	// Create a distribution - floats between 500.0 and 500.0
	uniform_real_distribution<float> distribution1(-500.0, 500.0);

	// Seed with real random number if available
	random_device r2;
	// Create random number generator
	default_random_engine e2(r2());
	// Create a distribution - floats between -1.0 and 1.0
	uniform_real_distribution<float> distribution2(-1.0, 1.0);

	//create bodies and load them into a vector
	for (int i = 0; i < BODIES; i++)
	{
		Body *a = new Body;
		a->setMesh(Mesh(Mesh::CUBE));
		a->getMesh().setShader(Shader(vShader, fShader));
		a->scale(glm::vec3(2.3f, 2.3f, 2.3f));
		a->setVel(glm::vec3(0.0, 0.0, 0.0));
		a->setAcc(glm::vec3(0.0, 0.0, 0.0));
		a->setPos(glm::vec3(distribution1(e1), distribution1(e1), distribution1(e1)));
		
		a->setMass(10);
		bodyList.push_back(a);
	}

	const float dt = 0.03f;
	float accumulator = 0.0f;
	GLfloat currentTime = (GLfloat)glfwGetTime();




	


	// Loop until window closed
	while (!app.getShouldClose())
	{
		//New frame time
		GLfloat newTime = (GLfloat)glfwGetTime();
		GLfloat frameTime = newTime - currentTime;

		//*******************************************************************************************************************
		frameTime *= 1;
		currentTime = newTime;
		accumulator += frameTime;

		app.showFPS();
		
		auto start = chrono::system_clock::now();
		
		//load data into GPU
		solver.loadBuffers(BODIES, bodyList, gravs, dt);
		solver.getGravities(gravs, BODIES, dt);

		auto end = chrono::system_clock::now();
		duration<double, milli> diff = end - start;
		output << diff.count() << ",";

		
		while (accumulator >= dt)
		{
			for (int i = 0; i < BODIES; i++)
			{
				//move the body
				// integration position
				bodyList[i]->setAcc(gravs[i]);
				bodyList[i]->setVel(bodyList[i]->getVel() + dt * bodyList[i]->getAcc());
				bodyList[i]->setPos(bodyList[i]->getPos() + dt * bodyList[i]->getVel());

			}
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
		for (auto &m : bodyList)
		{
			app.draw(m->getMesh());
		}
		app.display();
	}

	output.close();

	return 0;
}