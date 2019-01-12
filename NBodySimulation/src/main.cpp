
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include <limits>

#include <GL\glew.h>
#include <GLFW\glfw3.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>


#include "Mesh.h"
#include "Body.h"
#include "Shader.h"
#include "Application.h"


// Vertex Shader
static const char* vShader = "Shaders/physics.vert";

// Fragment Shader
static const char* fShader = "Shaders/physics.frag";


constexpr int BODIES = 512;

std::vector<Body*> bodyList;
std::vector<Shader*> shaderList;
std::vector<glm::vec3> gravs(BODIES);
using namespace std;
using namespace std::chrono;

void getGravity(std::vector<Body*> bodyList, int start, int end)
{
	for (int i = start; i < end ; i++)
	{
		glm::vec3 result = glm::vec3(0.0f);

		for (auto &b : bodyList)
		{

			//calculate distance between bodies
			float deltaX = b->getPos().x - bodyList[i]->getPos().x;
			float deltaY = b->getPos().y - bodyList[i]->getPos().y;
			float deltaZ = b->getPos().z - bodyList[i]->getPos().z;

			float distance = sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);

			if (distance > 1)
			{
				//direction of the gravity force
				glm::vec3 dir = glm::normalize(glm::vec3(deltaX, deltaY, deltaZ));
				//size of the gravity force F = G * m1*m2/d^2
				result += ((float)6.67408* (bodyList[i]->getMass() * b->getMass()) / (distance * distance)) * dir;
				//std::cout << "result " << glm::to_string(result) << std::endl;
			}
		}
		gravs[i] = result;
	}
	
}

int main() 
{
	//create file for statistics
	ofstream output;
	output.open("Nbody512.csv");

	//create app
	Application app = Application::Application("NbodySim");
	app.initRender();
	app.camera.setPos(glm::vec3(0.0f, 5.0f, 1000.0f));

	// Seed with real random number if available
	random_device r1;
	// Create random number generator
	default_random_engine e1(r1());
	// Create a distribution - we want doubles between 500.0 and 500.0
	uniform_real_distribution<float> distribution1(-500.0, 500.0);

	// Seed with real random number if available
	random_device r2;
	// Create random number generator
	default_random_engine e2(r2());
	// Create a distribution - we want doubles between -1.0 and 1.0
	uniform_real_distribution<float> distribution2(-1.0, 1.0);


	for (int i = 0; i < BODIES; i++)
	{
		Body *a = new Body;
		a->setMesh(Mesh(Mesh::CUBE));
		a->getMesh().setShader(Shader(vShader, fShader));
		a->scale(glm::vec3(2.3f, 2.3f, 2.3f));
		a->setPos(glm::vec3( distribution1(e1), distribution1(e1), distribution1(e1)));
		//a->setVel(glm::vec3(distribution2(e2), distribution2(e2), 0.0f));
		a->setMass(10);

		bodyList.push_back(a);
	}

		
	const float dt = 0.03f;
	float accumulator = 0.0f;
	GLfloat currentTime = (GLfloat)glfwGetTime();

	int i = 0;

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

		
		
		while (accumulator >= dt)
		{	
			//sequential
			auto start = chrono::system_clock::now();
			getGravity(bodyList, 0, BODIES);
			auto end = chrono::system_clock::now();
			duration<double, milli> diff = end - start;
			if(i<100)
			{
				output << diff.count() << ",";
				i++;
			}
			

			for (int i=0;i<BODIES;i++)
			{
				//move the body
				// integration position
				bodyList[i]->setAcc(gravs[i]);
				bodyList[i]->setVel(bodyList[i]->getVel() + dt * bodyList[i]->getAcc());
				bodyList[i]->setPos(bodyList[i]->getPos() + dt * (bodyList[i]->getVel()));
				
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

