#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <vector>
#include "Body.h"
#include <iostream>


void getGravities(int BODIES,
	std::vector<float> massList,
	std::vector<Body*> bodyList,
	std::vector<glm::vec3>& gravs);
