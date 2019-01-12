#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <vector>
#include "Body.h"
#include <iostream>

class Cuda
{
public:
	Cuda() {};
	~Cuda() {}; 
	void getGravities(std::vector<glm::vec3>& gravs, int BODIES);

	void loadBuffers(int BODIES, std::vector<Body*> bodyList, std::vector<glm::vec3>& gravs);

private:
	float4 *positionBuff;
	float4 *gravityBuff;
	float4 *velocityBuf;
	float *dtBuf;
};

