#pragma once

#include "CudaNbody.cuh"

using namespace std;



__global__ void getGrav(float3 *cu_pos, float3 *cu_gravs, float *cu_mass)
{
	

	// Get block index
	unsigned int block_idx = blockIdx.x;
	// Get thread index
	unsigned int thread_idx = threadIdx.x;
	// Get the number of threads per block
	unsigned int block_dim = blockDim.x;
	// Get the thread's unique ID - (block_idx * block_dim) + thread_idx;
	unsigned int idx = (block_idx * block_dim) + thread_idx;

	float3 result = { 0.0f,0.0f,0.0f };

	for (int i = 0; i< 1024 * 2; i++)
	{
		//calculate distance between bodies
		float deltaX = cu_pos[idx].x - cu_pos[i].x;
		//printf("deltaX -> %f", deltaX);
		float deltaY = cu_pos[idx].y - cu_pos[i].y;
		//printf("deltaY -> %f", deltaY);
		float deltaZ = cu_pos[idx].z - cu_pos[i].z;
		//printf("deltaZ -> %f", deltaZ);

		//printf("Value of cu_pos in x dir -> %f, Value of cu_pos in y dir -> %f, Value of cu_pos in z dir -> %f", cu_pos[idx].x, cu_pos[idx].y, cu_pos[idx].z, "\n");

		float distance = sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);

		//printf("Value of distance %f \n", distance);

		if (distance > 0.1f)
		{
			//direction of the gravity force
			float3 dir = {- deltaX / distance, -deltaY / distance, -deltaZ / distance };
			//size of the gravity force F = G * m1*m2/d^2
			result.x += ((float)6.67408* (cu_mass[idx] * cu_mass[i]) / (distance * distance)) * dir.x;
			result.y += ((float)6.67408* (cu_mass[idx] * cu_mass[i]) / (distance * distance)) * dir.y;
			result.z += ((float)6.67408* (cu_mass[idx] * cu_mass[i]) / (distance * distance)) * dir.z;
		
			//printf("Value of result in x dir -> %f, Value of result in y dir -> %f, Value of result in z dir -> %f", result.x, result.y, result.z, "\n");
			//printf("distance -> %f", distance);
		}

		
	}

	//printf("Value of grav in x dir -> %f, Value of grav in y dir -> %f, Value of grav in z dir -> %f", result.x, result.y, result.z ,"\n");


	cu_gravs[idx] = result;
	
	
	
}

void getGravities(int BODIES, std::vector<float> massList, std::vector<Body*> bodyList, std::vector<glm::vec3>& gravs)
{
	
	//Prepare data for GPU

			//data size
	auto posList_size = sizeof(float3) * BODIES;
	auto grav_size = sizeof(float3) * BODIES;
	auto massList_size = sizeof(float) * BODIES;

	//buffers
	float *massBuffer;
	float3 posLis[1024 * 2];
	float3 *posBuf;
	float3 gravList[1024 * 2];
	float3 *gravBuf;

	//populate position list nad grav list
	for (int i = 0; i < BODIES; i++)
	{
		gravList[i] = { bodyList[i]->getAcc().x,bodyList[i]->getAcc().y,bodyList[i]->getAcc().z };
		posLis[i] = { bodyList[i]->getPos().x,bodyList[i]->getPos().y,bodyList[i]->getPos().z };
	}

	//allocate memory
	cudaMalloc((void**)&posBuf, posList_size);
	cudaMalloc((void**)&gravBuf, grav_size);
	cudaMalloc((void**)&massBuffer, massList_size);


	//copy data to GPU
	cudaMemcpy(posBuf, &posLis[0], posList_size, cudaMemcpyHostToDevice);
	cudaMemcpy(massBuffer, &massList[0], massList_size, cudaMemcpyHostToDevice);

	//call the kernel
	getGrav <<<BODIES/1024,1024>>> (posBuf, gravBuf, massBuffer);

	// Wait for kernel to complete
	cudaDeviceSynchronize();

	//get the data back from the buffer 
	auto temp = cudaMemcpy(&gravList[0], gravBuf, grav_size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < BODIES; i++)
	{
		gravs[i].x = gravList[i].x;
		gravs[i].y = gravList[i].y;
		gravs[i].z = gravList[i].z;
	}
	cudaFree(posBuf);
	cudaFree(gravBuf);
	cudaFree(massBuffer);

	
}