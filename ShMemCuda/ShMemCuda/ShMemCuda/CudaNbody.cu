#pragma once

#include "CudaNbody.cuh"

using namespace std;



__global__ void getGrav(float4*cu_pos, float4 *cu_gravs)
{
	__shared__ float4 sharedPos[1024 * 2];
	//the pos of the current thread
	float4 myPosition;

	int i;
	
	//result to be copied back to host
	float4 result = { 0.0f, 0.0f, 0.0f, 0.0f };

	//thread id
	int gtid;
	gtid = ( blockIdx.x * blockDim.x ) + threadIdx.x;


	myPosition = cu_pos[gtid];
	sharedPos[gtid] = cu_pos[gtid];

	for (i = 0; i < 1024 * 2; i ++)
	{
		
			//body body interaction part
			float3 delta;

			delta.x = sharedPos[i].x - myPosition.x;
			delta.y = sharedPos[i].y - myPosition.y;
			delta.z = sharedPos[i].z - myPosition.z;

			float distance = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z) ;

			if (distance > 0.1f)
			{
				//direction of the gravity force
				float4 dir = { delta.x / distance, delta.y / distance, delta.z / distance, 0.0f };
				//size of the gravity force F = G * m1*m2/d^2
				result.x += ((float)6.67408* (sharedPos[i].w * myPosition.w) / (distance * distance)) * dir.x;
				result.y += ((float)6.67408* (sharedPos[i].w * myPosition.w) / (distance * distance)) * dir.y;
				result.z += ((float)6.67408* (sharedPos[i].w * myPosition.w) / (distance * distance)) * dir.z;
				result.w = 0.0f;
				//printf("distance -> %f", distance);
				__syncthreads();
			}

	}
	cu_gravs[gtid] = result;

	
}

void Cuda::getGravities( std::vector<glm::vec3>& gravs, int BODIES, float dt)
{
	auto grav_size = sizeof(float4) * BODIES;

	float4 gravList[1024 * 2];

	//call the kernel
	getGrav <<<BODIES / 512, 512 >>> (positionBuff, gravityBuff);

	// Wait for kernel to complete check for errors
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

	//get the data back from the buffer 
	auto temp = cudaMemcpy(&gravList[0], gravityBuff, grav_size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < BODIES; i++)
	{
		gravs[i].x = gravList[i].x;
		gravs[i].y = gravList[i].y;
		gravs[i].z = gravList[i].z;
	}

	cudaFree(positionBuff);
	cudaFree(gravityBuff);
	cudaFree(velocityBuf);
	cudaFree(dtBuf);

	
}


void  Cuda::loadBuffers(int BODIES, std::vector<Body*> bodyList, std::vector<glm::vec3>& gravs, float dt)
{
	//data size
	auto posList_size = sizeof(float4) * BODIES;
	auto grav_size = sizeof(float4) * BODIES;
	auto vel_size = sizeof(float4) * BODIES;
	auto dt_size = sizeof(float);

	//lists
	float4 posLis[1024 * 2];
	float4 gravList[1024 * 2];
	float4 velList[1024 * 2];


	//Prepare data for GPU
	


	//populate position list nad grav list
	for (int i = 0; i < BODIES; i++)
	{
		posLis[i] = { bodyList[i]->getPos().x,bodyList[i]->getPos().y,bodyList[i]->getPos().z, bodyList[i]->getMass() };	
	}

	//allocate memory
	cudaMalloc((void**)&positionBuff, posList_size);
	cudaMalloc((void**)&gravityBuff, grav_size);
	

	
	//copy data to GPU
	cudaError_t s1 = cudaMemcpy(positionBuff, &posLis[0], posList_size, cudaMemcpyHostToDevice);
	if (s1 != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(s1));

	cudaError_t s2 = cudaMemcpy(gravityBuff, &gravList[0], grav_size, cudaMemcpyHostToDevice);
	if (s2 != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(s2));
	
	

}
