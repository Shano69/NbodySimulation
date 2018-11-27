#pragma once

#include "Mesh.h"
#include<vector>

class Body
{
protected:
	Mesh b_mesh;
	float b_mass;
	float b_cor;

	glm::vec3 b_acc;
	glm::vec3 b_vel;
	glm::vec3 b_pos;

public:
	Body();
	~Body();

	Mesh &getMesh() { return b_mesh; }

	// transform matrices
	glm::mat4 getTranslate() const { return b_mesh.getTranslate(); }
	glm::mat4 getRotate() const { return b_mesh.getRotate(); }
	glm::mat4 getScale() const { return b_mesh.getScale(); }

	// dynamic variables
	glm::vec3 & getAcc() { return b_acc; }
	glm::vec3 & getVel() { return b_vel; }
	glm::vec3 & getPos() { return b_pos; }


	// physical properties
	float getMass() const { return b_mass; }

	//mesh
	void setMesh(Mesh m) { b_mesh = m; }


	// variables for movement
	void setAcc(const glm::vec3 & vect) { b_acc = vect; }
	void setVel(const glm::vec3 & vect) { b_vel = vect; }
	void setVel(int i, float v) { b_vel[i] = v; } // set the ith coordinate of the velocity vector
	void setPos(const glm::vec3 & vect) { b_pos = vect; b_mesh.setPos(vect); }
	void setPos(int i, float p) { b_pos[i] = p; b_mesh.setPos(i, p); } // set the ith coordinate of the position vector


	  // mass 
	void setMass(float mass) { b_mass = mass; }


	// transforms
	void translate(const glm::vec3 & vect);
	void rotate(float angle, const glm::vec3 & vect);
	void scale(const glm::vec3 & vect);
};