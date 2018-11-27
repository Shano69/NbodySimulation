#pragma once
#include<GL\glew.h>
#include <string>
#include<glm\glm.hpp>
#include <vector>
#include "Shader.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>


/*
** VERTEX CLASS
*/
class Vertex {
public:

	Vertex() {
		m_coord = glm::vec3();
	}

	Vertex(const glm::vec3& coord) {
		m_coord = coord;
	}

	inline bool operator<(const Vertex& other) const
	{
		return m_coord.x != other.m_coord.x ? m_coord.x < other.m_coord.x
			: m_coord.y != other.m_coord.y ? m_coord.y < other.m_coord.y
			: m_coord.z < other.m_coord.z;
	}

	glm::vec3 getCoord() const { return m_coord; }
	void setCoord(const glm::vec3& coord) { m_coord = coord; }

protected:
private:
	glm::vec3 m_coord;
};


class Mesh
{
public:

	enum MeshType
	{
		TRIANGLE,
		QUAD,
		CUBE
	};
	/*
	Constructors
	*/
	//default
	Mesh();
	//from mesh type
	Mesh(MeshType);
	//from file
	/*
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		!!!!!!!! TODO !!!!!!!!!!!!!!!!!!!!!!
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		V V V V V V V V V V V V V V V V V V
	*/
	Mesh(const std::string& fileName);


	void CreateMesh(Vertex* vertices, glm::vec3* normals);
	void RenderMesh();
	void ClearMesh();

	virtual ~Mesh();

	/*
	** GET AND SET METHODS
	*/

	// getModel computes the model matrix any time it is required
	glm::vec3 getPos() const { return getTranslate()[3]; }
	glm::mat4 getModel() const { return getTranslate() * getRotate() * getScale(); }
	glm::mat4 getTranslate() const { return m_translate; }
	glm::mat4 getRotate() const { return m_rotate; }
	glm::mat4 getScale() const { return m_scale; }
	Shader getShader() const { return m_shader; }

	// get buffers and array references
	GLuint getVertexArrayObject() const { return VAO; }
	GLuint getVertexBuffer() const { return VBO; }
	GLuint getNormalBuffer() const { return NBO; }

	// get number of vertices
	unsigned int getNumIndices() const { return m_numIndices; }

	//set initial values of all meshes
	void initTransform();

	// create vector of unique vertices (no duplicates) from vector of all mesh vertices
	void createUniqueVertices();

	std::vector<Vertex> getVertices() { return m_vertices; }

	// set position of mesh center to specified 3D position vector
	void setPos(const glm::vec3 &position) {
		m_translate[3][0] = position[0];
		m_translate[3][1] = position[1];
		m_translate[3][2] = position[2];
	}
	// set i_th coordinate of mesh center to float p (x: i=0, y: i=1, z: i=2)
	void setPos(int i, float p) { m_translate[3][i] = p; }

	// set rotation matrix
	void setRotate(const glm::mat4 &mat) { m_rotate = mat; }

	// allocate shader to mesh
	void setShader(const Shader &shader) {
		m_shader = shader;
		m_shader.UseShader();
	}

	// translate mesh by a vector
	void translate(const glm::vec3 &vect);
	// rotate mesh by a vector
	void rotate(const float &angle, const glm::vec3 &vect);
	// scale mesh by a vector
	void scale(const glm::vec3 &vect);

	GLuint VAO, VBO, NBO;
private:

	unsigned int m_numIndices;

	glm::mat4 m_translate; // translation matrix
	glm::mat4 m_rotate; // rotation matrix
	glm::mat4 m_scale; // scale matrix

	std::vector<Vertex> m_vertices; // mesh vertices (without duplication)
	Shader m_shader; // shader

};

