#include "Mesh.h"

Mesh::Mesh()
{
	// Create triangle vertices
	Vertex vertices[] = { Vertex(glm::vec3(-1.0,-1.0,0.0)),
		Vertex(glm::vec3(0, 1.0, 0.0)),
		Vertex(glm::vec3(1.0, -1.0, 0.0))


	};

	// tirangle normals
	glm::vec3 normals[] = { glm::vec3(.0f, .0f, 1.0f), glm::vec3(.0f, .0f, 1.0f), glm::vec3(.0f, .0f, 1.0f) };

	// create vertex vector without duplicates (easy for a triangle)
	m_vertices = std::vector<Vertex>(std::begin(vertices), std::end(vertices));

	// number of vertices
	m_numIndices = 3;

	//create mesh
	CreateMesh(vertices, normals);

}

Mesh::Mesh(MeshType type)
{
	Vertex vertices[36];
	glm::vec3 normals[36];

	switch (type)
	{
	case TRIANGLE:
		// Create triangle
		vertices[0] = Vertex(glm::vec3(-1.0, -1.0, 0.0));
		vertices[1] = Vertex(glm::vec3(0, 1.0, 0.0));
		vertices[2] = Vertex(glm::vec3(1.0, -1.0, 0.0));
		normals[0] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[1] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[2] = glm::vec3(0.0f, 1.0f, 0.0f);

		// number of vertices
		m_numIndices = 3;

		break;

	case QUAD:
		// create quad vertices
		vertices[0] = Vertex(glm::vec3(-1.0f, 0.0f, -1.0f));
		vertices[1] = Vertex(glm::vec3(1.0f, 0.0f, -1.0f));
		vertices[2] = Vertex(glm::vec3(-1.0f, 0.0f, 1.0f));
		vertices[3] = Vertex(glm::vec3(1.0f, 0.0f, -1.0f));
		vertices[4] = Vertex(glm::vec3(-1.0f, 0.0f, 1.0f));
		vertices[5] = Vertex(glm::vec3(1.0f, 0.0f, 1.0f));

		// create normals
		normals[0] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[1] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[2] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[3] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[4] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[5] = glm::vec3(0.0f, 1.0f, 0.0f);

		// number of vertices
		m_numIndices = 6;

		break;

	case CUBE:
		// create cube
		vertices[0] = Vertex(glm::vec3(-1.0f, -1.0f, -1.0f));
		vertices[1] = Vertex(glm::vec3(1.0f, -1.0f, -1.0f));
		vertices[2] = Vertex(glm::vec3(1.0f, 1.0f, -1.0f));
		vertices[3] = Vertex(glm::vec3(-1.0f, -1.0f, -1.0f));
		vertices[4] = Vertex(glm::vec3(1.0f, 1.0f, -1.0f));
		vertices[5] = Vertex(glm::vec3(-1.0f, 1.0f, -1.0f));
		vertices[6] = Vertex(glm::vec3(-1.0f, -1.0f, 1.0f));
		vertices[7] = Vertex(glm::vec3(1.0f, -1.0f, 1.0f));
		vertices[8] = Vertex(glm::vec3(1.0f, 1.0f, 1.0f));
		vertices[9] = Vertex(glm::vec3(-1.0f, -1.0f, 1.0f));
		vertices[10] = Vertex(glm::vec3(1.0f, 1.0f, 1.0f));
		vertices[11] = Vertex(glm::vec3(-1.0f, 1.0f, 1.0f));
		vertices[12] = Vertex(glm::vec3(-1.0f, -1.0f, -1.0f));
		vertices[13] = Vertex(glm::vec3(1.0f, -1.0f, -1.0f));
		vertices[14] = Vertex(glm::vec3(1.0f, -1.0f, 1.0f));
		vertices[15] = Vertex(glm::vec3(-1.0f, -1.0f, -1.0f));
		vertices[16] = Vertex(glm::vec3(1.0f, -1.0f, 1.0f));
		vertices[17] = Vertex(glm::vec3(-1.0f, -1.0f, 1.0f));
		vertices[18] = Vertex(glm::vec3(-1.0f, 1.0f, -1.0f));
		vertices[19] = Vertex(glm::vec3(1.0f, 1.0f, -1.0f));
		vertices[20] = Vertex(glm::vec3(1.0f, 1.0f, 1.0f));
		vertices[21] = Vertex(glm::vec3(-1.0f, 1.0f, -1.0f));
		vertices[22] = Vertex(glm::vec3(1.0f, 1.0f, 1.0f));
		vertices[23] = Vertex(glm::vec3(-1.0f, 1.0f, 1.0f));
		vertices[24] = Vertex(glm::vec3(-1.0f, -1.0f, -1.0f));
		vertices[25] = Vertex(glm::vec3(-1.0f, 1.0f, -1.0f));
		vertices[26] = Vertex(glm::vec3(-1.0f, 1.0f, 1.0f));
		vertices[27] = Vertex(glm::vec3(-1.0f, -1.0f, -1.0f));
		vertices[28] = Vertex(glm::vec3(-1.0f, 1.0f, 1.0f));
		vertices[29] = Vertex(glm::vec3(-1.0f, -1.0f, 1.0f));
		vertices[30] = Vertex(glm::vec3(1.0f, -1.0f, -1.0f));
		vertices[31] = Vertex(glm::vec3(1.0f, 1.0f, -1.0f));
		vertices[32] = Vertex(glm::vec3(1.0f, 1.0f, 1.0f));
		vertices[33] = Vertex(glm::vec3(1.0f, -1.0f, -1.0f));
		vertices[34] = Vertex(glm::vec3(1.0f, 1.0f, 1.0f));
		vertices[35] = Vertex(glm::vec3(1.0f, -1.0f, 1.0f));

		//normals
		normals[0] = glm::vec3(0.0f, 0.0f, -1.0f);
		normals[1] = glm::vec3(0.0f, 0.0f, -1.0f);
		normals[2] = glm::vec3(0.0f, 0.0f, -1.0f);
		normals[3] = glm::vec3(0.0f, 0.0f, -1.0f);
		normals[4] = glm::vec3(0.0f, 0.0f, -1.0f);
		normals[5] = glm::vec3(0.0f, 0.0f, -1.0f);
		normals[6] = glm::vec3(0.0f, 0.0f, 1.0f);
		normals[7] = glm::vec3(0.0f, 0.0f, 1.0f);
		normals[8] = glm::vec3(0.0f, 0.0f, 1.0f);
		normals[9] = glm::vec3(0.0f, 0.0f, 1.0f);
		normals[10] = glm::vec3(0.0f, 0.0f, 1.0f);
		normals[11] = glm::vec3(0.0f, 0.0f, 1.0f);
		normals[12] = glm::vec3(0.0f, -1.0f, 0.0f);
		normals[13] = glm::vec3(0.0f, -1.0f, 0.0f);
		normals[14] = glm::vec3(0.0f, -1.0f, 0.0f);
		normals[15] = glm::vec3(0.0f, -1.0f, 0.0f);
		normals[16] = glm::vec3(0.0f, -1.0f, 0.0f);
		normals[17] = glm::vec3(0.0f, -1.0f, 0.0f);
		normals[18] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[19] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[20] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[21] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[22] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[23] = glm::vec3(0.0f, 1.0f, 0.0f);
		normals[24] = glm::vec3(-1.0f, 0.0f, 0.0f);
		normals[25] = glm::vec3(-1.0f, 0.0f, 0.0f);
		normals[26] = glm::vec3(-1.0f, 0.0f, 0.0f);
		normals[27] = glm::vec3(-1.0f, 0.0f, 0.0f);
		normals[28] = glm::vec3(-1.0f, 0.0f, 0.0f);
		normals[29] = glm::vec3(-1.0f, 0.0f, 0.0f);
		normals[30] = glm::vec3(1.0f, 0.0f, 0.0f);
		normals[31] = glm::vec3(1.0f, 0.0f, 0.0f);
		normals[32] = glm::vec3(1.0f, 0.0f, 0.0f);
		normals[33] = glm::vec3(1.0f, 0.0f, 0.0f);
		normals[34] = glm::vec3(1.0f, 0.0f, 0.0f);
		normals[35] = glm::vec3(1.0f, 0.0f, 0.0f);

		// number of vertices
		m_numIndices = 36;

		break;
	}



	// generate unique vertex vector (no duplicates)
	m_vertices = std::vector<Vertex>(std::begin(vertices), std::end(vertices));
	createUniqueVertices();

	//create mesh
	CreateMesh(vertices, normals);

	// create model matrix (identity)
	initTransform();
}


void Mesh::CreateMesh(Vertex* vertices, glm::vec3* normals)
{

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	// vertex buffer
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, m_numIndices * sizeof(vertices[0]), &vertices[0], GL_STATIC_DRAW);

	// normal buffer
	glGenBuffers(1, &NBO);
	glBindBuffer(GL_ARRAY_BUFFER, NBO);
	glBufferData(GL_ARRAY_BUFFER, m_numIndices * sizeof(vertices[0]), &normals[0], GL_STATIC_DRAW);

	// vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	// normals
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, NBO);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindVertexArray(0);
}

void Mesh::RenderMesh()
{

}

void Mesh::ClearMesh()
{
	if (NBO != 0)
	{
		glDeleteBuffers(1, &NBO);
		NBO = 0;
	}

	if (VBO != 0)
	{
		glDeleteBuffers(1, &VBO);
		VBO = 0;
	}

	if (VAO != 0)
	{
		glDeleteVertexArrays(1, &VAO);
		VAO = 0;
	}

	m_numIndices = 0;
}


Mesh::~Mesh()
{
	//ClearMesh();
}

// set initial values so no nulls
void Mesh::initTransform()
{
	m_translate = glm::mat4(1.0f);
	m_rotate = glm::mat4(1.0f);
	m_scale = glm::mat4(1.0f);
}


void Mesh::createUniqueVertices()
{

	std::vector<Vertex> temp_vertices = m_vertices;

	unsigned int i = 0;
	while (i < m_vertices.size()) {
		unsigned int j = i + 1;
		bool duplicateFound = false;
		while (j < m_vertices.size() && !duplicateFound) {
			if (m_vertices.at(i).getCoord() == m_vertices.at(j).getCoord()) {
				duplicateFound = true;
			}
			j++;
		}
		if (duplicateFound) {
			m_vertices.erase(m_vertices.begin() + i);
		}
		else {
			++i;
		}
	}


}

// translate
void Mesh::translate(const glm::vec3 &vect) {
	m_translate = glm::translate(m_translate, vect);
}

// rotate
void Mesh::rotate(const float &angle, const glm::vec3 &vect) {
	m_rotate = glm::rotate(m_rotate, angle, vect);
}

// scale
void Mesh::scale(const glm::vec3 &vect) {
	m_scale = glm::scale(m_scale, vect);
}

