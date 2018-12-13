#pragma once

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>


#include <GL\glew.h>

class Shader
{
public:

	GLuint Program;

	Shader();
	Shader(const char * vertexLocation, const char * fragmentLocation);

	void CreateFromString(const char* vertexCode, const char* fragmentCode);
	void CreateFromFiles(const char* vertexLocation, const char* fragmentLocation);

	std::string ReadFile(const char* fileLocation);

	GLuint GetProjectionLocation();
	GLuint GetModelLocation();
	GLuint GetViewLocation();

	void UseShader();
	~Shader();

public:
	GLuint shaderID;

private:
	GLuint uniformProjection, uniformModel, uniformView;
	void CompileShader(const char* vertexCode, const char* fragmentCode);
	void AddShader(GLuint theProgram, const char* shaderCode, GLenum shaderType);
};

