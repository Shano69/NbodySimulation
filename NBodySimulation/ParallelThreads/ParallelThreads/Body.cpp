#include "Body.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

Body::Body()
{
	b_mass = 1;
}


Body::~Body()
{
}




void Body::translate(const glm::vec3 & vect) {
	b_pos = b_pos + vect;
	b_mesh.translate(vect);

}

void Body::rotate(float angle, const glm::vec3 & vect) {
	b_mesh.rotate(angle, vect);

}

void Body::scale(const glm::vec3 & vect) {
	b_mesh.scale(vect);

}