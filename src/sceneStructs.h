#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
	MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};

struct Triangle {
	Vertex v0;
	Vertex v1;
	Vertex v2;
};

struct AABB {
	glm::vec3 min;
	glm::vec3 max;
};

struct Geom
{
    enum GeomType type;
	int geometryid;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
	int startTriangleIndex;
	int endTriangleIndex;
	// Has albedo map
	bool hasAlbedo = false;
	// Has normal data
	bool hasNormals = false;
	// Has uv data
	bool hasUVs = false;
	// Index
	int albedoTextureId = -1;
	int bvhRootNodeIdx = -1;
#if BOUNDING_VOLUME_INTERSECTION_CULLING_ENABLED
	// for boundary culling
	glm::vec3 min;
	glm::vec3 max;
#endif
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
	// flag
	bool hasAlbedoTexture = false;
	bool hasNormalTexture = false;
	bool hasRoughnessTexture = false;
	bool hasMetalnessTexture = false;
	bool hasSpecularTexture = false;

	// Texture IDs
	int albedoTextureId = -1;
	int normalTextureId = -1;
	int roughnessTextureId = -1;
	int metalnessTextureId = -1;
	int specularTextureId = -1;

};

enum TextureType {
	AlbedoMap,      
	NormalMap,
	BumpMap,
	RoughnessMap,   
	MetalnessMap,   
	SpecularMap,
	SkyboxMap
};

struct Texture {
	// incremental: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
	int id;
	TextureType type;
	int width;
	int height;
	int numChannels;
	// start index of the texture data in texturesData
	int startIdx;
	int endIdx;
	// for HDR image
	float* data;
};


struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
	// For depth of field. Credits for CIS 4610 Lecture for Visual Effects 
#if DEPTH_OF_FIELD
	float lensRadius;
	float focalLength;
#endif
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
	// Added for accumulated color
	glm::vec3 accumColor;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  bool hasAlbedo = false;
  glm::vec2 uv;
  int textureId = -1;
 
};

#if BVH_ENABLED
struct BVHNode {
	AABB bounds;
	int leftChild;
	int rightChild;
	int firstTriIndex;
	int triCount;

	__host__ __device__ BVHNode()
		: leftChild(-1), rightChild(-1), firstTriIndex(-1), triCount(0) {}
};

#endif
