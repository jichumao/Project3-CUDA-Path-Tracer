#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
	void loadFromGltf(const std::string& gltfName, Geom& gltfMesh);
public:
    Scene(string filename);
    ~Scene();

	std::vector<Geom> geoms;
	std::vector<Material> materials;
	std::vector<Triangle> meshTris;
    std::vector<Texture> textures;
    std::vector<glm::vec3> texturesData;
    RenderState state;

    bool enable_skybox;
    Texture* skyboxTexture;

#if BVH_ENABLED
    std::vector<BVHNode> bvhNodes;
    std::vector<int> bvhTriIndices;
#endif
};
