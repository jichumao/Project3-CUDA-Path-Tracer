#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"

#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"
#include "stb_image.h"
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;

#if ENVIRONMENT_MAP_ENABLED
	int width, height, channels;
	int desired_channels = 4; 
    std::string fileName = "meadow_2_4k.hdr";
	std::string dir_skyboxTex = "../resources/environment_maps/" + fileName;
	float* h_image = stbi_loadf(dir_skyboxTex.c_str(), &width, &height, &channels, desired_channels);
	if (!h_image) {
		std::cerr << "Failed to load SKYBOX image!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// Assign to skyboxTexture
	skyboxTexture = new Texture();
	skyboxTexture->width = width;
	skyboxTexture->height = height;
	skyboxTexture->numChannels = desired_channels;
	skyboxTexture->type = SkyboxMap;
	skyboxTexture->data = h_image;
	enable_skybox = true;
#endif
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
#if BVH_ENABLED
		buildBVH(); // build BVH
#endif
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            const float& roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f - roughness;

			const auto& col2 = p["SPECRGB"];
			newMaterial.specular.color = glm::vec3(col2[0], col2[1], col2[2]);
            
		}
		else if (p["TYPE"] == "Refractive")
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.indexOfRefraction = p["IOR"];
			newMaterial.hasRefractive = 1.0f;

            const auto& col2 = p["SPECRGB"];
            newMaterial.specular.color = glm::vec3(col2[0], col2[1], col2[2]);
		}
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];

	uint32_t gid = 0;
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;

        newGeom.geometryid = gid++;

        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
		else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh_gltf")
        {
			newGeom.type = MESH;
			loadFromGltf(p["FILE"], newGeom);
		}
		else {
			std::cerr << "Unknown object type: " << type << std::endl;
			continue;
		}
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];

#if DEPTH_OF_FIELD
	camera.lensRadius = cameraData["LENSRADIUS"];
	camera.focalLength = cameraData["FOCALLENGTH"];
#endif

    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

// Reference https://www.slideshare.net/slideshow/gltf-20-reference-guide/78149291#1
void Scene::loadFromGltf(const std::string& gltfName, Geom& meshGeom) {

	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	std::string dir_gltf = "../resources/" + gltfName + "/glTF/" + gltfName + ".gltf";
	bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, dir_gltf);

	if (!warn.empty()) {
		std::cout << "Warn: " << warn << std::endl;
	}

	if (!err.empty()) {
		std::cerr << "Error: " << err << std::endl;
	}

    if (!ret) {
		std::cerr << "Failed to load glTF: " << dir_gltf << std::endl;
		return;
    }

    meshGeom.startTriangleIndex = meshTris.size();
	// For each mesh in the glTF file
    for (const auto& mesh : model.meshes) {
        // For each primitive in the mesh
        for (const auto& primitive : mesh.primitives) {

            const float* positions = nullptr;
            const float* normals = nullptr;
            const float* texcoords = nullptr;

            size_t vertexCount = 0;

            for (const auto& attr : primitive.attributes) {
                const tinygltf::Accessor& accessor = model.accessors[attr.second];
                const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
                const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

                if (attr.first == "POSITION") {
                    positions = reinterpret_cast<const float*>(dataPtr);
                    vertexCount = accessor.count;
                }
                else if (attr.first == "NORMAL") {
                    normals = reinterpret_cast<const float*>(dataPtr);
                    meshGeom.hasNormals = true;
                }
                else if (attr.first == "TEXCOORD_0") {
                    texcoords = reinterpret_cast<const float*>(dataPtr);
                    meshGeom.hasUVs = true;
                }
            }

			// Get the indices from the primitive
			std::vector<unsigned int> indices;
			if (primitive.indices >= 0) {
				const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
				const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
				const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
				const unsigned char* indexData = indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset;

				indices.resize(indexAccessor.count);
                // 5123
				if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
					const uint16_t* buf = reinterpret_cast<const uint16_t*>(indexData);
					for (size_t i = 0; i < indexAccessor.count; ++i) {
						indices[i] = static_cast<unsigned int>(buf[i]);
					}
				}
                // 5125
				else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
					const uint32_t* buf = reinterpret_cast<const uint32_t*>(indexData);
					for (size_t i = 0; i < indexAccessor.count; ++i) {
						indices[i] = static_cast<unsigned int>(buf[i]);
					}
				}
				else {
					std::cerr << "Unsupported Indices component" << std::endl;
					continue;
				}
			}
			else {
				// if no indices, generate them
				indices.resize(vertexCount);
				for (unsigned int i = 0; i < vertexCount; ++i) {
					indices[i] = i;
				}
			}
			// Add the texture to the material
            if (primitive.material >= 0) {
                int idx = model.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index;
				if (idx >= 0) {
					const tinygltf::Texture& texture = model.textures[idx];
					const tinygltf::Image& image = model.images[texture.source];

					Texture tex;
					tex.id = textures.size();
					tex.width = image.width;
					tex.height = image.height;
					tex.numChannels = image.component;
					tex.type = AlbedoMap;
                    tex.startIdx = texturesData.size();

					// The start index of the texture data in texturesData
					meshGeom.albedoTextureId = tex.id;
					meshGeom.hasAlbedo = true;
                    
					// Add color from image to texturesData
					for (size_t i = 0; i < image.image.size(); i += image.component) {
						glm::vec3 color;
						if (image.component == 1) {
							color = glm::vec3(image.image[i]);
						}
						else if (image.component == 4) {
							color = glm::vec3(image.image[i]/255.f, image.image[i + 1] / 255.f, image.image[i + 2] / 255.f);
						}
						else {
							std::cerr << "Unsupported number of channels in texture" << std::endl;
							continue;
						}
						texturesData.push_back(color);
					}
                    tex.endIdx = texturesData.size() - 1;
                    textures.push_back(tex);
				}
            }
			// Create triangles from the indices
			for (size_t i = 0; i + 2 < indices.size(); i += 3) {
				Triangle tri;
				auto setVertex = [&](Vertex& vert, unsigned int idx) {
					vert.position = glm::vec3(
						positions[idx * 3],
						positions[idx * 3 + 1],
						positions[idx * 3 + 2]
					);
					vert.normal = normals ? glm::vec3(
						normals[idx * 3],
						normals[idx * 3 + 1],
						normals[idx * 3 + 2]
					) : glm::vec3(0.0f);
					vert.uv = texcoords ? glm::vec2(
						texcoords[idx * 2],
						texcoords[idx * 2 + 1]
					) : glm::vec2(0.0f);
					};

				setVertex(tri.v0, indices[i]);
				setVertex(tri.v1, indices[i + 1]);
				setVertex(tri.v2, indices[i + 2]);

				meshTris.push_back(tri);
			}
        }

        meshGeom.endTriangleIndex = meshTris.size() - 1;
    }
}

#if BVH_ENABLED


void Scene::buildBVH() {
	// 收集所有三角形的包围盒
	std::vector<PrimitiveInfo> primInfos;
	primInfos.reserve(meshTris.size());
	for (int i = 0; i < meshTris.size(); ++i) {
		const Triangle& tri = meshTris[i];
		AABB bbox;
		bbox.expand(tri.v0.position);
		bbox.expand(tri.v1.position);
		bbox.expand(tri.v2.position);
		primInfos.emplace_back(i, bbox);
	}

	// 初始化扁平化数组的大小（初始估计）
	int estimatedNodes = 2 * meshTris.size(); // 保守估计
	flattenedBVH.resize(estimatedNodes);
	int totalNodes = 0;

	// 递归构建 BVH
	buildBVHRecursive(primInfos, 0, primInfos.size(), totalNodes, 8);

	// 重新调整扁平化数组的大小
	flattenedBVH.resize(totalNodes);
}

int Scene::buildBVHRecursive(
	std::vector<PrimitiveInfo>& primInfos,
	int start, int end,
	int& totalNodes, int maxLeafSize) {

	// 当前节点的索引
	int currentIdx = totalNodes++;
	if (currentIdx >= flattenedBVH.size()) {
		flattenedBVH.resize(flattenedBVH.size() * 2 + 1);
	}

	BVHNode& node = flattenedBVH[currentIdx];

	// 计算节点的包围盒
	AABB bbox;
	for (int i = start; i < end; ++i) {
		bbox.expand(primInfos[i].bbox);
	}
	node.bbox = bbox;

	int numPrimitives = end - start;
	if (numPrimitives <= maxLeafSize) {
		// 创建叶子节点
		node.isLeaf = true;
		node.start = start;
		node.range = numPrimitives;
		node.left = -1;
		node.right = -1;
	}
	else {
		// 计算质心的包围盒
		AABB centroidBBox;
		for (int i = start; i < end; ++i) {
			centroidBBox.expand(primInfos[i].centroid);
		}
		int dim = centroidBBox.maxExtent();

		// 按照质心在最大扩展轴上的坐标进行排序
		std::sort(primInfos.begin() + start, primInfos.begin() + end,
			[dim](const PrimitiveInfo& a, const PrimitiveInfo& b) {
				return a.centroid[dim] < b.centroid[dim];
			});

		int mid = (start + end) / 2;
		node.isLeaf = false;

		// 递归构建左子节点和右子节点
		node.left = buildBVHRecursive(primInfos, start, mid, totalNodes, maxLeafSize);
		node.right = buildBVHRecursive(primInfos, mid, end, totalNodes, maxLeafSize);
	}

	return currentIdx;
}

void Scene::deleteBVHRecursive(int nodeIdx) {
	if (nodeIdx == -1) return;

	BVHNode& node = flattenedBVH[nodeIdx];
	if (!node.isLeaf) {
		deleteBVHRecursive(node.left);
		deleteBVHRecursive(node.right);
	}
}


#endif