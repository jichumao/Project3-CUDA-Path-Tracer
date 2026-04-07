#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include <cuda_runtime.h>          
#include <cuda_texture_types.h>    
#include <cuda_runtime_api.h>

#include "stb_image.h"
#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Sampling Inline Functions
// Reference https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models#fragment-Computepointonplaneoffocus-0
__host__ __device__ 
glm::vec2 SampleUniformDiskConcentric(float u1, float u2)
{
	// Map uniform random numbers to [-1, 1]^2
	glm::vec2 uOffset = 2.0f * glm::vec2(u1, u2) - glm::vec2(1.0f, 1.0f);

	// Handle degeneracy at the origin
	if (uOffset.x == 0 && uOffset.y == 0) return glm::vec2(0.0f, 0.0f);

	// Apply concentric mapping to point
	float theta, r;
	if (glm::abs(uOffset.x) > glm::abs(uOffset.y))
	{
		r = uOffset.x;
		theta = PI_OVER_FOUR * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = PI_OVER_TWO - PI_OVER_FOUR * (uOffset.x / uOffset.y);
	}
	return glm::vec2(r * glm::cos(theta), r * glm::sin(theta));
}

// Used for material sorting of thrust
struct sort_material
{
	__host__ __device__
	bool operator()(const ShadeableIntersection& si1, const ShadeableIntersection& si2)
	{
		return si1.materialId < si2.materialId;
	}
};

// Used for ray compaction
struct if_terminated
{
	__host__ __device__
	bool operator()(const PathSegment& ps)
	{
		return ps.remainingBounces != 0;
	}
};

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        pix /= iter;
#if REINHARD_TONE_MAPPING
		pix /= (pix + glm::vec3(1.0f));
#endif 
#if ACES_TONE_MAPPING
		pix = (pix * (2.51f * pix + 0.03f)) / (pix * (2.43f * pix + 0.59f) + 0.14f);
#endif 
#if GAMMA_CORRECTION
		pix = glm::pow(pix, glm::vec3(1.f / 2.2f));
#endif
        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static Triangle* dev_triangles = NULL;
static Texture* dev_textures = NULL;
static glm::vec3* dev_texturesData = NULL;
static BVHNode* dev_bvhNodes = NULL;
static int* dev_bvhTriIndices = NULL;

static cudaArray_t dev_skyboxArray = NULL;
static cudaTextureObject_t dev_skyboxTex = 0;


void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

	cudaMalloc(&dev_triangles, scene->meshTris.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->meshTris.data(), scene->meshTris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
	cudaMemcpy(dev_textures,scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_texturesData, scene->texturesData.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_texturesData, scene->texturesData.data(), scene->texturesData.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

#if BVH_ENABLED
	cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvhTriIndices, scene->bvhTriIndices.size() * sizeof(int));
	cudaMemcpy(dev_bvhTriIndices, scene->bvhTriIndices.data(), scene->bvhTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
#endif

#if ENVIRONMENT_MAP_ENABLED
	if (scene->enable_skybox) {
		// create cudaArray
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaMallocArray(&dev_skyboxArray, &channelDesc, scene->skyboxTexture->width, scene->skyboxTexture->height);

		// copy data to cudaArray
		size_t width_in_bytes = scene->skyboxTexture->width * scene->skyboxTexture->numChannels * sizeof(float);
		cudaMemcpy2DToArray(
			dev_skyboxArray,
			0, 0,
			scene->skyboxTexture->data,
			width_in_bytes,
			width_in_bytes,
			scene->skyboxTexture->height,
			cudaMemcpyHostToDevice
		);

		// create resource descriptor
		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = dev_skyboxArray;

		// create texture descriptor
		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType; // for float data
		texDesc.normalizedCoords = 1;

		// create texture object
		cudaCreateTextureObject(&dev_skyboxTex, &resDesc, &texDesc, NULL);
	}
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_triangles);
	cudaFree(dev_textures);
	cudaFree(dev_texturesData);
#if BVH_ENABLED
	cudaFree(dev_bvhNodes);
	cudaFree(dev_bvhTriIndices);
#endif
#if ENVIRONMENT_MAP_ENABLED
	if (dev_skyboxTex) {
		cudaDestroyTextureObject(dev_skyboxTex);
		dev_skyboxTex = 0;
	}
	if (dev_skyboxArray) {
		cudaFreeArray(dev_skyboxArray);
		dev_skyboxArray = NULL;
	}

	//if (hst_scene && hst_scene->skyboxTexture) { 
	//	stbi_image_free(hst_scene->skyboxTexture->data);
	//	delete hst_scene->skyboxTexture;
	//	hst_scene->skyboxTexture = nullptr;
	//}
#endif


    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.accumColor = glm::vec3(1.0f, 1.0f, 1.0f);
		
        // TODO: implement antialiasing by jittering the ray

 // Setting for generating random numbers used by AA and DOF
#if ANTI_ALIASING || DEPTH_OF_FIELD
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> uX(0, 1);
		thrust::uniform_real_distribution<float> uY(0, 1);
#endif

#if ANTI_ALIASING			
		float dx = uX(rng);
        float dy = uY(rng);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)(x + dx) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)(y + dy) - (float)cam.resolution.y * 0.5f)
		);
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)(x) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)(y) - (float)cam.resolution.y * 0.5f)
        );
#endif

 // PBR based DOF https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models#fragment-Computepointonplaneoffocus-0
#if DEPTH_OF_FIELD
		// Sample a point on the lens
		glm::vec2 pLens = cam.lensRadius * SampleUniformDiskConcentric(uX(rng), uY(rng));
		// Compute point on plane of focus
		float ft = cam.focalLength  / glm::dot(segment.ray.direction, cam.view);
		glm::vec3 pFocus = getPointOnRay(segment.ray, ft);
		// Update ray for effect of lens
        segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0);
		segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
#endif
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
	Triangle* triangles,
	BVHNode* bvhNodes,
	int* bvhTriIndices,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {   
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
		glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;
        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == MESH)
            {   
				t = meshIntersectionTest(geom, triangles, bvhNodes, bvhTriIndices, pathSegment.ray, tmp_intersect, tmp_normal,tmp_uv, outside);
            }

            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }
           // The ray hits nothing
        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            pathSegment.remainingBounces = 0;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;

			intersections[path_index].hasAlbedo = geoms[hit_geom_index].hasAlbedo;
			intersections[path_index].uv = tmp_uv;
			intersections[path_index].textureId = geoms[hit_geom_index].albedoTextureId;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    int depth,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
#if ENVIRONMENT_MAP_ENABLED
	cudaTextureObject_t skyboxTex,
#endif
    Material* materials,
	Texture* textures,
	glm::vec3* textureData)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {   
		// Stop Tracing
		if (pathSegments[idx].remainingBounces == 0)
			return;
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];

            if (intersection.hasAlbedo) {
                float texWidth = textures[intersection.textureId].width ;
				float texHeight = textures[intersection.textureId].height ;
                // clamp
                int X = glm::min(texWidth * intersection.uv.x, texWidth - 1.f);
				int Y = glm::min(texHeight * intersection.uv.y, texHeight - 1.f);

				int idx = Y * texWidth + X + textures[intersection.textureId].startIdx;
                material.color = textureData[idx];
                material.specular.color = material.color;
            }
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
				pathSegments[idx].color = (materialColor * material.emittance) * pathSegments[idx].accumColor;
				pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                pathSegments[idx].remainingBounces--;
				scatterRay(pathSegments[idx],
					getPointOnRay(pathSegments[idx].ray, intersection.t),
					intersection.surfaceNormal,
					material,
					rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
#if ENVIRONMENT_MAP_ENABLED
			// ENVIRONMENT_MAP
			glm::vec3 dir = glm::normalize(pathSegments[idx].ray.direction);

			float theta = acosf(-dir.y);
			float phi = atan2f(-dir.z, dir.x) + PI;

			float u = phi / (2.0f * PI);
			float v = 1.0f - (theta / PI); // Flip vertically

			float4 texColor = tex2D<float4>(skyboxTex, u, v);

			pathSegments[idx].color = glm::vec3(texColor.x, texColor.y, texColor.z) * pathSegments[idx].accumColor;
			pathSegments[idx].remainingBounces = 0;
#else
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
#endif          
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
            , dev_triangles
            , dev_bvhNodes
            , dev_bvhTriIndices
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
#if MATERIAL_SORT
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sort_material());
#endif
		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			depth,
			dev_intersections,
			dev_paths,
#if ENVIRONMENT_MAP_ENABLED
			dev_skyboxTex,
#endif
			dev_materials,
			dev_textures,
			dev_texturesData
			);

        cudaDeviceSynchronize();

#if STREAM_COMPACTION
		// Stream Compaction
		thrust::device_ptr<PathSegment> thrust_dev_paths(dev_paths);

		thrust::device_ptr<PathSegment> end = thrust::stable_partition(
            thrust::device,
            thrust_dev_paths,
            thrust_dev_paths + num_paths,
            if_terminated());

		num_paths = thrust::distance(thrust_dev_paths, end);
#endif

        iterationComplete = ((num_paths <= 0) || (depth == traceDepth));

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
