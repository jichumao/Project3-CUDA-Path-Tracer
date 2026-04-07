#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f
// Some other useful constants from CIS 5610
#define FOUR_PI          12.5663706143591729538505735331180115367883f
#define INV_PI           0.3183098861837906715377675267450287240689f
#define INV_TWO_PI       0.1591549430918953357688837633725143620344f
#define INV_FOUR_PI      0.0795774715459476728844418816862571810172f
#define PI_OVER_TWO      1.57079632679489662f
#define PI_OVER_FOUR     0.78539816339744831f
#define ONE_THIRD        0.3333333333333333333333333333333333333333f
#define OneMinusEpsilon  0.99999f
// Switches
#define STREAM_COMPACTION       1
#define MATERIAL_SORT           0
#define ANTI_ALIASING           1
#define DEPTH_OF_FIELD          0
#define GAMMA_CORRECTION        1
#define BVH_ENABLED             1
#define ENVIRONMENT_MAP_ENABLED 1
#define BOUNDING_VOLUME_INTERSECTION_CULLING_ENABLED 0
// Tone mapping
#define REINHARD_TONE_MAPPING   0 && !ACES_TONE_MAPPING
#define ACES_TONE_MAPPING       1 && !REINHARD_TONE_MAPPING


class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
