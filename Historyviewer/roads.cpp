#define WINVER 0x0501
#define _WIN32_WINNT 0x0501
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <GL/gl.h>

#include "roads.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace roads {

static std::vector<GLuint> textures;
static std::map<std::string, std::string> tileFiles;

static double maximumDistance = 1200.0;
static volatile LONG failedLoads = 0;

bool enabled = true;
unsigned long drawnRoads = 0;
unsigned long drawnTriangles = 0;


// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

static bool readBytes(
    std::ifstream& file,
    void* target,
    std::size_t bytes
) {
    file.read((char*)target, (std::streamsize)bytes);
    return !!file;
}

static bool readLine(std::ifstream& file, std::string& line) {
    if (!std::getline(file, line))
        return false;

    if (!line.empty() && line[line.size() - 1] == '\r')
        line.erase(line.size() - 1);

    return true;
}

static bool readNumber(std::ifstream& file, unsigned int& number) {
    std::string line;
    if (!readLine(file, line))
        return false;

    std::istringstream input(line);
    return !!(input >> number);
}

static std::string joinPath(
    const std::string& folder,
    std::string relative
) {
    std::replace(relative.begin(), relative.end(), '/', '\\');
    return folder + "\\" + relative;
}

static std::string fileKey(const std::string& filename) {
    std::string::size_type separator = filename.find_last_of("\\/");
    std::string result = separator == std::string::npos
        ? filename
        : filename.substr(separator + 1);

    for (std::size_t i = 0; i < result.size(); ++i) {
        if (result[i] >= 'A' && result[i] <= 'Z')
            result[i] = (char)(result[i] - 'A' + 'a');
    }

    return result;
}

static bool fileExists(const std::string& filename) {
    DWORD attributes = GetFileAttributesA(filename.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES
        && !(attributes & FILE_ATTRIBUTE_DIRECTORY);
}

static bool finiteValue(double value) {
    return value == value && value > -1e20 && value < 1e20;
}


// ---------------------------------------------------------------------------
// Texture loading
// ---------------------------------------------------------------------------

static GLuint loadTexture(const std::string& filename) {
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file)
        return 0;

    unsigned char header[18];
    if (!readBytes(file, header, sizeof(header)))
        return 0;

    unsigned int width = header[12] | (header[13] << 8);
    unsigned int height = header[14] | (header[15] << 8);

    // Same TGA format as other generated textures:
    // uncompressed true-color, 32-bit BGRA, bottom-left origin.
    if (header[0] != 0 || header[1] != 0 || header[2] != 2
        || header[16] != 32 || header[17] != 8
        || width == 0 || height == 0
        || width > 2048 || height > 2048) {
        return 0;
    }

    if ((width & (width - 1)) || (height & (height - 1)))
        return 0;

    GLint supportedSize = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &supportedSize);

    if (width > (unsigned int)supportedSize
        || height > (unsigned int)supportedSize) {
        return 0;
    }

    std::vector<unsigned char> pixels(
        (std::size_t)width * height * 4
    );

    if (!readBytes(file, &pixels[0], pixels.size()))
        return 0;

    for (std::size_t i = 0; i < pixels.size(); i += 4)
        std::swap(pixels[i], pixels[i + 2]);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    if (!texture)
        return 0;

    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Roads repeat along and across their geometry.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    while (glGetError() != GL_NO_ERROR) {}

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        &pixels[0]
    );

    if (glGetError() != GL_NO_ERROR) {
        glDeleteTextures(1, &texture);
        return 0;
    }

    return texture;
}


// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

bool initialize(const std::string& folder, std::string& error) {
    shutdown();
    InterlockedExchange(&failedLoads, 0);

    if (sizeof(float) != 4
        || sizeof(double) != 8
        || sizeof(unsigned int) != 4
        || sizeof(unsigned short) != 2
        || sizeof(Vertex) != 32) {
        error = "Unsupported compiler data layout for ROAD1.";
        return false;
    }

    std::ifstream materials(
        (folder + "\\road_materials.txt").c_str()
    );

    std::string line;
    unsigned int count = 0;

    if (!readLine(materials, line)
        || line != "ROAD_MATERIALS_V1"
        || !readNumber(materials, count)
        || count == 0 || count > 4096) {
        error = "Cannot read valid road_materials.txt.";
        return false;
    }

    textures.assign(count, 0);

    for (unsigned int i = 0; i < count; ++i) {
        unsigned int identifier;
        std::string materialName;
        std::string relative;

        if (!readNumber(materials, identifier)
            || identifier != i
            || !readLine(materials, materialName)
            || !readLine(materials, relative)) {
            error = "Invalid road material record.";
            shutdown();
            return false;
        }

        std::string filename = joinPath(folder, relative);
        textures[i] = loadTexture(filename);

        if (!textures[i]) {
            error = "Cannot load road texture:\n" + filename;
            shutdown();
            return false;
        }
    }

    std::ifstream index(
        (folder + "\\road_index.txt").c_str()
    );

    if (!readLine(index, line)
        || line != "ROAD_INDEX_V1"
        || !readNumber(index, count)
        || count > 100000) {
        error = "Cannot read valid road_index.txt.";
        shutdown();
        return false;
    }

    for (unsigned int i = 0; i < count; ++i) {
        std::string terrainName;
        std::string relative;

        if (!readLine(index, terrainName)
            || !readLine(index, relative)) {
            error = "Truncated road_index.txt.";
            shutdown();
            return false;
        }

        std::string filename = joinPath(folder, relative);

        if (!fileExists(filename)) {
            std::string warning =
                "Road file missing, skipping: " + filename + "\n";
            OutputDebugStringA(warning.c_str());
            continue;
        }

        std::string key = fileKey(terrainName);

        if (tileFiles.find(key) != tileFiles.end()) {
            error = "Duplicate terrain entry in road_index.txt:\n"
                + terrainName;
            shutdown();
            return false;
        }

        tileFiles[key] = filename;
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

std::string filenameForTerrain(const std::string& terrainFilename) {
    std::map<std::string, std::string>::const_iterator found =
        tileFiles.find(fileKey(terrainFilename));

    return found == tileFiles.end() ? "" : found->second;
}


// ---------------------------------------------------------------------------
// Memory estimate
// ---------------------------------------------------------------------------

std::size_t Data::bytes() const {
    std::size_t total = sizeof(Data)
        + items.capacity() * sizeof(Road);

    for (std::size_t i = 0; i < items.size(); ++i) {
        total += items[i].id.capacity();
        total += items[i].groups.capacity() * sizeof(Group);

        for (std::size_t j = 0; j < items[i].groups.size(); ++j)
            total += items[i].groups[j].vertices.capacity()
                * sizeof(Vertex);
    }

    return total;
}


// ---------------------------------------------------------------------------
// Binary loading
// ---------------------------------------------------------------------------

static bool readData(
    std::ifstream& file,
    Data& data,
    bool (*cancelled)()
) {
    char magic[4];
    unsigned int roadCount;

    if (!readBytes(file, magic, 4)
        || std::memcmp(magic, "ROAD1", 4) != 0
        || !readBytes(file, &roadCount, 4)
        || roadCount > 500000) {
        return false;
    }

    const std::size_t maximumVertices = 8000000;
    std::size_t totalVertices = 0;

    for (unsigned int i = 0; i < roadCount; ++i) {
        if (cancelled())
            return false;

        Road road;
        unsigned short idLength;
        unsigned int groupCount;

        if (!readBytes(file, &idLength, 2) || idLength == 0)
            return false;

        road.id.resize(idLength);
        if (!readBytes(file, &road.id[0], idLength))
            return false;

        if (!readBytes(file, &road.easting, 8)
            || !readBytes(file, &road.northing, 8)
            || !readBytes(file, &road.elevation, 8)
            || !readBytes(file, road.minimum, 12)
            || !readBytes(file, road.maximum, 12)
            || !readBytes(file, &road.luokka, 4)
            || !readBytes(file, &road.versuh, 4)
            || !readBytes(file, &groupCount, 4)
            || groupCount > 4096) {
            return false;
        }

        if (!finiteValue(road.easting)
            || !finiteValue(road.northing)
            || !finiteValue(road.elevation)) {
            return false;
        }

        for (int axis = 0; axis < 3; ++axis) {
            if (!finiteValue(road.minimum[axis])
                || !finiteValue(road.maximum[axis])
                || road.minimum[axis] > road.maximum[axis]) {
                return false;
            }
        }

        for (unsigned int j = 0; j < groupCount; ++j) {
            if (cancelled())
                return false;

            Group group;
            unsigned int vertexCount;

            if (!readBytes(file, &group.category, 4)
                || !readBytes(file, &group.material, 4)
                || !readBytes(file, &vertexCount, 4)
                || vertexCount % 3 != 0) {
                return false;
            }

            if (group.material >= textures.size())
                return false;

            if (vertexCount > maximumVertices - totalVertices)
                return false;

            totalVertices += vertexCount;
            group.vertices.resize(vertexCount);

            for (unsigned int start = 0; start < vertexCount; ) {
                if (cancelled())
                    return false;

                unsigned int block = std::min(
                    4096u, vertexCount - start
                );

                if (!readBytes(
                    file,
                    &group.vertices[start],
                    (std::size_t)block * sizeof(Vertex)
                )) {
                    return false;
                }

                for (unsigned int k = start; k < start + block; ++k) {
                    const Vertex& vertex = group.vertices[k];

                    if (!finiteValue(vertex.x)
                        || !finiteValue(vertex.y)
                        || !finiteValue(vertex.z)
                        || !finiteValue(vertex.nx)
                        || !finiteValue(vertex.ny)
                        || !finiteValue(vertex.nz)
                        || !finiteValue(vertex.u)
                        || !finiteValue(vertex.v)) {
                        return false;
                    }
                }

                start += block;
            }

            road.groups.push_back(Group());
            Group& destination = road.groups.back();
            destination.category = group.category;
            destination.material = group.material;
            destination.vertices.swap(group.vertices);
        }

        data.items.push_back(Road());
        Road& destination = data.items.back();

        destination.id.swap(road.id);
        destination.easting = road.easting;
        destination.northing = road.northing;
        destination.elevation = road.elevation;
        destination.luokka = road.luokka;
        destination.versuh = road.versuh;

        std::memcpy(
            destination.minimum,
            road.minimum,
            sizeof(road.minimum)
        );
        std::memcpy(
            destination.maximum,
            road.maximum,
            sizeof(road.maximum)
        );

        destination.groups.swap(road.groups);
    }

    return !cancelled();
}

Data* load(const std::string& filename, bool (*cancelled)()) {
    // Empty filename is normal for terrain tiles without roads.
    if (filename.empty() || cancelled())
        return 0;

    Data* data = new Data;

    try {
        std::ifstream file(filename.c_str(), std::ios::binary);

        if (file && readData(file, *data, cancelled))
            return data;

    } catch (...) {
        // Report as a load failure below.
    }

    delete data;

    if (!cancelled()) {
        InterlockedIncrement(&failedLoads);
        std::string warning = "Cannot load ROAD1: " + filename + "\n";
        OutputDebugStringA(warning.c_str());
    }

    return 0;
}


// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void setPreset(int preset) {
    if (preset == 0)
        maximumDistance = 800;
    else if (preset == 1)
        maximumDistance = 1200;
    else
        maximumDistance = 10000;
}

void beginFrame() {
    drawnRoads = 0;
    drawnTriangles = 0;
}

static bool visibleBox(
    const double minimum[3],
    const double maximum[3],
    const double planes[6][4]
) {
    for (int p = 0; p < 6; ++p) {
        double x = planes[p][0] >= 0 ? maximum[0] : minimum[0];
        double y = planes[p][1] >= 0 ? maximum[1] : minimum[1];
        double z = planes[p][2] >= 0 ? maximum[2] : minimum[2];

        if (planes[p][0] * x
            + planes[p][1] * y
            + planes[p][2] * z
            + planes[p][3] < 0) {
            return false;
        }
    }

    return true;
}

static double boxDistanceSquared(
    const double minimum[3],
    const double maximum[3]
) {
    double result = 0.0;

    for (int axis = 0; axis < 3; ++axis) {
        double distance = std::max(
            minimum[axis],
            std::max(0.0, -maximum[axis])
        );
        result += distance * distance;
    }

    return result;
}

void render(
    const Data* data,
    double cameraX, double cameraY, double cameraZ,
    double originX, double originY, double originZ,
    const double planes[6][4],
    bool wireframe
) {
    if (!enabled || !data)
        return;

    glPushAttrib(
        GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_COLOR_BUFFER_BIT
        | GL_DEPTH_BUFFER_BIT | GL_POLYGON_BIT | GL_CURRENT_BIT
        | GL_LIGHTING_BIT
    );
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);

    glDisable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    glDisable(GL_CULL_FACE);

    glPolygonMode(
        GL_FRONT_AND_BACK,
        wireframe ? GL_LINE : GL_FILL
    );

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    glDisableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    const double rangeSquared = maximumDistance * maximumDistance;

    for (std::size_t i = 0; i < data->items.size(); ++i) {
        const Road& road = data->items[i];

        double translation[3] = {
            road.easting - originX - cameraX,
            road.elevation - originZ - cameraY,
            originY - road.northing - cameraZ
        };

        double minimum[3], maximum[3];
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = translation[axis] + road.minimum[axis];
            maximum[axis] = translation[axis] + road.maximum[axis];
        }

        if (!visibleBox(minimum, maximum, planes))
            continue;

        if (boxDistanceSquared(minimum, maximum) > rangeSquared)
            continue;

        glPushMatrix();
        glTranslated(
            translation[0],
            translation[1],
            translation[2]
        );

        bool rendered = false;

        for (std::size_t j = 0; j < road.groups.size(); ++j) {
            const Group& group = road.groups[j];
            if (group.vertices.empty())
                continue;

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, textures[group.material]);
            glColor3f(1, 1, 1);

            const Vertex* vertices = &group.vertices[0];

            glVertexPointer(
                3, GL_FLOAT, sizeof(Vertex), &vertices[0].x
            );
            glNormalPointer(
                GL_FLOAT, sizeof(Vertex), &vertices[0].nx
            );
            glTexCoordPointer(
                2, GL_FLOAT, sizeof(Vertex), &vertices[0].u
            );

            glDrawArrays(
                GL_TRIANGLES,
                0,
                (GLsizei)group.vertices.size()
            );

            drawnTriangles +=
                (unsigned long)(group.vertices.size() / 3);
            rendered = true;
        }

        glPopMatrix();

        if (rendered)
            ++drawnRoads;
    }

    glPopClientAttrib();
    glPopAttrib();
}


// ---------------------------------------------------------------------------
// Diagnostics and shutdown
// ---------------------------------------------------------------------------

unsigned int indexedTileCount() {
    return (unsigned int)tileFiles.size();
}

long failedLoadCount() {
    return (long)InterlockedCompareExchange(&failedLoads, 0, 0);
}

void shutdown() {
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (textures[i])
            glDeleteTextures(1, &textures[i]);
    }

    textures.clear();
    tileFiles.clear();
}

} // namespace roads

