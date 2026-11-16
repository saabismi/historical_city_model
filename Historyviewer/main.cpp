// Request Windows XP-compatible API declarations.
// Actual XP compatibility also depends on the compiler and runtime.
#define WINVER 0x0501
#define _WIN32_WINNT 0x0501

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <GL/gl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "vegetation.h"
#include "buildings.h"
#include "roads.h"

using namespace std;

// Existing terrain location:
const string tilesLocation = "espoo2008_terrain_quick";

// Generated land-cover package containing vegetation and ground materials.
const string vegetationLocation = "espoo2008_landcover_viewer";

// NEW: folder containing building_index.txt, building_materials.txt,
// textures/, and tiles/. Change this name to match your actual output folder.
const string buildingsLocation = "espoo2008_buildings_all";

// Location of the generated roads (includes textures)
const string roadsLocation = "espoo2008_roads";
const string groundLocation = "espoo2008_landcover_viewer";

const double PI = 3.14159265358979323846;

// Viewer coordinates: X east, Y up, Z south.
struct Vec {
    double x, y, z;

    Vec(double X = 0, double Y = 0, double Z = 0)
        : x(X), y(Y), z(Z) {}

    Vec operator+(const Vec& other) const {
        return Vec(x + other.x, y + other.y, z + other.z);
    }

    Vec operator-(const Vec& other) const {
        return Vec(x - other.x, y - other.y, z - other.z);
    }

    Vec operator*(double scalar) const {
        return Vec(x * scalar, y * scalar, z * scalar);
    }
};

double dot(const Vec& a, const Vec& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

Vec cross(const Vec& a, const Vec& b) {
    return Vec(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

Vec unit(const Vec& vector) {
    double length = std::sqrt(dot(vector, vector));
    return length > 1e-12
        ? vector * (1.0 / length)
        : Vec(0, 1, 0);
}

bool finiteFloat(float value) {
    return value == value && value > -1e30f && value < 1e30f;
}

// Terrain vertex: position followed by normal.
struct Vertex {
    float x, y, z;
    float nx, ny, nz;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<unsigned int> triangleCells;

    struct MaterialGroup {
        unsigned int material;
        std::vector<unsigned int> indices;
    };
    std::vector<MaterialGroup> materialGroups;
    std::vector<unsigned int> untexturedIndices;

    size_t bytes() const {
        size_t total = vertices.capacity() * sizeof(Vertex)
            + indices.capacity() * sizeof(unsigned int)
            + triangleCells.capacity() * sizeof(unsigned int);
        total += untexturedIndices.capacity() * sizeof(unsigned int);
        for (size_t i = 0; i < materialGroups.size(); ++i)
            total += sizeof(MaterialGroup)
                + materialGroups[i].indices.capacity()
                    * sizeof(unsigned int);
        return total;
    }
};

struct GroundData {
    unsigned int columns;
    unsigned int rows;
    float resolution;
    std::vector<unsigned short> materials;
};

struct GroundMaterial {
    std::string id;
    GLuint texture;
};

// Each terrain tile may have vegetation and buildings, or neither.
//
// Only building tiles listed in building_index.txt receive a
// buildingFilename. All other terrain tiles work normally.
struct Tile {
    std::string filename;
    std::string vegetationFilename;
    std::string buildingFilename;

    std::string roadFilename;
    std::string groundFilename;
    roads::Data* roadsData;
    GroundData* groundData;


    vegetation::Data* plants;
    buildings::Data* structures;

    double xmin, ymin;
    double xmax, ymax;
    double zmin, zmax;

    Mesh* mesh;
    int loadedLOD;
    int wantedLOD;
    bool pending;
    bool failed;

    Tile()
        : plants(0),
          structures(0),
          roadsData(0),
          groundData(0),
          mesh(0),
          loadedLOD(-1),
          wantedLOD(-1),
          pending(false),
          failed(false) {}
};

struct Request {
    int tile;
    int lod;
};

struct Result {
    int tile;
    int lod;
    Mesh* mesh;
    vegetation::Data* plants;
    buildings::Data* structures;
    roads::Data* roadsData;
    GroundData* groundData;
};

std::vector<GroundMaterial> groundMaterials;
std::map<std::string, std::string> groundFiles;
bool groundEnabled = false;

// Window and OpenGL objects.
HWND windowHandle = 0;
HDC deviceContext = 0;
HGLRC renderingContext = 0;
GLuint fontBase = 0;

int viewportWidth = 1100;
int viewportHeight = 750;

// Controls.
bool running = true;
bool helpVisible = true;
bool wireframe = false;
bool orbitMode = true;
bool settingsVisible = false;
int selectedSetting = 0;

bool leftDragging = false;
bool rightDragging = false;
int mouseX = 0;
int mouseY = 0;

bool keys[256] = { false };

// Camera.
Vec camera;
Vec target;
double yaw = 0;
double pitch = -0.5;
double orbitDistance = 800;
double movementSpeed = 80;

// Terrain index and shared map origin.
std::vector<Tile> tiles;
std::string terrainFolder;

double originX = 0;
double originY = 0;
double originZ = 0;

// Streaming defaults. The graphics presets below retain your actual
// configured values.
double loadRadius = 650;
double unloadRadius = 850;
double farPlane = 2000;
size_t memoryBudget = 96 * 1024 * 1024;
int lodBias = 1;

struct ViewerSettings {
    double viewDistance;
    double loadDistance;
    double unloadDistance;
    int cacheMegabytes;
    int terrainDetail;
    int vegetationCards;
    int vegetationLodFloor;
};

ViewerSettings viewerSettings = {
    1200, 1500, 1800, 96, 1, 3, 0
};

struct SettingsItem {
    const char* label;
    int kind;
    double minimum;
    double maximum;
    double step;
};

enum {
    SETTING_VIEW_DISTANCE,
    SETTING_LOAD_DISTANCE,
    SETTING_UNLOAD_DISTANCE,
    SETTING_CACHE,
    SETTING_TERRAIN_DETAIL,
    SETTING_VEGETATION_CARDS,
    SETTING_VEGETATION_LOD,
    SETTING_EXIT
};

const SettingsItem settingsItems[] = {
    {"View distance", SETTING_VIEW_DISTANCE, 400, 20000, 200},
    {"Load radius", SETTING_LOAD_DISTANCE, 400, 20000, 200},
    {"Unload radius", SETTING_UNLOAD_DISTANCE, 600, 25000, 200},
    {"Cache (MB)", SETTING_CACHE, 32, 512, 16},
    {"Terrain detail", SETTING_TERRAIN_DETAIL, 0, 3, 1},
    {"Vegetation cards", SETTING_VEGETATION_CARDS, 2, 4, 1},
    {"Vegetation LOD floor", SETTING_VEGETATION_LOD, 0, 2, 1},
    {"Exit to Windows", SETTING_EXIT, 0, 0, 0}
};

const int settingsItemCount =
    (int)(sizeof(settingsItems) / sizeof(settingsItems[0]));

// Terrain frame statistics.
int drawnTiles = 0;
unsigned long drawnTriangles = 0;
double fps = 0;

// Background worker state.
CRITICAL_SECTION loaderLock;
HANDLE loaderEvent = 0;
HANDLE loaderThread = 0;
volatile LONG loaderStop = 0;

std::vector<Request> requests;
std::vector<Result> results;

bool loaderShouldStop() {
    return InterlockedCompareExchange(&loaderStop, 0, 0) != 0;
}

// ---------------------------------------------------------------------------
// Terrain-file loading
// ---------------------------------------------------------------------------

bool readBytes(std::ifstream& file, void* data, size_t bytes) {
    file.read((char*)data, (std::streamsize)bytes);
    return !!file;
}

bool readUInt(std::ifstream& file, unsigned int& value) {
    return readBytes(file, &value, 4);
}

std::string terrainKey(const std::string& filename) {
    std::string result = filename;
    std::string::size_type separator = result.find_last_of("\\/");
    if (separator != std::string::npos)
        result = result.substr(separator + 1);
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i] >= 'A' && result[i] <= 'Z')
            result[i] = (char)(result[i] - 'A' + 'a');
    }
    return result;
}

GLuint loadGroundTexture(const std::string& filename) {
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file) return 0;

    unsigned char header[18];
    if (!readBytes(file, header, sizeof(header)))
        return 0;

    unsigned int width = header[12] | (header[13] << 8);
    unsigned int height = header[14] | (header[15] << 8);
    if (header[0] != 0 || header[1] != 0 || header[2] != 2
        || header[16] != 32 || header[17] != 8
        || width == 0 || height == 0 || width > 2048 || height > 2048)
        return 0;

    std::vector<unsigned char> pixels(
        (size_t)width * height * 4
    );
    if (!readBytes(file, &pixels[0], pixels.size()))
        return 0;

    for (size_t i = 0; i < pixels.size(); i += 4)
        std::swap(pixels[i], pixels[i + 2]);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    if (!texture) return 0;
    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, &pixels[0]
    );
    return texture;
}

bool initializeGroundMaterials(const std::string& folder) {
    std::ifstream file((folder + "\\ground_materials.txt").c_str());
    if (!file) return false;

    std::string line;
    std::getline(file, line);
    if (!line.empty() && line[line.size() - 1] == '\r')
        line.erase(line.size() - 1);
    if (line != "GROUND_MATERIALS_V1")
        return false;

    unsigned int count = 0;
    if (!(file >> count) || count == 0 || count > 256)
        return false;
    std::getline(file, line);

    for (unsigned int i = 0; i < count; ++i) {
        GroundMaterial material;
        std::string relative;
        if (!(file >> material.id >> relative))
            return false;
        material.texture = loadGroundTexture(
            folder + "\\" + relative
        );
        if (!material.texture)
            return false;
        groundMaterials.push_back(material);
    }
    return true;
}

bool initializeGroundIndex(const std::string& folder) {
    std::ifstream file((folder + "\\ground_index.txt").c_str());
    if (!file) return false;
    std::string magic;
    std::getline(file, magic);
    if (!magic.empty() && magic[magic.size() - 1] == '\r')
        magic.erase(magic.size() - 1);
    if (magic != "GROUND_INDEX_V1")
        return false;

    unsigned int count = 0;
    if (!(file >> count) || count > 100000)
        return false;
    for (unsigned int i = 0; i < count; ++i) {
        std::string terrain;
        std::string relative;
        if (!(file >> terrain >> relative))
            return false;
        groundFiles[terrainKey(terrain)] = folder + "\\" + relative;
    }

    return true;
}

void shutdownGround() {
    for (size_t i = 0; i < groundMaterials.size(); ++i) {
        if (groundMaterials[i].texture)
            glDeleteTextures(1, &groundMaterials[i].texture);
    }
    groundMaterials.clear();
    groundFiles.clear();
    groundEnabled = false;
}

GroundData* loadGround(const std::string& filename) {
    if (filename.empty()) return 0;
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file) return 0;
    char magic[4];
    unsigned int columns, rows;
    float resolution;
    if (!readBytes(file, magic, 4)
        || std::string(magic, 4) != "LCM1"
        || !readUInt(file, columns)
        || !readUInt(file, rows)
        || !readBytes(file, &resolution, sizeof(resolution))
        || columns == 0 || rows == 0
        || columns > 4096 || rows > 4096
        || resolution <= 0)
        return 0;

    GroundData* data = new GroundData;
    data->columns = columns;
    data->rows = rows;
    data->resolution = resolution;
    data->materials.resize((size_t)columns * rows);
    if (!readBytes(
        file, &data->materials[0],
        data->materials.size() * sizeof(unsigned short)
    )) {
        delete data;
        return 0;
    }
    return data;
}

void buildGroundGroups(
    Mesh* mesh,
    GroundData* ground,
    const Tile& tile
) {
    if (!mesh || !ground || mesh->triangleCells.empty())
        return;

    std::map<unsigned int, size_t> groups;
    size_t triangleCount = mesh->triangleCells.size();
    for (size_t triangle = 0; triangle < triangleCount; ++triangle) {
        unsigned int first = (unsigned int)(triangle * 3);
        if (first + 2 >= mesh->indices.size())
            continue;
        unsigned int a = mesh->indices[first];
        unsigned int b = mesh->indices[first + 1];
        unsigned int c = mesh->indices[first + 2];
        double x = (
            mesh->vertices[a].x
            + mesh->vertices[b].x
            + mesh->vertices[c].x
        ) / 3.0;
        double south = (
            mesh->vertices[a].z
            + mesh->vertices[b].z
            + mesh->vertices[c].z
        ) / 3.0;
        double worldY = tile.ymax - south;
        int column = (int)std::floor(
            x / ground->resolution
        );
        int row = (int)std::floor(
            (worldY - tile.ymin) / ground->resolution
        );
        if (column < 0 || row < 0
            || column >= (int)ground->columns
            || row >= (int)ground->rows) {
            mesh->untexturedIndices.push_back(a);
            mesh->untexturedIndices.push_back(b);
            mesh->untexturedIndices.push_back(c);
            continue;
        }
        unsigned int material =
            ground->materials[(size_t)row * ground->columns + column];
        std::map<unsigned int, size_t>::iterator found =
            groups.find(material);
        size_t groupIndex;
        if (found == groups.end()) {
            groupIndex = mesh->materialGroups.size();
            groups[material] = groupIndex;
            Mesh::MaterialGroup group;
            group.material = material;
            mesh->materialGroups.push_back(group);
        } else {
            groupIndex = found->second;
        }
        Mesh::MaterialGroup& group = mesh->materialGroups[groupIndex];
        group.indices.push_back(a);
        group.indices.push_back(b);
        group.indices.push_back(c);
    }
}

// Read one selected terrain LOD on the background thread.
Mesh* loadTerrain(const std::string& filename, int requestedLOD) {
    if (loaderShouldStop()) return 0;

    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file) return 0;

    char magic[4];
    unsigned int levels;
    double bounds[4];
    float elevationBounds[2];

    if (!readBytes(file, magic, 4)
        || std::string(magic, 4) != "TRN1"
        || !readUInt(file, levels)
        || levels < 1 || levels > 10
        || !readBytes(file, bounds, sizeof(bounds))
        || !readBytes(
            file, elevationBounds, sizeof(elevationBounds)
        )) {
        return 0;
    }

    if (loaderShouldStop()) return 0;

    int selected = std::max(
        0, std::min(requestedLOD, (int)levels - 1)
    );

    for (unsigned int level = 0; level < levels; ++level) {
        if (loaderShouldStop()) return 0;

        unsigned int columns, rows;

        if (!readUInt(file, columns) || !readUInt(file, rows))
            return 0;

        if (columns < 2 || rows < 2
            || columns > 4096 || rows > 4096) {
            return 0;
        }

        size_t haloCount =
            (size_t)(columns + 2) * (rows + 2);

        size_t bytes =
            ((size_t)columns + rows + haloCount) * sizeof(float);

        if ((int)level != selected) {
            file.seekg((std::streamoff)bytes, std::ios::cur);
            if (!file) return 0;
            continue;
        }

        std::vector<float> east(columns);
        std::vector<float> south(rows);
        std::vector<float> heights(haloCount);

        if (!readBytes(
            file, &east[0], columns * sizeof(float)
        )) return 0;

        if (loaderShouldStop()) return 0;

        if (!readBytes(
            file, &south[0], rows * sizeof(float)
        )) return 0;

        if (loaderShouldStop()) return 0;

        if (!readBytes(
            file, &heights[0], haloCount * sizeof(float)
        )) return 0;

        if (loaderShouldStop()) return 0;

        double baseSpacing =
            east[columns - 1] / (columns - 1);

        if (columns > 2)
            baseSpacing = east[1] - east[0];

        if (baseSpacing <= 0) return 0;

        Mesh* mesh = new Mesh;
        mesh->vertices.resize((size_t)columns * rows);

        for (unsigned int r = 0; r < rows; ++r) {
            if (loaderShouldStop()) {
                delete mesh;
                return 0;
            }

            for (unsigned int c = 0; c < columns; ++c) {
                size_t index = (size_t)r * columns + c;
                size_t h =
                    (size_t)(r + 1) * (columns + 2) + c + 1;

                float height = heights[h];
                Vertex& vertex = mesh->vertices[index];

                vertex.x = east[c];
                vertex.y = finiteFloat(height)
                    ? (float)(height - originZ)
                    : 0;
                vertex.z = south[r];

                float left = heights[h - 1];
                float right = heights[h + 1];
                float north = heights[h - (columns + 2)];
                float southHeight =
                    heights[h + (columns + 2)];

                double dx =
                    (c == 0
                        ? baseSpacing
                        : east[c] - east[c - 1])
                    + (c + 1 == columns
                        ? baseSpacing
                        : east[c + 1] - east[c]);

                double dz =
                    (r == 0
                        ? baseSpacing
                        : south[r] - south[r - 1])
                    + (r + 1 == rows
                        ? baseSpacing
                        : south[r + 1] - south[r]);

                Vec normal(0, 1, 0);

                if (dx > 0 && dz > 0
                    && finiteFloat(left)
                    && finiteFloat(right)
                    && finiteFloat(north)
                    && finiteFloat(southHeight)) {
                    normal = unit(Vec(
                        -(right - left) / dx,
                        1,
                        -(southHeight - north) / dz
                    ));
                }

                vertex.nx = (float)normal.x;
                vertex.ny = (float)normal.y;
                vertex.nz = (float)normal.z;
            }
        }

        if (loaderShouldStop()) {
            delete mesh;
            return 0;
        }

        mesh->indices.reserve(
            (size_t)(columns - 1) * (rows - 1) * 6
        );

        for (unsigned int r = 0; r + 1 < rows; ++r) {
            if (loaderShouldStop()) {
                delete mesh;
                return 0;
            }

            for (unsigned int c = 0; c + 1 < columns; ++c) {
                unsigned int a = r * columns + c;
                unsigned int b = a + 1;
                unsigned int d =
                    (r + 1) * columns + c + 1;
                unsigned int e = d - 1;

                unsigned int candidates[6] = {
                    a, e, b,
                    b, e, d
                };

                for (int triangle = 0; triangle < 2; ++triangle) {
                    bool valid = true;

                    for (int k = 0; k < 3; ++k) {
                        unsigned int vertexIndex =
                            candidates[triangle * 3 + k];

                        unsigned int rr =
                            vertexIndex / columns;
                        unsigned int cc =
                            vertexIndex % columns;

                        size_t hi =
                            (size_t)(rr + 1) * (columns + 2)
                            + cc + 1;

                        if (!finiteFloat(heights[hi]))
                            valid = false;
                    }

                    if (valid) {
                        for (int k = 0; k < 3; ++k) {
                            mesh->indices.push_back(
                                candidates[triangle * 3 + k]
                            );
                        }
                        mesh->triangleCells.push_back(
                            r * (columns - 1) + c
                        );
                    }
                }
            }
        }

        if (loaderShouldStop() || mesh->indices.empty()) {
            delete mesh;
            return 0;
        }

        return mesh;
    }

    return 0;
}

// Read all terrain tile metadata. Geometry is loaded later.
bool loadIndex(const std::string& folder) {
    std::ifstream file(
        (folder + "\\terrain_index.txt").c_str()
    );
    if (!file) return false;

    std::string magic;
    std::getline(file, magic);

    if (!magic.empty()
        && magic[magic.size() - 1] == '\r') {
        magic.erase(magic.size() - 1);
    }

    if (magic != "TERRAIN_INDEX_V1")
        return false;

    unsigned int count;

    if (!(file >> originX >> originY >> originZ >> count)
        || count == 0 || count > 100000) {
        return false;
    }

    tiles.reserve(count);

    for (unsigned int i = 0; i < count; ++i) {
        Tile tile;
        std::string relative;

        if (!(file >> relative
             >> tile.xmin >> tile.ymin
             >> tile.xmax >> tile.ymax
             >> tile.zmin >> tile.zmax)) {
            return false;
        }

        std::replace(
            relative.begin(), relative.end(), '/', '\\'
        );
        tile.filename = folder + "\\" + relative;

        if (tile.xmax <= tile.xmin
            || tile.ymax <= tile.ymin) {
            return false;
        }

        tiles.push_back(tile);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Background loading
// ---------------------------------------------------------------------------

// Load terrain plus any vegetation/buildings mapped to that terrain tile.
// A missing building mapping is normal for the 100-building sample.
DWORD WINAPI loaderMain(void*) {
    while (!loaderShouldStop()) {
        WaitForSingleObject(loaderEvent, 100);

        if (loaderShouldStop())
            break;

        Request request;
        bool available = false;

        EnterCriticalSection(&loaderLock);

        if (!requests.empty()) {
            request = requests.front();
            requests.erase(requests.begin());
            available = true;
        }

        LeaveCriticalSection(&loaderLock);

        if (!available)
            continue;

        Mesh* mesh = loadTerrain(
            tiles[request.tile].filename,
            request.lod
        );

        vegetation::Data* plants = 0;
        buildings::Data* structures = 0;
        roads::Data* roadsData = 0;
        GroundData* groundData = 0;

        if (mesh && !loaderShouldStop()) {
            plants = vegetation::load(
                tiles[request.tile].vegetationFilename,
                loaderShouldStop
            );
        }

        if (mesh && !loaderShouldStop()) {
            structures = buildings::load(
                tiles[request.tile].buildingFilename,
                loaderShouldStop
            );
        }

        if (mesh && !loaderShouldStop()) {
            roadsData = roads::load(
                tiles[request.tile].roadFilename,
                loaderShouldStop
            );
        }
        if (mesh && !loaderShouldStop()) {
            groundData = loadGround(
                tiles[request.tile].groundFilename
            );
        }

        if (loaderShouldStop()) {
            delete mesh;
            delete plants;
            delete structures;
            delete roadsData;
            delete groundData;
            break;
        }

        Result result;
        result.tile = request.tile;
        result.lod = request.lod;
        result.mesh = mesh;
        result.plants = plants;
        result.structures = structures;
        result.roadsData = roadsData;
        result.groundData = groundData;


        EnterCriticalSection(&loaderLock);
        results.push_back(result);
        LeaveCriticalSection(&loaderLock);
    }

    return 0;
}

// Horizontal distance to a terrain tile's rectangular footprint.
double tileDistance(const Tile& tile) {
    double x0 = tile.xmin - originX;
    double x1 = tile.xmax - originX;
    double z0 = originY - tile.ymax;
    double z1 = originY - tile.ymin;

    double dx = std::max(
        x0 - camera.x,
        std::max(0.0, camera.x - x1)
    );
    double dz = std::max(
        z0 - camera.z,
        std::max(0.0, camera.z - z1)
    );

    return std::sqrt(dx*dx + dz*dz);
}

// A terrain tile is visible only as one atomic scene unit. Optional data is
// considered ready when no file was indexed for that terrain tile.
bool tileReady(const Tile& tile) {
    return tile.mesh
        && (!groundEnabled || tile.groundFilename.empty() || tile.groundData)
        && (tile.vegetationFilename.empty() || tile.plants)
        && (tile.buildingFilename.empty() || tile.structures)
        && (tile.roadFilename.empty() || tile.roadsData);
}

// Approximate CPU memory occupied by resident terrain, plants and buildings.
size_t cacheBytes() {
    size_t bytes = 0;

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (tiles[i].mesh)
            bytes += tiles[i].mesh->bytes();

        if (tiles[i].plants)
            bytes += tiles[i].plants->bytes();

        if (tiles[i].structures)
            bytes += tiles[i].structures->bytes();

        if (tiles[i].roadsData)
            bytes += tiles[i].roadsData->bytes();
        if (tiles[i].groundData)
            bytes += sizeof(GroundData)
                + tiles[i].groundData->materials.capacity()
                    * sizeof(unsigned short);
    }

    return bytes;
}

// Take ownership of completed worker data and maintain the residency cache.
void updateStreaming() {
    std::vector<Result> completed;

    EnterCriticalSection(&loaderLock);
    completed.swap(results);
    LeaveCriticalSection(&loaderLock);

    for (size_t i = 0; i < completed.size(); ++i) {
        Result& result = completed[i];
        Tile& tile = tiles[result.tile];

        tile.pending = false;

        if (!result.mesh) {
            delete result.plants;
            delete result.structures;
            delete result.roadsData;
            delete result.groundData;

            tile.failed = true;
            continue;
        }

        if (tileDistance(tile) > unloadRadius) {
            delete result.mesh;
            delete result.plants;
            delete result.structures;
            delete result.roadsData;
            delete result.groundData;
            continue;
        }

        delete tile.mesh;
        delete tile.plants;
        delete tile.structures;
        delete tile.roadsData;
        delete tile.groundData;

        tile.mesh = result.mesh;
        tile.plants = result.plants;
        tile.structures = result.structures;
        tile.roadsData = result.roadsData;
        tile.groundData = result.groundData;
        buildGroundGroups(tile.mesh, tile.groundData, tile);
        tile.loadedLOD = result.lod;
    }

    // Evict terrain and its associated data together.
    for (size_t i = 0; i < tiles.size(); ++i) {
        Tile& tile = tiles[i];

        if (tileDistance(tile) > unloadRadius) {
            delete tile.mesh;
            delete tile.plants;
            delete tile.structures;
            delete tile.roadsData;
            delete tile.groundData;

            tile.mesh = 0;
            tile.plants = 0;
            tile.structures = 0;
            tile.roadsData = 0;
            tile.groundData = 0;
            tile.loadedLOD = -1;
        }
    }

    size_t bytes = cacheBytes();

    // Keep your existing cache budget, evicting farthest data first.
    while (bytes > memoryBudget) {
        int farthest = -1;
        double farthestDistance = -1;

        for (size_t i = 0; i < tiles.size(); ++i) {
            if (tiles[i].mesh
                || tiles[i].plants
                || tiles[i].structures
                || tiles[i].roadsData
                || tiles[i].groundData) {
                double distance = tileDistance(tiles[i]);

                if (distance > farthestDistance) {
                    farthestDistance = distance;
                    farthest = (int)i;
                }
            }
        }

        if (farthest < 0)
            break;

        Tile& tile = tiles[farthest];

        if (tile.mesh)
            bytes -= tile.mesh->bytes();

        if (tile.plants)
            bytes -= tile.plants->bytes();

        if (tile.structures)
            bytes -= tile.structures->bytes();

        if (tile.roadsData)
            bytes -= tile.roadsData->bytes();
        if (tile.groundData)
            bytes -= sizeof(GroundData)
                + tile.groundData->materials.capacity()
                    * sizeof(unsigned short);

        delete tile.mesh;
        delete tile.plants;
        delete tile.structures;
        delete tile.roadsData;
        delete tile.groundData;

        tile.mesh = 0;
        tile.plants = 0;
        tile.structures = 0;
        tile.roadsData = 0;
        tile.groundData = 0;
        tile.loadedLOD = -1;
    }

    int best = -1;
    double bestDistance = 1e100;

    for (size_t i = 0; i < tiles.size(); ++i) {
        Tile& tile = tiles[i];
        double distance = tileDistance(tile);

        int lod = lodBias;
        if (distance > 200) ++lod;
        if (distance > 450) ++lod;
        if (distance > 800) ++lod;
        lod = std::min(lod, 4);

        tile.wantedLOD = lod;

        bool needs =
            !tile.mesh
            || tile.loadedLOD != lod
            || (
                !tile.vegetationFilename.empty()
                && !tile.plants
            )
            || (
                !tile.buildingFilename.empty()
                && !tile.structures
            )
            || (
                !tile.roadFilename.empty()
                && !tile.roadsData
            )
            || (
                groundEnabled
                && !tile.groundFilename.empty()
                && !tile.groundData
            );

        if (distance <= loadRadius
            && needs
            && !tile.pending
            && !tile.failed
            && distance < bestDistance) {
            bestDistance = distance;
            best = (int)i;
        }
    }

    EnterCriticalSection(&loaderLock);

    int pendingCount = 0;

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (tiles[i].pending)
            ++pendingCount;
    }

    // Continue using the existing one-worker, short-queue approach.
    if (best >= 0 && pendingCount < 2) {
        Request request;
        request.tile = best;
        request.lod = tiles[best].wantedLOD;

        tiles[best].pending = true;
        requests.push_back(request);
        SetEvent(loaderEvent);
    }

    LeaveCriticalSection(&loaderLock);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

Vec forwardVector() {
    return Vec(
        std::sin(yaw) * std::cos(pitch),
        std::sin(pitch),
        -std::cos(yaw) * std::cos(pitch)
    );
}

void updateOrbitCamera() {
    camera = target - forwardVector() * orbitDistance;
}

void homeCamera() {
    double xmin = tiles[0].xmin;
    double xmax = tiles[0].xmax;
    double ymin = tiles[0].ymin;
    double ymax = tiles[0].ymax;
    double highest = tiles[0].zmax;

    for (size_t i = 1; i < tiles.size(); ++i) {
        xmin = std::min(xmin, tiles[i].xmin);
        xmax = std::max(xmax, tiles[i].xmax);
        ymin = std::min(ymin, tiles[i].ymin);
        ymax = std::max(ymax, tiles[i].ymax);
        highest = std::max(highest, tiles[i].zmax);
    }

    target = Vec(
        (xmin + xmax) * 0.5 - originX,
        highest - originZ,
        originY - (ymin + ymax) * 0.5
    );

    orbitDistance = loadRadius * 0.8;
    yaw = 0;
    pitch = -0.55;
    updateOrbitCamera();
}

// Move the camera to a tile in the partial building dataset.
void focusBuildingSample() {
    for (size_t i = 0; i < tiles.size(); ++i) {
        if (tiles[i].buildingFilename.empty())
            continue;

        target = Vec(
            (tiles[i].xmin + tiles[i].xmax) * 0.5
                - originX,
            tiles[i].zmin - originZ + 5.0,
            originY
                - (tiles[i].ymin + tiles[i].ymax) * 0.5
        );

        orbitMode = true;
        orbitDistance = 350;
        yaw = 0;
        pitch = -0.55;
        updateOrbitCamera();
        return;
    }

    MessageBoxA(
        windowHandle,
        "No building files match the loaded terrain tiles.",
        "Building sample",
        MB_OK | MB_ICONINFORMATION
    );
}

void switchCamera() {
    if (orbitMode) {
        orbitMode = false;
    } else {
        target = camera
               + forwardVector() * orbitDistance;
        orbitMode = true;
    }
}

void updateCamera(double dt) {
    double speed = movementSpeed * dt;

    if (keys[VK_SHIFT])
        speed *= 4;

    Vec forward = forwardVector();
    Vec right(std::cos(yaw), 0, std::sin(yaw));
    Vec movement;

    if (keys['W'])
        movement = movement + forward * speed;
    if (keys['S'])
        movement = movement - forward * speed;
    if (keys['D'])
        movement = movement + right * speed;
    if (keys['A'])
        movement = movement - right * speed;
    if (keys['E'])
        movement.y += speed;
    if (keys['Q'])
        movement.y -= speed;

    if (orbitMode) {
        target = target + movement;
        updateOrbitCamera();
    } else {
        camera = camera + movement;
    }
}

// ---------------------------------------------------------------------------
// Frustum culling
// ---------------------------------------------------------------------------

// Camera-relative clipping planes shared with vegetation/buildings.
double planes[6][4];

void extractFrustum() {
    double projection[16];
    double view[16];
    double clip[16];

    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetDoublev(GL_MODELVIEW_MATRIX, view);

    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            clip[column*4 + row] = 0;

            for (int k = 0; k < 4; ++k) {
                clip[column*4 + row] +=
                    projection[k*4 + row]
                    * view[column*4 + k];
            }
        }
    }

    for (int p = 0; p < 6; ++p) {
        int axis = p / 2;
        double sign = (p % 2 == 0) ? 1 : -1;

        for (int k = 0; k < 4; ++k) {
            planes[p][k] =
                clip[k*4 + 3]
                + sign * clip[k*4 + axis];
        }

        double length = std::sqrt(
            planes[p][0]*planes[p][0]
            + planes[p][1]*planes[p][1]
            + planes[p][2]*planes[p][2]
        );

        if (length > 0) {
            for (int k = 0; k < 4; ++k)
                planes[p][k] /= length;
        }
    }
}

bool visible(const Tile& tile) {
    double x0 = tile.xmin - originX - camera.x;
    double x1 = tile.xmax - originX - camera.x;
    double y0 = tile.zmin - originZ - camera.y;
    double y1 = tile.zmax - originZ - camera.y;
    double z0 = originY - tile.ymax - camera.z;
    double z1 = originY - tile.ymin - camera.z;

    for (int p = 0; p < 6; ++p) {
        double x = planes[p][0] >= 0 ? x1 : x0;
        double y = planes[p][1] >= 0 ? y1 : y0;
        double z = planes[p][2] >= 0 ? z1 : z0;

        if (planes[p][0]*x
            + planes[p][1]*y
            + planes[p][2]*z
            + planes[p][3] < 0) {
            return false;
        }
    }

    return true;
}

void applyViewerSettings() {
    viewerSettings.loadDistance = std::max(
        viewerSettings.loadDistance,
        viewerSettings.viewDistance
    );
    viewerSettings.unloadDistance = std::max(
        viewerSettings.unloadDistance,
        viewerSettings.loadDistance + 100.0
    );

    loadRadius = viewerSettings.loadDistance;
    unloadRadius = viewerSettings.unloadDistance;
    farPlane = viewerSettings.viewDistance;
    memoryBudget =
        (size_t)viewerSettings.cacheMegabytes * 1024 * 1024;
    lodBias = viewerSettings.terrainDetail;

    buildings::setMaximumDistance(viewerSettings.viewDistance);
    roads::setMaximumDistance(viewerSettings.viewDistance);
    vegetation::setRenderSettings(
        viewerSettings.vegetationLodFloor,
        viewerSettings.vegetationCards
    );
}

void setInitialViewerSettings(int preset) {
    viewerSettings.viewDistance = farPlane;
    viewerSettings.loadDistance = loadRadius;
    viewerSettings.unloadDistance = unloadRadius;
    viewerSettings.cacheMegabytes =
        (int)(memoryBudget / (1024 * 1024));
    viewerSettings.terrainDetail = lodBias;
    viewerSettings.vegetationCards = preset == 0 ? 2 : preset == 1 ? 3 : 4;
    viewerSettings.vegetationLodFloor = preset == 0 ? 1 : 0;
}

double settingValue(int index) {
    switch (settingsItems[index].kind) {
    case SETTING_VIEW_DISTANCE: return viewerSettings.viewDistance;
    case SETTING_LOAD_DISTANCE: return viewerSettings.loadDistance;
    case SETTING_UNLOAD_DISTANCE: return viewerSettings.unloadDistance;
    case SETTING_CACHE: return viewerSettings.cacheMegabytes;
    case SETTING_TERRAIN_DETAIL: return viewerSettings.terrainDetail;
    case SETTING_VEGETATION_CARDS: return viewerSettings.vegetationCards;
    case SETTING_VEGETATION_LOD: return viewerSettings.vegetationLodFloor;
    default: return 0;
    }
}

void changeSetting(int direction) {
    if (selectedSetting < 0 || selectedSetting >= settingsItemCount)
        return;

    const SettingsItem& item = settingsItems[selectedSetting];
    if (item.kind == SETTING_EXIT) {
        if (direction > 0) running = false;
        return;
    }

    double value = settingValue(selectedSetting)
        + direction * item.step;
    value = std::max(item.minimum, std::min(item.maximum, value));

    switch (item.kind) {
    case SETTING_VIEW_DISTANCE:
        viewerSettings.viewDistance = value;
        break;
    case SETTING_LOAD_DISTANCE:
        viewerSettings.loadDistance = value;
        break;
    case SETTING_UNLOAD_DISTANCE:
        viewerSettings.unloadDistance = value;
        break;
    case SETTING_CACHE:
        viewerSettings.cacheMegabytes = (int)value;
        break;
    case SETTING_TERRAIN_DETAIL:
        viewerSettings.terrainDetail = (int)value;
        break;
    case SETTING_VEGETATION_CARDS:
        viewerSettings.vegetationCards = (int)value;
        break;
    case SETTING_VEGETATION_LOD:
        viewerSettings.vegetationLodFloor = (int)value;
        break;
    }

    applyViewerSettings();
}

void formatSettingValue(int index, char* output, size_t capacity) {
    if (settingsItems[index].kind == SETTING_EXIT) {
        std::sprintf(output, "button");
        return;
    }

    double value = settingValue(index);
    if (settingsItems[index].kind == SETTING_CACHE
        || settingsItems[index].kind == SETTING_TERRAIN_DETAIL
        || settingsItems[index].kind == SETTING_VEGETATION_CARDS
        || settingsItems[index].kind == SETTING_VEGETATION_LOD) {
        std::sprintf(output, "%d", (int)value);
    } else {
        std::sprintf(output, "%.0f m", value);
    }
    (void)capacity;
}

// ---------------------------------------------------------------------------
// Text overlay
// ---------------------------------------------------------------------------

void text(int x, int y, const char* value) {
    glRasterPos2i(x, y);
    glListBase(fontBase);

    glCallLists(
        (GLsizei)std::strlen(value),
        GL_UNSIGNED_BYTE,
        value
    );
}

void drawOverlay() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glOrtho(
        0, viewportWidth,
        viewportHeight, 0,
        -1, 1
    );

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    char status[256];

    std::sprintf(
        status,
        "%s | %.0f FPS | %d tiles | %lu terrain triangles | %.1f MB cache",
        orbitMode ? "Orbit" : "Flight",
        fps,
        drawnTiles,
        drawnTriangles,
        cacheBytes() / (1024.0 * 1024.0)
    );

    text(12, 22, status);

        char roadStatus[192];

    std::sprintf(
        roadStatus,
        "Roads: %s | %lu drawn | %lu triangles | %u indexed tiles | %ld load errors",
        roads::enabled ? "on" : "off",
        roads::drawnRoads,
        roads::drawnTriangles,
        roads::indexedTileCount(),
        roads::failedLoadCount()
    );

    text(12, 190, roadStatus);


    if (helpVisible) {
        text(
            12, 46,
            "Tab: camera  F1: help  F2: wireframe  F3: vegetation  F4: buildings F6: roads"
        );
        text(
            12, 66,
            "WASD: move    Q/E: down/up    Shift: faster"
        );
        text(
            12, 86,
            "Flight: right-drag look; wheel changes speed"
        );
        text(
            12, 106,
            "Orbit: left-drag rotate; right-drag pan; wheel zoom"
        );
        text(
            12, 126,
            "Home: world view  F5: building sample  Esc: settings"
        );
    }

    char vegetationStatus[128];

    std::sprintf(
        vegetationStatus,
        "Vegetation: %s | %lu plants drawn",
        vegetation::enabled ? "on" : "off",
        vegetation::drawnPlants
    );
    text(12, 150, vegetationStatus);

    char groundStatus[128];
    std::sprintf(
        groundStatus,
        "Ground textures: %s",
        groundEnabled ? "on" : "off"
    );
    text(12, 170, groundStatus);

    char buildingStatus[192];

    std::sprintf(
        buildingStatus,
        "Buildings: %s | %lu drawn | %lu triangles | %u indexed tiles | %ld load errors",
        buildings::enabled ? "on" : "off",
        buildings::drawnBuildings,
        buildings::drawnTriangles,
        buildings::indexedTileCount(),
        buildings::failedLoadCount()
    );
    text(12, 170, buildingStatus);

    glEnable(GL_DEPTH_TEST);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void drawSettingsMenu() {
    if (!settingsVisible) return;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, viewportWidth, viewportHeight, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const int left = std::max(20, viewportWidth / 2 - 260);
    const int top = std::max(20, viewportHeight / 2 - 250);
    const int width = 520;
    const int rowHeight = 38;

    const int bottom =
        top + 60 + settingsItemCount * rowHeight;

    glColor4f(0.16f, 0.16f, 0.16f, 1.0f);
    glBegin(GL_QUADS);
    glVertex2i(left, top);
    glVertex2i(left + width, top);
    glVertex2i(left + width, bottom);
    glVertex2i(left, bottom);
    glEnd();

    glColor3f(1, 1, 1);
    glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2i(left, top);
    glVertex2i(left + width, top);
    glVertex2i(left + width, bottom);
    glVertex2i(left, bottom);
    glEnd();

    glColor3f(1, 1, 1);
    text(left + 24, top + 32, "Historyviewer settings");
    text(
        left + 24, top + 52,
        "Up/Down select  Left/Right change  Enter activate  Esc close"
    );

    for (int i = 0; i < settingsItemCount; ++i) {
        int y = top + 68 + i * rowHeight;
        bool selected = i == selectedSetting;

        char value[32];
        char line[128];
        formatSettingValue(i, value, sizeof(value));
        std::sprintf(line, "%s: %s", settingsItems[i].label, value);
        glColor3f(
            selected ? 1.0f : 1.0f,
            selected ? 0.85f : 1.0f,
            selected ? 0.0f : 1.0f
        );
        text(left + 28, y, line);
    }

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ---------------------------------------------------------------------------
// Scene rendering
// ---------------------------------------------------------------------------

void render() {
    glViewport(0, 0, viewportWidth, viewportHeight);
    glClearColor(0.25f, 0.32f, 0.40f, 1);

    glClear(
        GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT
    );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    double nearPlane = 1.0;
    double halfHeight =
        nearPlane * std::tan(60.0 * PI / 360.0);
    double halfWidth =
        halfHeight * viewportWidth / viewportHeight;

    glFrustum(
        -halfWidth, halfWidth,
        -halfHeight, halfHeight,
        nearPlane, farPlane
    );

    Vec forward = unit(forwardVector());
    Vec right = unit(
        cross(forward, Vec(0, 1, 0))
    );
    Vec up = cross(right, forward);

    // Rotation only; each renderer supplies camera-relative translation.
    double view[16] = {
        right.x, up.x, -forward.x, 0,
        right.y, up.y, -forward.y, 0,
        right.z, up.z, -forward.z, 0,
        0, 0, 0, 1
    };

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(view);
    extractFrustum();

    GLfloat lightPosition[] = {
        -0.4f, 0.8f, 0.3f, 0
    };
    GLfloat diffuse[] = {
        0.85f, 0.85f, 0.80f, 1
    };
    GLfloat ambient[] = {
        0.30f, 0.30f, 0.32f, 1
    };

    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightModelfv(
        GL_LIGHT_MODEL_AMBIENT, ambient
    );

    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);

    glColorMaterial(
        GL_FRONT_AND_BACK,
        GL_AMBIENT_AND_DIFFUSE
    );

    glColor3f(0.65f, 0.67f, 0.61f);

    glPolygonMode(
        GL_FRONT_AND_BACK,
        wireframe ? GL_LINE : GL_FILL
    );

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    drawnTiles = 0;
    drawnTriangles = 0;

    // Terrain.
    for (size_t i = 0; i < tiles.size(); ++i) {
        Tile& tile = tiles[i];

        if (!tileReady(tile) || !visible(tile))
            continue;

        Mesh& mesh = *tile.mesh;

        glPushMatrix();

        glTranslated(
            tile.xmin - originX - camera.x,
            -camera.y,
            originY - tile.ymax - camera.z
        );

        glVertexPointer(
            3, GL_FLOAT, sizeof(Vertex),
            &mesh.vertices[0].x
        );
        glNormalPointer(
            GL_FLOAT, sizeof(Vertex),
            &mesh.vertices[0].nx
        );

        if (groundEnabled && !mesh.materialGroups.empty()) {
            glColor3f(1.0f, 1.0f, 1.0f);
            glEnable(GL_TEXTURE_2D);
            glEnable(GL_TEXTURE_GEN_S);
            glEnable(GL_TEXTURE_GEN_T);
            glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
            glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
            GLfloat sPlane[] = { 0.05f, 0.0f, 0.0f, 0.0f };
            GLfloat tPlane[] = { 0.0f, 0.0f, 0.05f, 0.0f };
            glTexGenfv(GL_S, GL_OBJECT_PLANE, sPlane);
            glTexGenfv(GL_T, GL_OBJECT_PLANE, tPlane);
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

            for (size_t group = 0;
                 group < mesh.materialGroups.size();
                 ++group) {
                unsigned int material =
                    mesh.materialGroups[group].material;
                if (material >= groundMaterials.size())
                    continue;
                glBindTexture(
                    GL_TEXTURE_2D,
                    groundMaterials[material].texture
                );
                const std::vector<unsigned int>& indices =
                    mesh.materialGroups[group].indices;
                if (!indices.empty()) {
                    glDrawElements(
                        GL_TRIANGLES,
                        (GLsizei)indices.size(),
                        GL_UNSIGNED_INT,
                        &indices[0]
                    );
                }
            }
            if (!mesh.untexturedIndices.empty()) {
                glDisable(GL_TEXTURE_2D);
                glColor3f(0.65f, 0.67f, 0.61f);
                glDrawElements(
                    GL_TRIANGLES,
                    (GLsizei)mesh.untexturedIndices.size(),
                    GL_UNSIGNED_INT,
                    &mesh.untexturedIndices[0]
                );
            }
            glDisable(GL_TEXTURE_GEN_S);
            glDisable(GL_TEXTURE_GEN_T);
            glDisable(GL_TEXTURE_2D);
            glColor3f(0.65f, 0.67f, 0.61f);
        } else {
            glDrawElements(
                GL_TRIANGLES,
                (GLsizei)mesh.indices.size(),
                GL_UNSIGNED_INT,
                &mesh.indices[0]
            );
        }

        glPopMatrix();

        ++drawnTiles;
        drawnTriangles +=
            (unsigned long)(mesh.indices.size() / 3);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Render roads before buildings or vegetation
    roads::beginFrame();

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (!tileReady(tiles[i]) || !visible(tiles[i]))
            continue;

        roads::render(
            tiles[i].roadsData,
            camera.x, camera.y, camera.z,
            originX, originY, originZ,
            planes,
            wireframe
        );
    }


    // Opaque buildings must be drawn before alpha-tested plants.
    // Building culling uses complete building bounds, rather than merely
    // the terrain tile's rectangular footprint.
    buildings::beginFrame();

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (!tileReady(tiles[i]) || !visible(tiles[i]))
            continue;

        buildings::render(
            tiles[i].structures,
            camera.x, camera.y, camera.z,
            originX, originY, originZ,
            planes,
            wireframe
        );
    }

    // Vegetation.
    vegetation::beginFrame();

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (!tileReady(tiles[i]) || !visible(tiles[i]))
            continue;

        vegetation::collect(
            tiles[i].plants,
            camera.x, camera.y, camera.z,
            originX, originY, originZ,
            planes,
            yaw,
            pitch
        );
    }

    vegetation::draw();

    if (orbitMode) {
        glDisable(GL_LIGHTING);
        glColor3f(1, 0.5f, 0.1f);

        Vec point = target - camera;
        double size =
            std::max(1.0, orbitDistance * 0.004);

        glBegin(GL_LINES);

        glVertex3d(
            point.x - size, point.y, point.z
        );
        glVertex3d(
            point.x + size, point.y, point.z
        );

        glVertex3d(
            point.x, point.y - size, point.z
        );
        glVertex3d(
            point.x, point.y + size, point.z
        );

        glEnd();
    }

    drawOverlay();
    drawSettingsMenu();
    SwapBuffers(deviceContext);
}

// ---------------------------------------------------------------------------
// Windows input
// ---------------------------------------------------------------------------

LRESULT CALLBACK windowProc(
    HWND hwnd,
    UINT message,
    WPARAM wParam,
    LPARAM lParam
) {
    switch (message) {
    case WM_CLOSE:
        running = false;
        return 0;

    case WM_SIZE:
        viewportWidth =
            std::max(1, (int)LOWORD(lParam));
        viewportHeight =
            std::max(1, (int)HIWORD(lParam));
        return 0;

    case WM_KILLFOCUS:
        for (int i = 0; i < 256; ++i)
            keys[i] = false;

        leftDragging = false;
        rightDragging = false;
        ReleaseCapture();
        return 0;

    case WM_KEYDOWN:
        if (settingsVisible) {
            if (!(lParam & (1L << 30))) {
                if (wParam == VK_ESCAPE) {
                    settingsVisible = false;
                    selectedSetting = 0;
                } else if (wParam == VK_UP) {
                    selectedSetting =
                        (selectedSetting + settingsItemCount - 1)
                        % settingsItemCount;
                } else if (wParam == VK_DOWN) {
                    selectedSetting =
                        (selectedSetting + 1) % settingsItemCount;
                } else if (wParam == VK_LEFT) {
                    changeSetting(-1);
                } else if (wParam == VK_RIGHT) {
                    changeSetting(1);
                } else if (wParam == VK_RETURN) {
                    if (settingsItems[selectedSetting].kind
                        == SETTING_EXIT) {
                        running = false;
                    } else {
                        changeSetting(1);
                    }
                }
            }
            return 0;
        }

        if (wParam < 256)
            keys[wParam] = true;

        // Ignore repeats for actions that toggle a state.
        if (!(lParam & (1L << 30))) {
            if (wParam == VK_ESCAPE) {
                settingsVisible = true;
                leftDragging = false;
                rightDragging = false;
                for (int i = 0; i < 256; ++i)
                    keys[i] = false;
            }

            if (wParam == VK_F1)
                helpVisible = !helpVisible;

            if (wParam == VK_F2)
                wireframe = !wireframe;

            if (wParam == VK_F3)
                vegetation::enabled =
                    !vegetation::enabled;

            if (wParam == VK_F4)
                buildings::enabled =
                    !buildings::enabled;

            if (wParam == VK_F5)
                focusBuildingSample();

            if (wParam == VK_F6)
                roads::enabled = !roads::enabled;

            if (wParam == VK_TAB)
                switchCamera();

            if (wParam == VK_HOME)
                homeCamera();
        }

        return 0;

    case WM_KEYUP:
        if (wParam < 256)
            keys[wParam] = false;
        return 0;

    case WM_LBUTTONDOWN:
        if (settingsVisible) {
            int x = (short)LOWORD(lParam);
            int y = (short)HIWORD(lParam);
            int left = std::max(20, viewportWidth / 2 - 260);
            int top = std::max(20, viewportHeight / 2 - 250);
            int row =
                (y - (top + 46)) / 38;

            if (x >= left + 16 && x <= left + 504
                && row >= 0 && row < settingsItemCount) {
                selectedSetting = row;
                if (settingsItems[row].kind == SETTING_EXIT)
                    running = false;
                else
                    changeSetting(1);
            }
            return 0;
        }

    case WM_RBUTTONDOWN:
        if (message == WM_LBUTTONDOWN)
            leftDragging = true;
        else
            rightDragging = true;

        mouseX = (short)LOWORD(lParam);
        mouseY = (short)HIWORD(lParam);
        SetCapture(hwnd);
        return 0;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
        if (message == WM_LBUTTONUP)
            leftDragging = false;
        else
            rightDragging = false;

        if (!leftDragging && !rightDragging)
            ReleaseCapture();

        return 0;

    case WM_MOUSEMOVE: {
        int x = (short)LOWORD(lParam);
        int y = (short)HIWORD(lParam);
        int dx = x - mouseX;
        int dy = y - mouseY;

        mouseX = x;
        mouseY = y;

        if ((!orbitMode && rightDragging)
            || (orbitMode && leftDragging)) {
            yaw += dx * 0.005;
            pitch -= dy * 0.005;
            pitch = std::max(
                -1.50,
                std::min(1.50, pitch)
            );
        } else if (orbitMode && rightDragging) {
            Vec right(
                std::cos(yaw), 0, std::sin(yaw)
            );
            Vec up = unit(
                cross(right, forwardVector())
            );
            double scale =
                orbitDistance * 0.0015;

            target =
                target
                - right * (dx * scale)
                + up * (dy * scale);
        }

        if (orbitMode)
            updateOrbitCamera();

        return 0;
    }

    case WM_MOUSEWHEEL: {
        double steps =
            (short)HIWORD(wParam) / 120.0;

        if (orbitMode) {
            orbitDistance *= std::pow(
                0.85, steps
            );
            orbitDistance = std::max(
                5.0,
                std::min(5000.0, orbitDistance)
            );
            updateOrbitCamera();
        } else {
            movementSpeed *= std::pow(
                1.2, steps
            );
            movementSpeed = std::max(
                1.0,
                std::min(2000.0, movementSpeed)
            );
        }

        return 0;
    }

    case WM_ERASEBKGND:
        return 1;
    }

    return DefWindowProcA(
        hwnd, message, wParam, lParam
    );
}

// ---------------------------------------------------------------------------
// Startup graphics selector
// ---------------------------------------------------------------------------

int selectedPreset = -1;

LRESULT CALLBACK launcherProc(
    HWND hwnd,
    UINT message,
    WPARAM wParam,
    LPARAM lParam
) {
    if (message == WM_COMMAND) {
        int id = LOWORD(wParam);

        if (id >= 101 && id <= 103) {
            selectedPreset = id - 101;
            DestroyWindow(hwnd);
        }

        return 0;
    }

    if (message == WM_CLOSE) {
        selectedPreset = -2;
        DestroyWindow(hwnd);
        return 0;
    }

    return DefWindowProcA(
        hwnd, message, wParam, lParam
    );
}

int choosePreset(HINSTANCE instance) {
    WNDCLASSA wc;
    ZeroMemory(&wc, sizeof(wc));

    wc.lpfnWndProc = launcherProc;
    wc.hInstance = instance;
    wc.lpszClassName = "TerrainLauncher";
    wc.hCursor = LoadCursor(0, IDC_ARROW);
    wc.hbrBackground =
        (HBRUSH)(COLOR_BTNFACE + 1);

    RegisterClassA(&wc);

    HWND launcher = CreateWindowA(
        wc.lpszClassName,
        "Historyviewer - graphics",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
        CW_USEDEFAULT, CW_USEDEFAULT,
        470, 180,
        0, 0, instance, 0
    );

    if (!launcher)
        return -2;

    CreateWindowA(
        "STATIC",
        "Choose a preset. Start with Low on older hardware.",
        WS_CHILD | WS_VISIBLE,
        20, 20, 420, 25,
        launcher, 0, instance, 0
    );

    const char* labels[] = {
        "Low", "Medium", "High"
    };

    for (int i = 0; i < 3; ++i) {
        CreateWindowA(
            "BUTTON",
            labels[i],
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            20 + i * 145, 65,
            130, 35,
            launcher,
            (HMENU)(INT_PTR)(101 + i),
            instance,
            0
        );
    }

    ShowWindow(launcher, SW_SHOW);

    MSG message;

    while (selectedPreset == -1
           && GetMessage(&message, 0, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessage(&message);
    }

    return selectedPreset;
}

// ---------------------------------------------------------------------------
// OpenGL initialization
// ---------------------------------------------------------------------------

bool initializeOpenGL(HINSTANCE instance) {
    WNDCLASSA wc;
    ZeroMemory(&wc, sizeof(wc));

    wc.style = CS_OWNDC;
    wc.lpfnWndProc = windowProc;
    wc.hInstance = instance;
    wc.lpszClassName = "LegacyTerrainViewer";
    wc.hCursor = LoadCursor(0, IDC_ARROW);

    if (!RegisterClassA(&wc))
        return false;

    windowHandle = CreateWindowA(
        wc.lpszClassName,
        "Historyviewer",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        viewportWidth, viewportHeight,
        0, 0, instance, 0
    );

    if (!windowHandle)
        return false;

    deviceContext = GetDC(windowHandle);

    PIXELFORMATDESCRIPTOR format;
    ZeroMemory(&format, sizeof(format));

    format.nSize = sizeof(format);
    format.nVersion = 1;
    format.dwFlags =
        PFD_DRAW_TO_WINDOW
        | PFD_SUPPORT_OPENGL
        | PFD_DOUBLEBUFFER;
    format.iPixelType = PFD_TYPE_RGBA;
    format.cColorBits = 24;
    format.cDepthBits = 24;
    format.iLayerType = PFD_MAIN_PLANE;

    int selected =
        ChoosePixelFormat(deviceContext, &format);

    if (!selected
        || !SetPixelFormat(
            deviceContext, selected, &format
        )) {
        return false;
    }

    renderingContext =
        wglCreateContext(deviceContext);

    if (!renderingContext
        || !wglMakeCurrent(
            deviceContext, renderingContext
        )) {
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_SMOOTH);

    fontBase = glGenLists(256);

    HFONT font = CreateFontA(
        -15, 0, 0, 0,
        FW_NORMAL,
        FALSE, FALSE, FALSE,
        ANSI_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY,
        FIXED_PITCH,
        "Courier New"
    );

    HGDIOBJ oldFont =
        SelectObject(deviceContext, font);

    BOOL fontOK = wglUseFontBitmapsA(
        deviceContext, 0, 256, fontBase
    );

    SelectObject(deviceContext, oldFont);
    DeleteObject(font);

    if (!fontOK)
        return false;

    ShowWindow(windowHandle, SW_SHOW);
    return true;
}

// ---------------------------------------------------------------------------
// Application entry point
// ---------------------------------------------------------------------------

int WINAPI WinMain(
    HINSTANCE instance,
    HINSTANCE,
    LPSTR commandLine,
    int
) {
    // Existing terrain-folder command-line behavior is unchanged.
    terrainFolder = commandLine;

    while (!terrainFolder.empty()
           && terrainFolder[0] == ' ') {
        terrainFolder.erase(0, 1);
    }

    while (!terrainFolder.empty()
           && terrainFolder[
               terrainFolder.size() - 1
           ] == ' ') {
        terrainFolder.erase(
            terrainFolder.size() - 1
        );
    }

    if (terrainFolder.size() >= 2
        && terrainFolder[0] == '"'
        && terrainFolder[
            terrainFolder.size() - 1
        ] == '"') {
        terrainFolder = terrainFolder.substr(
            1, terrainFolder.size() - 2
        );
    }

    if (terrainFolder.empty())
        terrainFolder = tilesLocation;

    if (!loadIndex(terrainFolder)) {
        MessageBoxA(
            0,
            "Cannot read terrain_index.txt.\n\n"
            "Launch with the prepared terrain folder "
            "as an argument.",
            "Historyviewer",
            MB_OK | MB_ICONERROR
        );
        return 1;
    }

    int preset = choosePreset(instance);

    if (preset < 0)
        return 0;

    // YOUR EXISTING GRAPHICS PRESET VALUES: unchanged.
    if (preset == 0) {
        loadRadius = 800;
        unloadRadius = 1100;
        farPlane = 800;
        memoryBudget = 48 * 1024 * 1024;
        lodBias = 2;
    } else if (preset == 1) {
        loadRadius = 1200;
        unloadRadius = 1500;
        farPlane = 1200;
        memoryBudget = 96 * 1024 * 1024;
        lodBias = 1;
    } else {
        loadRadius = 10000;
        unloadRadius = 15000;
        farPlane = 10000;
        memoryBudget = 192 * 1024 * 1024;
        lodBias = 0;
    }
    setInitialViewerSettings(preset);

    if (!initializeOpenGL(instance)) {
        MessageBoxA(
            0,
            "OpenGL initialization failed.",
            "Historyviewer",
            MB_OK | MB_ICONERROR
        );
        return 1;
    }

    // Textures must be created while the OpenGL context is current.
    vegetation::setPreset(preset);

    std::string vegetationError;

    if (!vegetation::initialize(
        vegetationLocation,
        vegetationError
    )) {
        MessageBoxA(
            windowHandle,
            vegetationError.c_str(),
            "Vegetation initialization failed",
            MB_OK | MB_ICONWARNING
        );

        vegetation::enabled = false;
    }

    for (size_t i = 0; i < tiles.size(); ++i) {
        tiles[i].vegetationFilename =
            vegetation::filenameForTerrain(
                tiles[i].filename
            );
    }

    // Initialize shared building materials and optional tile mappings.
    buildings::setPreset(preset);

    std::string buildingError;

    if (!buildings::initialize(
        buildingsLocation,
        buildingError
    )) {
        MessageBoxA(
            windowHandle,
            buildingError.c_str(),
            "Building initialization failed",
            MB_OK | MB_ICONWARNING
        );

        // Terrain and vegetation still work if buildings cannot load.
        buildings::enabled = false;
    }

        roads::setPreset(preset);

    std::string roadError;

    if (!roads::initialize(
        roadsLocation,
        roadError
    )) {
        MessageBoxA(
            windowHandle,
            roadError.c_str(),
            "Road initialization failed",
            MB_OK | MB_ICONWARNING
        );

        roads::enabled = false;
    }

    unsigned int matchedRoadTiles = 0;

    for (size_t i = 0; i < tiles.size(); ++i) {
        tiles[i].roadFilename =
            roads::filenameForTerrain(tiles[i].filename);

        if (!tiles[i].roadFilename.empty())
            ++matchedRoadTiles;
    }

    if (roads::indexedTileCount() > 0 && matchedRoadTiles == 0) {
        MessageBoxA(
            windowHandle,
            "Road files were indexed, but none matched terrain filenames.",
            "Road tile mapping",
            MB_OK | MB_ICONWARNING
        );
    }

    if (initializeGroundMaterials(groundLocation)
        && initializeGroundIndex(groundLocation)) {
        groundEnabled = true;
    } else {
        shutdownGround();
        MessageBoxA(
            windowHandle,
            "Ground material package was not found or is invalid.\n"
            "Terrain will render with the legacy untextured mode.",
            "Ground materials",
            MB_OK | MB_ICONWARNING
        );
    }

    for (size_t i = 0; i < tiles.size(); ++i) {
        std::map<std::string, std::string>::const_iterator found =
            groundFiles.find(terrainKey(tiles[i].filename));
        if (found != groundFiles.end())
            tiles[i].groundFilename = found->second;
    }

    applyViewerSettings();


    unsigned int matchedBuildingTiles = 0;

    for (size_t i = 0; i < tiles.size(); ++i) {
        tiles[i].buildingFilename =
            buildings::filenameForTerrain(
                tiles[i].filename
            );

        if (!tiles[i].buildingFilename.empty())
            ++matchedBuildingTiles;
    }

    if (buildings::indexedTileCount() > 0
        && matchedBuildingTiles == 0) {
        MessageBoxA(
            windowHandle,
            "Building files were indexed, but none matched "
            "terrain filenames.\n"
            "Check that the buildings were generated "
            "for this terrain dataset.",
            "Building tile mapping",
            MB_OK | MB_ICONWARNING
        );
    }

    // Start the worker only after the indexes and textures are ready.
    InitializeCriticalSection(&loaderLock);

    loaderEvent = CreateEventA(
        0, FALSE, FALSE, 0
    );

    if (loaderEvent) {
        loaderThread = CreateThread(
            0, 0, loaderMain, 0, 0, 0
        );
    }

    if (!loaderEvent || !loaderThread) {
        MessageBoxA(
            0,
            "Cannot start the background loader.",
            "Historyviewer",
            MB_OK | MB_ICONERROR
        );

        if (loaderEvent)
            CloseHandle(loaderEvent);

        DeleteCriticalSection(&loaderLock);

        buildings::shutdown();
        vegetation::shutdown();
        shutdownGround();

        if (fontBase)
            glDeleteLists(fontBase, 256);

        wglMakeCurrent(0, 0);
        wglDeleteContext(renderingContext);
        ReleaseDC(windowHandle, deviceContext);
        DestroyWindow(windowHandle);

        return 1;
    }

    homeCamera();

    LARGE_INTEGER frequency;
    LARGE_INTEGER previous;
    LARGE_INTEGER now;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&previous);

    double fpsTime = 0;
    int fpsFrames = 0;

    while (running) {
        MSG message;

        while (PeekMessage(
            &message, 0, 0, 0, PM_REMOVE
        )) {
            if (message.message == WM_QUIT)
                running = false;

            TranslateMessage(&message);
            DispatchMessage(&message);
        }

        if (!running)
            break;

        QueryPerformanceCounter(&now);

        double dt =
            (double)(
                now.QuadPart - previous.QuadPart
            ) / frequency.QuadPart;

        previous = now;
        dt = std::min(dt, 0.1);

        updateCamera(dt);
        updateStreaming();
        render();

        fpsTime += dt;
        ++fpsFrames;

        if (fpsTime >= 0.5) {
            fps = fpsFrames / fpsTime;
            fpsTime = 0;
            fpsFrames = 0;
        }

        Sleep(1);
    }

    // Cooperatively stop the background loader.
    InterlockedExchange(&loaderStop, 1);

    EnterCriticalSection(&loaderLock);
    requests.clear();
    LeaveCriticalSection(&loaderLock);

    SetEvent(loaderEvent);

    // Keep servicing Windows messages until the loader has really stopped.
    for (;;) {
        DWORD waitResult = MsgWaitForMultipleObjects(
            1,
            &loaderThread,
            FALSE,
            INFINITE,
            QS_ALLINPUT
        );

        if (waitResult == WAIT_OBJECT_0) {
            break;
        }

        if (waitResult == WAIT_OBJECT_0 + 1) {
            MSG message;

            while (PeekMessage(
                &message, 0, 0, 0, PM_REMOVE
            )) {
                if (message.message != WM_QUIT) {
                    TranslateMessage(&message);
                    DispatchMessage(&message);
                }
            }
        } else {
            // Do not free objects that a live worker may still access.
            ExitProcess(1);
        }
    }

    CloseHandle(loaderThread);
    CloseHandle(loaderEvent);

    // Release any results that the main thread had not yet adopted.
    for (size_t i = 0; i < results.size(); ++i) {
        delete results[i].mesh;
        delete results[i].plants;
        delete results[i].structures;
        delete results[i].roadsData;
        delete results[i].groundData;

    }

    // Release all resident tile data.
    for (size_t i = 0; i < tiles.size(); ++i) {
        delete tiles[i].mesh;
        delete tiles[i].plants;
        delete tiles[i].structures;
        delete tiles[i].roadsData;
        delete tiles[i].groundData;

    }

    DeleteCriticalSection(&loaderLock);

    // Release OpenGL resources while the context still exists.
    if (fontBase)
        glDeleteLists(fontBase, 256);

    vegetation::shutdown();
    buildings::shutdown();
    shutdownGround();

    wglMakeCurrent(0, 0);
    wglDeleteContext(renderingContext);

    ReleaseDC(windowHandle, deviceContext);
    DestroyWindow(windowHandle);

    return 0;
}
