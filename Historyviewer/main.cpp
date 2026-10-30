// Request Windows XP-compatible API declarations.
// Actual XP compatibility also depends on your compiler/runtime.
#define WINVER 0x0501
#define _WIN32_WINNT 0x0501

// Exclude unnecessary Windows headers and avoid min/max macros.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <GL/gl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vegetation.h"

using namespace std;

// Config for setting the terrain files' location (folder with terrain_index.txt and "tiles" folder which has .trn files)
const string tilesLocation = "2008_terrain";

// Folder produced by pack_viewer_vegetation.py.
const string vegetationLocation = "2008_vegetation_viewer";

// ===========================================================================
// Math helpers
// ===========================================================================

// Used to convert angles between degrees and radians.
const double PI = 3.14159265358979323846;

// A three-dimensional vector.
//
// Viewer coordinates:
//   X: east
//   Y: up
//   Z: south
struct Vec {
    double x, y, z; // Components of the vector.

    // Construct a vector; omitted components default to zero.
    Vec(double X = 0, double Y = 0, double Z = 0)
        : x(X), y(Y), z(Z) {}

    // Add two vectors.
    Vec operator+(const Vec& b) const {
        return Vec(x + b.x, y + b.y, z + b.z);
    }

    // Subtract another vector.
    Vec operator-(const Vec& b) const {
        return Vec(x - b.x, y - b.y, z - b.z);
    }

    // Multiply the vector by a scalar.
    Vec operator*(double s) const {
        return Vec(x * s, y * s, z * s);
    }
};

// Calculate the dot product, used for lengths and projections.
double dot(const Vec& a, const Vec& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Calculate a vector perpendicular to both input vectors.
Vec cross(const Vec& a, const Vec& b) {
    return Vec(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

// Normalize a vector to length one.
// Return an upward vector if the input is too small to normalize.
Vec unit(const Vec& a) {
    double length = std::sqrt(dot(a, a)); // Input vector length.
    return length > 1e-12 ? a * (1.0 / length) : Vec(0, 1, 0);
}

// Check whether a terrain value is usable.
// NaN fails "value == value"; the bounds reject infinities and extreme values.
bool finiteFloat(float value) {
    return value == value && value > -1e30f && value < 1e30f;
}

// ===========================================================================
// Terrain data structures
// ===========================================================================

// One renderable vertex: position followed by its surface normal.
struct Vertex {
    float x, y, z;    // Tile-local position; Y uses the shared elevation origin.
    float nx, ny, nz; // Unit normal used by fixed-function lighting.
};

// One terrain mesh at one level of detail.
struct Mesh {
    std::vector<Vertex> vertices;       // Vertex positions and normals.
    std::vector<unsigned int> indices;  // Three vertex indexes per triangle.

    // Estimate allocated CPU storage, including spare vector capacity.
    size_t bytes() const {
        return vertices.capacity() * sizeof(Vertex)
             + indices.capacity() * sizeof(unsigned int);
    }
};

// Metadata and loading state for a terrain tile.
struct Tile {
    std::string filename; // Full path to the tile's .trn file.

    std::string vegetationFilename; // Associated VEG1 placement file.
    vegetation::Data* plants;       // Resident vegetation placements.


    double xmin, ymin; // Minimum easting and northing.
    double xmax, ymax; // Maximum easting and northing.
    double zmin, zmax; // Minimum and maximum source elevation.

    Mesh* mesh;       // Currently resident mesh; null if not loaded.
    int loadedLOD;    // Requested LOD of resident mesh; -1 if absent.
    int wantedLOD;    // LOD currently selected by the distance rule.
    bool pending;     // True while a request is queued or being processed.
    bool failed;      // True after a loading failure; prevents repeated retries.

    // Initialize loading state. Bounds are filled by loadIndex().
    Tile()
        : mesh(0), loadedLOD(-1), wantedLOD(-1),
          pending(false), failed(false), plants(0) {}

};

// A task sent from the rendering thread to the background loader.
struct Request {
    int tile; // Index into the global tiles vector.
    int lod;  // Requested detail level.
};

// A completed background-loading task.
struct Result {
    int tile;                    // Destination terrain tile.
    int lod;                     // Requested terrain detail level.
    Mesh* mesh;                  // Newly loaded terrain geometry.
    vegetation::Data* plants;    // Newly loaded vegetation placements.
};


// ===========================================================================
// Window and rendering state
// ===========================================================================

HWND windowHandle = 0;          // Main Windows window.
HDC deviceContext = 0;          // Window drawing surface.
HGLRC renderingContext = 0;     // OpenGL rendering context.
GLuint fontBase = 0;            // First bitmap-font display-list identifier.

int viewportWidth = 1100;      // Current render area width.
int viewportHeight = 750;      // Current render area height.

bool running = true;           // Main-loop continuation flag.
bool helpVisible = true;       // Whether to show keyboard/mouse help.
bool wireframe = false;        // Draw triangle outlines instead of filled faces.
bool orbitMode = true;         // Orbit mode if true; free flight otherwise.

bool leftDragging = false;     // Left mouse button is held.
bool rightDragging = false;    // Right mouse button is held.
int mouseX = 0;                // Previous mouse X position.
int mouseY = 0;                // Previous mouse Y position.

bool keys[256] = { false };    // Current state of Windows virtual keys.

// ===========================================================================
// Camera and terrain placement
// ===========================================================================

Vec camera;                    // Camera position relative to the shared origin.
Vec target;                    // Orbit target relative to the shared origin.
double yaw = 0;                // Horizontal viewing angle in radians.
double pitch = -0.5;           // Vertical viewing angle in radians.
double orbitDistance = 800;    // Camera-to-target distance in metres.
double movementSpeed = 80;     // Normal keyboard movement speed in metres/second.

std::vector<Tile> tiles;       // Metadata for every tile; meshes load on demand.
std::string terrainFolder;    // Folder containing terrain_index.txt.

double originX = 0;            // Shared easting origin.
double originY = 0;            // Shared northing origin.
double originZ = 0;            // Shared elevation origin.

// ===========================================================================
// Streaming and graphics-preset settings
// ===========================================================================

double loadRadius = 650;       // Request tiles within this horizontal distance.
double unloadRadius = 850;     // Unload beyond this distance; provides hysteresis.
double farPlane = 1200;        // Maximum camera-space rendering distance.
size_t memoryBudget = 96 * 1024 * 1024; // Resident CPU mesh-cache budget.
int lodBias = 1;               // Starting LOD: larger numbers mean less detail.

// ===========================================================================
// Statistics
// ===========================================================================

int drawnTiles = 0;            // Tiles drawn during the most recent frame.
unsigned long drawnTriangles = 0; // Triangles submitted during that frame.
double fps = 0;                // Smoothed displayed frame rate.

// ===========================================================================
// Background-loader synchronization
// ===========================================================================

CRITICAL_SECTION loaderLock;   // Protects requests and results.
HANDLE loaderEvent = 0;        // Wakes the worker when a request is added.
HANDLE loaderThread = 0;       // Background thread handle.
volatile LONG loaderStop = 0;  // Atomic stop flag checked by the worker.

std::vector<Request> requests; // Pending tasks; protected by loaderLock.
std::vector<Result> results;   // Completed tasks; protected by loaderLock.

// Read the background loader's stop flag atomically.
// Must appear before loadTerrain() and loaderMain().
bool loaderShouldStop() {
    return InterlockedCompareExchange(&loaderStop, 0, 0) != 0;
}

// ===========================================================================
// Binary terrain-file loading
// ===========================================================================

// Read exactly the requested number of bytes.
// Return false if the file is truncated or unreadable.
bool readBytes(std::ifstream& file, void* data, size_t bytes) {
    file.read((char*)data, (std::streamsize)bytes);
    return !!file;
}

// Read one little-endian 32-bit unsigned integer.
// This directly matches the intended Windows x86 target.
bool readUInt(std::ifstream& file, unsigned int& value) {
    return readBytes(file, &value, 4);
}

// Read one terrain LOD and build its mesh on the background thread.
//
// Cancellation is checked before loading, between major processing stages,
// and on each grid row. A synchronous file read itself is not cancellable
// through this flag, but processing stops once that read returns.
Mesh* loadTerrain(const std::string& filename, int requestedLOD) {
    // Do not start loading if shutdown has already been requested.
    if (loaderShouldStop()) return 0;

    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file) return 0;

    char magic[4];             // File signature: TRN1.
    unsigned int levels;      // Number of stored LODs.
    double bounds[4];         // xmin, ymin, xmax, ymax.
    float elevationBounds[2]; // Minimum and maximum elevation.

    // Validate the file header.
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

    // Clamp the requested LOD to the levels available in this file.
    int selected = std::max(
        0, std::min(requestedLOD, (int)levels - 1)
    );

    for (unsigned int level = 0; level < levels; ++level) {
        if (loaderShouldStop()) return 0;

        unsigned int columns, rows; // Core vertex-grid dimensions.

        if (!readUInt(file, columns) || !readUInt(file, rows))
            return 0;

        // Prevent unreasonable allocations from invalid file dimensions.
        if (columns < 2 || rows < 2
            || columns > 4096 || rows > 4096)
            return 0;

        // Elevations include one extra sample around every edge.
        size_t haloCount = (size_t)(columns + 2) * (rows + 2);

        // Stored arrays: east offsets, south offsets, halo elevations.
        size_t bytes = (
            (size_t)columns + rows + haloCount
        ) * sizeof(float);

        if ((int)level != selected) {
            // Skip levels that are not needed.
            file.seekg((std::streamoff)bytes, std::ios::cur);
            if (!file) return 0;
            continue;
        }

        std::vector<float> east(columns);
        std::vector<float> south(rows);
        std::vector<float> heights(haloCount);

        // Read separately so cancellation can be checked between reads.
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

        // Determine nominal sample spacing for halo normal calculations.
        double baseSpacing = east[columns - 1] / (columns - 1);
        if (columns > 2)
            baseSpacing = east[1] - east[0];

        if (baseSpacing <= 0) return 0;

        Mesh* mesh = new Mesh;
        mesh->vertices.resize((size_t)columns * rows);

        // Construct core vertices and their lighting normals.
        for (unsigned int r = 0; r < rows; ++r) {
            // Delete partially built geometry when cancelling.
            if (loaderShouldStop()) {
                delete mesh;
                return 0;
            }

            for (unsigned int c = 0; c < columns; ++c) {
                size_t index = (size_t)r * columns + c;
                size_t h = (
                    (size_t)(r + 1) * (columns + 2)
                    + c + 1
                );

                float height = heights[h];
                Vertex& v = mesh->vertices[index];

                v.x = east[c];
                v.y = finiteFloat(height)
                    ? (float)(height - originZ) : 0;
                v.z = south[r];

                // Neighboring elevation samples, including the halo.
                float left = heights[h - 1];
                float right = heights[h + 1];
                float north = heights[h - (columns + 2)];
                float southHeight = heights[h + (columns + 2)];

                // Distance between west/east neighboring samples.
                double dx =
                    (c == 0
                        ? baseSpacing
                        : east[c] - east[c - 1])
                    + (c + 1 == columns
                        ? baseSpacing
                        : east[c + 1] - east[c]);

                // Distance between north/south neighboring samples.
                double dz =
                    (r == 0
                        ? baseSpacing
                        : south[r] - south[r - 1])
                    + (r + 1 == rows
                        ? baseSpacing
                        : south[r + 1] - south[r]);

                Vec normal(0, 1, 0); // Flat fallback.

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

                v.nx = (float)normal.x;
                v.ny = (float)normal.y;
                v.nz = (float)normal.z;
            }
        }

        if (loaderShouldStop()) {
            delete mesh;
            return 0;
        }

        // Two triangles per cell, three indexes per triangle.
        mesh->indices.reserve(
            (size_t)(columns - 1) * (rows - 1) * 6
        );

        // Construct triangles, excluding those with missing elevations.
        for (unsigned int r = 0; r + 1 < rows; ++r) {
            if (loaderShouldStop()) {
                delete mesh;
                return 0;
            }

            for (unsigned int c = 0; c + 1 < columns; ++c) {
                unsigned int a = r * columns + c; // Northwest.
                unsigned int b = a + 1;           // Northeast.
                unsigned int d = (
                    (r + 1) * columns + c + 1
                );                               // Southeast.
                unsigned int e = d - 1;           // Southwest.

                // Upward-facing triangle winding.
                unsigned int candidates[6] = {
                    a, e, b,
                    b, e, d
                };

                for (int triangle = 0; triangle < 2; ++triangle) {
                    bool valid = true;

                    for (int k = 0; k < 3; ++k) {
                        unsigned int i = candidates[
                            triangle * 3 + k
                        ];
                        unsigned int rr = i / columns;
                        unsigned int cc = i % columns;

                        size_t hi = (
                            (size_t)(rr + 1) * (columns + 2)
                            + cc + 1
                        );

                        if (!finiteFloat(heights[hi]))
                            valid = false;
                    }

                    if (valid) {
                        for (int k = 0; k < 3; ++k) {
                            mesh->indices.push_back(
                                candidates[triangle * 3 + k]
                            );
                        }
                    }
                }
            }
        }

        // Avoid returning a mesh after shutdown or without valid triangles.
        if (loaderShouldStop() || mesh->indices.empty()) {
            delete mesh;
            return 0;
        }

        return mesh;
    }

    return 0;
}


// Load project placement and tile metadata from the lightweight text index.
// Terrain geometry is not read here; it will be streamed later.
bool loadIndex(const std::string& folder) {
    std::ifstream file((folder + "\\terrain_index.txt").c_str()); // Index file.
    if (!file) return false;

    std::string magic; // First-line format identifier.
    std::getline(file, magic);

    // Accept files with either LF or CRLF line endings.
    if (!magic.empty() && magic[magic.size()-1] == '\r')
        magic.erase(magic.size()-1);

    if (magic != "TERRAIN_INDEX_V1") return false;

    unsigned int count; // Number of tile records.

    if (!(file >> originX >> originY >> originZ >> count)
        || count == 0 || count > 100000)
        return false;

    tiles.reserve(count);

    for (unsigned int i = 0; i < count; ++i) {
        Tile tile;            // Metadata for the next tile.
        std::string relative; // Path relative to the terrain project folder.

        if (!(file >> relative
             >> tile.xmin >> tile.ymin >> tile.xmax >> tile.ymax
             >> tile.zmin >> tile.zmax))
            return false;

        // Convert portable index paths into Windows paths.
        std::replace(relative.begin(), relative.end(), '/', '\\');
        tile.filename = folder + "\\" + relative;

        if (tile.xmax <= tile.xmin || tile.ymax <= tile.ymin)
            return false;

        tiles.push_back(tile);
    }

    return true;
}

// ===========================================================================
// Background loading and terrain cache
// ===========================================================================

// Background-thread entry point.
//
// Wait for requests, load a mesh without blocking rendering, and publish the
// result. The tiles vector is not resized after this thread starts.
// Background loader: cancellation is checked before and after each task.

// Load terrain and its corresponding vegetation without OpenGL calls.
DWORD WINAPI loaderMain(void*) {
    while (!loaderShouldStop()) {
        WaitForSingleObject(loaderEvent, 100);

        if (loaderShouldStop()) break;

        Request request;
        bool available = false;

        EnterCriticalSection(&loaderLock);

        if (!requests.empty()) {
            request = requests.front();
            requests.erase(requests.begin());
            available = true;
        }

        LeaveCriticalSection(&loaderLock);

        if (!available) continue;

        Mesh* mesh = loadTerrain(
            tiles[request.tile].filename,
            request.lod
        );

        vegetation::Data* plants = 0;

        if (mesh && !loaderShouldStop()) {
            plants = vegetation::load(
                tiles[request.tile].vegetationFilename,
                loaderShouldStop
            );
        }

        if (loaderShouldStop()) {
            delete mesh;
            delete plants;
            break;
        }

        Result result;
        result.tile = request.tile;
        result.lod = request.lod;
        result.mesh = mesh;
        result.plants = plants;

        EnterCriticalSection(&loaderLock);
        results.push_back(result);
        LeaveCriticalSection(&loaderLock);
    }

    return 0;
}



// Horizontal distance from the camera to the nearest point of a tile.
//
// A camera horizontally inside a tile has distance zero, regardless of height.
// This is used for streaming; frustum culling separately decides visibility.
double tileDistance(const Tile& tile) {
    double x0 = tile.xmin - originX; // Western boundary in viewer coordinates.
    double x1 = tile.xmax - originX; // Eastern boundary.
    double z0 = originY - tile.ymax; // Northern boundary.
    double z1 = originY - tile.ymin; // Southern boundary.

    double dx = std::max(x0 - camera.x, std::max(0.0, camera.x - x1));
    double dz = std::max(z0 - camera.z, std::max(0.0, camera.z - z1));
    // dx and dz are distances outside each horizontal interval.

    return std::sqrt(dx*dx + dz*dz);
}

// Resident terrain plus vegetation CPU storage.
// Shared textures and temporary frame batches are not included.
size_t cacheBytes() {
    size_t bytes = 0;

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (tiles[i].mesh)
            bytes += tiles[i].mesh->bytes();

        if (tiles[i].plants)
            bytes += tiles[i].plants->bytes();
    }

    return bytes;
}


// Accept completed meshes, evict distant terrain, enforce the cache budget,
// and enqueue nearby tiles that need loading or a different detail level.
void updateStreaming() {
    std::vector<Result> completed; // Results owned locally by the main thread.

    EnterCriticalSection(&loaderLock);
    completed.swap(results);
    LeaveCriticalSection(&loaderLock);

    for (size_t i = 0; i < completed.size(); ++i) {
        Result& result = completed[i]; // One completed background task.
        Tile& tile = tiles[result.tile]; // Destination tile.

        tile.pending = false;

                if (!result.mesh) {
            delete result.plants;
            tile.failed = true;
            continue;
        }

        if (tileDistance(tile) > unloadRadius) {
            delete result.mesh;
            delete result.plants;
            continue;
        }

        delete tile.mesh;
        delete tile.plants;

        tile.mesh = result.mesh;
        tile.plants = result.plants;
        tile.loadedLOD = result.lod;

    }

    // Unload beyond the larger radius, rather than immediately at loadRadius.
    // This avoids repeated loads when moving near a radius boundary.
    for (size_t i = 0; i < tiles.size(); ++i) {
        Tile& tile = tiles[i]; // Tile being considered for distance eviction.

        if (tile.mesh && tileDistance(tile) > unloadRadius) {
            delete tile.mesh;
            tile.mesh = 0;
            tile.loadedLOD = -1;
        }

    }

    size_t bytes = cacheBytes(); // Current resident mesh storage.

    // If over budget, remove the most distant resident tiles first.
    while (bytes > memoryBudget) {
        int farthest = -1;            // Selected eviction candidate.
        double farthestDistance = -1; // Its distance from the camera.

        for (size_t i = 0; i < tiles.size(); ++i) {
            if (tiles[i].mesh) {
                double distance = tileDistance(tiles[i]); // Candidate distance.

                if (distance > farthestDistance) {
                    farthestDistance = distance;
                    farthest = (int)i;
                }
            }
        }

        if (farthest < 0) break;

        bytes -= tiles[farthest].mesh->bytes();

        if (tiles[farthest].plants)
            bytes -= tiles[farthest].plants->bytes();

        delete tiles[farthest].mesh;
        delete tiles[farthest].plants;

        tiles[farthest].mesh = 0;
        tiles[farthest].plants = 0;
        tiles[farthest].loadedLOD = -1;

    }

    int best = -1;              // Nearest tile needing a request.
    double bestDistance = 1e100; // Best request distance found so far.

    for (size_t i = 0; i < tiles.size(); ++i) {
        Tile& tile = tiles[i];              // Candidate tile.
        double distance = tileDistance(tile); // Horizontal proximity.

        int lod = lodBias; // Preset-specific starting detail.
        if (distance > 200) ++lod;
        if (distance > 450) ++lod;
        if (distance > 800) ++lod;
        lod = std::min(lod, 4);

        tile.wantedLOD = lod;

        bool needs = !tile.mesh || tile.loadedLOD != lod;
        // A missing mesh or different desired detail requires loading.

        if (distance <= loadRadius && needs
            && !tile.pending && !tile.failed
            && distance < bestDistance) {
            bestDistance = distance;
            best = (int)i;
        }
    }

    // Keep outstanding work short so old camera positions do not create a
    // large backlog. There can be one active task and one queued task.
    EnterCriticalSection(&loaderLock);

    int pendingCount = 0; // Queued plus in-progress tile requests.

    for (size_t i = 0; i < tiles.size(); ++i)
        if (tiles[i].pending) ++pendingCount;

    if (best >= 0 && pendingCount < 2) {
        Request request; // New task for the background worker.
        request.tile = best;
        request.lod = tiles[best].wantedLOD;

        tiles[best].pending = true;
        requests.push_back(request);
        SetEvent(loaderEvent);
    }

    LeaveCriticalSection(&loaderLock);
}

// ===========================================================================
// Camera functions
// ===========================================================================

// Convert yaw and pitch into the camera's forward direction.
Vec forwardVector() {
    return Vec(
        std::sin(yaw) * std::cos(pitch),
        std::sin(pitch),
        -std::cos(yaw) * std::cos(pitch)
    );
}

// Reposition the orbit camera using target, orientation, and distance.
void updateOrbitCamera() {
    camera = target - forwardVector() * orbitDistance;
}

// Reset to a local overview near the center of the entire project.
// This intentionally does not attempt to show all tiles at once.
void homeCamera() {
    double xmin = tiles[0].xmin; // Overall minimum easting.
    double xmax = tiles[0].xmax; // Overall maximum easting.
    double ymin = tiles[0].ymin; // Overall minimum northing.
    double ymax = tiles[0].ymax; // Overall maximum northing.
    double highest = tiles[0].zmax; // Highest elevation in the project.

    for (size_t i = 1; i < tiles.size(); ++i) {
        xmin = std::min(xmin, tiles[i].xmin);
        xmax = std::max(xmax, tiles[i].xmax);
        ymin = std::min(ymin, tiles[i].ymin);
        ymax = std::max(ymax, tiles[i].ymax);
        highest = std::max(highest, tiles[i].zmax);
    }

    target = Vec(
        (xmin+xmax)*0.5 - originX,
        highest - originZ,
        originY - (ymin+ymax)*0.5
    );

    orbitDistance = loadRadius * 0.8;
    yaw = 0;
    pitch = -0.55;
    updateOrbitCamera();
}

// Switch modes while keeping the current camera position and orientation.
void switchCamera() {
    if (orbitMode) {
        // Orbit camera already has a valid position for free flight.
        orbitMode = false;
    } else {
        // Place a new orbit target in front of the current flight camera.
        target = camera + forwardVector() * orbitDistance;
        orbitMode = true;
    }
}

// Apply held keyboard controls for this frame.
// dt is elapsed frame time in seconds.
void updateCamera(double dt) {
    double speed = movementSpeed * dt; // Movement distance for this frame.
    if (keys[VK_SHIFT]) speed *= 4;

    Vec forward = forwardVector(); // Current camera-forward direction.
    Vec right(std::cos(yaw), 0, std::sin(yaw)); // Horizontal right direction.
    Vec movement; // Combined movement requested by all held keys.

    if (keys['W']) movement = movement + forward * speed;
    if (keys['S']) movement = movement - forward * speed;
    if (keys['D']) movement = movement + right * speed;
    if (keys['A']) movement = movement - right * speed;
    if (keys['E']) movement.y += speed;
    if (keys['Q']) movement.y -= speed;

    if (orbitMode) {
        target = target + movement;
        updateOrbitCamera();
    } else {
        camera = camera + movement;
    }
}

// ===========================================================================
// Frustum culling
// ===========================================================================

// Six clipping planes: left, right, bottom, top, near, and far.
// Each plane contains coefficients A, B, C, D for Ax + By + Cz + D = 0.
double planes[6][4];

// Extract the view frustum from the current OpenGL projection/view matrices.
//
// The view matrix contains rotation only. Terrain is rendered relative to the
// camera, which avoids sending large map coordinates to OpenGL.
void extractFrustum() {
    double projection[16]; // Perspective projection matrix.
    double view[16];       // Camera rotation matrix.
    double clip[16];       // Combined projection * view matrix.

    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetDoublev(GL_MODELVIEW_MATRIX, view);

    // OpenGL stores matrices in column-major order.
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row) {
            clip[column*4+row] = 0;

            for (int k = 0; k < 4; ++k)
                clip[column*4+row] +=
                    projection[k*4+row] * view[column*4+k];
        }

    for (int p = 0; p < 6; ++p) {
        int axis = p / 2; // X, Y, or Z clipping pair.
        double sign = (p % 2 == 0) ? 1 : -1; // Side of that pair.

        for (int k = 0; k < 4; ++k)
            planes[p][k] = clip[k*4+3] + sign*clip[k*4+axis];

        double length = std::sqrt(
            planes[p][0]*planes[p][0]
            + planes[p][1]*planes[p][1]
            + planes[p][2]*planes[p][2]
        ); // Plane-normal length.

        if (length > 0)
            for (int k = 0; k < 4; ++k)
                planes[p][k] /= length;
    }
}

// Return false if a tile's 3D bounding box lies outside any frustum plane.
// Intersecting boxes are retained; individual triangles are clipped by OpenGL.
bool visible(const Tile& tile) {
    double x0 = tile.xmin-originX-camera.x; // Camera-relative western bound.
    double x1 = tile.xmax-originX-camera.x; // Eastern bound.
    double y0 = tile.zmin-originZ-camera.y; // Lowest elevation.
    double y1 = tile.zmax-originZ-camera.y; // Highest elevation.
    double z0 = originY-tile.ymax-camera.z; // Northern bound.
    double z1 = originY-tile.ymin-camera.z; // Southern bound.

    for (int p = 0; p < 6; ++p) {
        // Choose the box corner farthest toward the plane's inward normal.
        double x = planes[p][0] >= 0 ? x1 : x0;
        double y = planes[p][1] >= 0 ? y1 : y0;
        double z = planes[p][2] >= 0 ? z1 : z0;

        if (planes[p][0]*x + planes[p][1]*y
            + planes[p][2]*z + planes[p][3] < 0)
            return false;
    }

    return true;
}

// ===========================================================================
// Text and terrain rendering
// ===========================================================================

// Draw a string using the bitmap-font display lists.
// The caller establishes the screen-space projection and text color.
void text(int x, int y, const char* value) {
    glRasterPos2i(x, y);
    glListBase(fontBase);
    glCallLists((GLsizei)std::strlen(value), GL_UNSIGNED_BYTE, value);
}

// Draw statistics and optional help in a 2D screen-space overlay.
void drawOverlay() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, viewportWidth, viewportHeight, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    char status[256]; // Formatted statistics line.

    std::sprintf(
        status,
        "%s | %.0f FPS | %d tiles | %lu triangles | %.1f MB cache",
        orbitMode ? "Orbit" : "Flight",
        fps, drawnTiles, drawnTriangles,
        cacheBytes() / (1024.0*1024.0)
    );

    text(12, 22, status);

    char vegetationStatus[128];
    std::sprintf(
        vegetationStatus,
        "Vegetation: %s | %lu plants drawn",
        vegetation::enabled ? "on" : "off",
        vegetation::drawnPlants
    );
    text(12, 150, vegetationStatus);


    if (helpVisible) {
        text(12, 46, "Tab: camera mode    F1: help    F2: wireframe F3 vegetation");
        text(12, 66, "WASD: move    Q/E: down/up    Shift: faster");
        text(12, 86, "Flight: right-drag look; wheel changes speed");
        text(12, 106, "Orbit: left-drag rotate; right-drag pan; wheel zoom");
        text(12, 126, "Home: reset view    Esc: exit");
    }

    glEnable(GL_DEPTH_TEST);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// Render one frame: camera, lighting, visible terrain, target marker, and help.
void render() {
    glViewport(0, 0, viewportWidth, viewportHeight);
    glClearColor(0.25f, 0.32f, 0.40f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    double nearPlane = 1.0; // Near clipping distance in metres.
    double halfHeight = nearPlane * std::tan(60.0 * PI / 360.0);
    // halfHeight corresponds to a 60-degree vertical field of view.
    double halfWidth = halfHeight * viewportWidth / viewportHeight;
    // Horizontal extent incorporates the window aspect ratio.

    glFrustum(
        -halfWidth, halfWidth, -halfHeight, halfHeight,
        nearPlane, farPlane
    );

    Vec forward = unit(forwardVector()); // Camera-forward unit vector.
    Vec right = unit(cross(forward, Vec(0, 1, 0))); // Camera-right vector.
    Vec up = cross(right, forward); // Camera-up vector.

    // Rotation-only camera view matrix.
    // Translation is applied per tile using camera-relative coordinates.
    double view[16] = {
        right.x, up.x, -forward.x, 0,
        right.y, up.y, -forward.y, 0,
        right.z, up.z, -forward.z, 0,
        0, 0, 0, 1
    };

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(view);
    extractFrustum();

    GLfloat lightPosition[] = { -0.4f, 0.8f, 0.3f, 0 };
    // W=0 makes this a directional light rather than a local point light.

    GLfloat diffuse[] = { 0.85f, 0.85f, 0.80f, 1 };
    // Direct sunlight color/intensity.

    GLfloat ambient[] = { 0.30f, 0.30f, 0.32f, 1 };
    // Ambient illumination keeps shaded terrain from becoming black.

    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);

    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColor3f(0.65f, 0.67f, 0.61f);

    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    drawnTiles = 0;
    drawnTriangles = 0;

    for (size_t i = 0; i < tiles.size(); ++i) {
        Tile& tile = tiles[i]; // Candidate tile for this frame.

        if (!tile.mesh || !visible(tile)) continue;

        Mesh& mesh = *tile.mesh; // Its currently resident detail level.

        glPushMatrix();

        // Move tile-local vertices into camera-relative viewer coordinates.
        glTranslated(
            tile.xmin-originX-camera.x,
            -camera.y,
            originY-tile.ymax-camera.z
        );

        // Interleaved CPU arrays: position and normal are in the same struct.
        glVertexPointer(
            3, GL_FLOAT, sizeof(Vertex), &mesh.vertices[0].x
        );
        glNormalPointer(
            GL_FLOAT, sizeof(Vertex), &mesh.vertices[0].nx
        );

        glDrawElements(
            GL_TRIANGLES, (GLsizei)mesh.indices.size(),
            GL_UNSIGNED_INT, &mesh.indices[0]
        );

        glPopMatrix();

        ++drawnTiles;
        drawnTriangles += (unsigned long)(mesh.indices.size()/3);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Collect vegetation independently of the terrain bounding-box test:
    // a tall tree may be visible even when the ground beneath it is not.
    vegetation::beginFrame();

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (!tiles[i].plants) continue;

        vegetation::collect(
            tiles[i].plants,
            camera.x, camera.y, camera.z,
            originX, originY, originZ,
            planes
        );
    }

    vegetation::draw();


    if (orbitMode) {
        // Draw a small orange cross at the orbit target.
        // Depth testing remains enabled, so terrain may obscure it.
        glDisable(GL_LIGHTING);
        glColor3f(1, 0.5f, 0.1f);

        Vec p = target-camera; // Camera-relative marker position.
        double size = std::max(1.0, orbitDistance*0.004); // Marker half-size.

        glBegin(GL_LINES);
        glVertex3d(p.x-size, p.y, p.z);
        glVertex3d(p.x+size, p.y, p.z);
        glVertex3d(p.x, p.y-size, p.z);
        glVertex3d(p.x, p.y+size, p.z);
        glEnd();
    }

    drawOverlay();
    SwapBuffers(deviceContext);
}

// ===========================================================================
// Main-window input handling
// ===========================================================================

// Windows callback for resizing, keyboard input, mouse input, and closing.
// hwnd: receiving window.
// message: event identifier.
// wParam/lParam: event-specific data supplied by Windows.
LRESULT CALLBACK windowProc(HWND hwnd, UINT message,
                            WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CLOSE:
        // Let the main loop perform orderly thread/OpenGL cleanup.
        running = false;
        return 0;

    case WM_SIZE:
        viewportWidth = std::max(1, (int)LOWORD(lParam));
        viewportHeight = std::max(1, (int)HIWORD(lParam));
        return 0;

    case WM_KILLFOCUS:
        // Avoid stuck movement keys after Alt-Tab or clicking another window.
        for (int i = 0; i < 256; ++i) keys[i] = false;
        leftDragging = rightDragging = false;
        ReleaseCapture();
        return 0;

    case WM_KEYDOWN:
        if (wParam < 256) keys[wParam] = true;

        // Bit 30 indicates that the key was already down.
        // Ignore repeats for toggle actions, but retain movement-key state.
        if (!(lParam & (1L << 30))) {
            if (wParam == VK_ESCAPE) running = false;
            if (wParam == VK_F1) helpVisible = !helpVisible;
            if (wParam == VK_F2) wireframe = !wireframe;
            if (wParam == VK_TAB) switchCamera();
            if (wParam == VK_HOME) homeCamera();
            if (wParam == VK_F3) vegetation::enabled = !vegetation::enabled;

        }
        return 0;

    case WM_KEYUP:
        if (wParam < 256) keys[wParam] = false;
        return 0;

    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
        if (message == WM_LBUTTONDOWN) leftDragging = true;
        else rightDragging = true;

        mouseX = (short)LOWORD(lParam);
        mouseY = (short)HIWORD(lParam);

        // Continue receiving drag events even outside the window.
        SetCapture(hwnd);
        return 0;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
        if (message == WM_LBUTTONUP) leftDragging = false;
        else rightDragging = false;

        if (!leftDragging && !rightDragging) ReleaseCapture();
        return 0;

    case WM_MOUSEMOVE: {
        int x = (short)LOWORD(lParam); // New mouse X.
        int y = (short)HIWORD(lParam); // New mouse Y.
        int dx = x-mouseX;            // Horizontal movement since last event.
        int dy = y-mouseY;            // Vertical movement since last event.

        mouseX = x;
        mouseY = y;

        if ((!orbitMode && rightDragging)
            || (orbitMode && leftDragging)) {
            // Rotate camera orientation.
            yaw += dx*0.005;
            pitch -= dy*0.005;

            // Prevent looking exactly vertical, where the right vector
            // would become undefined.
            pitch = std::max(-1.50, std::min(1.50, pitch));
        } else if (orbitMode && rightDragging) {
            Vec right(std::cos(yaw), 0, std::sin(yaw)); // Pan-right direction.
            Vec up = unit(cross(right, forwardVector())); // Pan-up direction.
            double scale = orbitDistance*0.0015; // Metres per mouse pixel.

            target = target-right*(dx*scale)+up*(dy*scale);
        }

        if (orbitMode) updateOrbitCamera();
        return 0;
    }

    case WM_MOUSEWHEEL: {
        double steps = (short)HIWORD(wParam)/120.0; // Signed wheel increments.

        if (orbitMode) {
            orbitDistance *= std::pow(0.85, steps);
            orbitDistance = std::max(5.0, std::min(5000.0, orbitDistance));
            updateOrbitCamera();
        } else {
            movementSpeed *= std::pow(1.2, steps);
            movementSpeed = std::max(1.0, std::min(2000.0, movementSpeed));
        }
        return 0;
    }

    case WM_ERASEBKGND:
        // OpenGL clears the background; Windows need not clear it separately.
        return 1;
    }

    return DefWindowProcA(hwnd, message, wParam, lParam);
}

// ===========================================================================
// Startup graphics selector
// ===========================================================================

int selectedPreset = -1; // -1: waiting; -2: cancelled; 0/1/2: Low/Medium/High.

// Handle button presses and closing of the startup selector.
LRESULT CALLBACK launcherProc(HWND hwnd, UINT message,
                              WPARAM wParam, LPARAM lParam) {
    if (message == WM_COMMAND) {
        int id = LOWORD(wParam); // Identifier of the clicked control.

        if (id >= 101 && id <= 103) {
            selectedPreset = id-101;
            DestroyWindow(hwnd);
        }
        return 0;
    }

    if (message == WM_CLOSE) {
        selectedPreset = -2;
        DestroyWindow(hwnd);
        return 0;
    }

    return DefWindowProcA(hwnd, message, wParam, lParam);
}

// Create a small native Windows graphics-preset dialog.
// Return 0, 1, or 2; return a negative value if cancelled or creation fails.
int choosePreset(HINSTANCE instance) {
    WNDCLASSA wc; // Window-class description for the launcher.
    ZeroMemory(&wc, sizeof(wc));

    wc.lpfnWndProc = launcherProc;
    wc.hInstance = instance;
    wc.lpszClassName = "TerrainLauncher";
    wc.hCursor = LoadCursor(0, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE+1);
    RegisterClassA(&wc);

    HWND launcher = CreateWindowA(
        wc.lpszClassName, "Terrain viewer - graphics",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
        CW_USEDEFAULT, CW_USEDEFAULT, 470, 180,
        0, 0, instance, 0
    ); // Startup selector window.

    if (!launcher) return -2;

    CreateWindowA(
        "STATIC", "Choose a preset. Start with Low on older hardware.",
        WS_CHILD | WS_VISIBLE,
        20, 20, 420, 25, launcher, 0, instance, 0
    );

    const char* labels[] = { "Low", "Medium", "High" }; // Button captions.

    for (int i = 0; i < 3; ++i)
        CreateWindowA(
            "BUTTON", labels[i], WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            20+i*145, 65, 130, 35,
            launcher, (HMENU)(INT_PTR)(101+i), instance, 0
        );

    ShowWindow(launcher, SW_SHOW);

    MSG message; // Windows event received while waiting for a selection.

    while (selectedPreset == -1 && GetMessage(&message, 0, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessage(&message);
    }

    return selectedPreset;
}

// ===========================================================================
// OpenGL initialization
// ===========================================================================

// Create the main window, choose its pixel format, create the OpenGL context,
// and build a bitmap font. All OpenGL work stays on the main thread.
bool initializeOpenGL(HINSTANCE instance) {
    WNDCLASSA wc; // Main-window class definition.
    ZeroMemory(&wc, sizeof(wc));

    // CS_OWNDC gives this window a stable drawing-device context.
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = windowProc;
    wc.hInstance = instance;
    wc.lpszClassName = "LegacyTerrainViewer";
    wc.hCursor = LoadCursor(0, IDC_ARROW);

    if (!RegisterClassA(&wc)) return false;

    windowHandle = CreateWindowA(
        wc.lpszClassName, "Terrain viewer",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        viewportWidth, viewportHeight,
        0, 0, instance, 0
    );

    if (!windowHandle) return false;

    deviceContext = GetDC(windowHandle);

    PIXELFORMATDESCRIPTOR format; // Requested framebuffer configuration.
    ZeroMemory(&format, sizeof(format));

    format.nSize = sizeof(format);
    format.nVersion = 1;
    format.dwFlags = PFD_DRAW_TO_WINDOW
                   | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    format.iPixelType = PFD_TYPE_RGBA;
    format.cColorBits = 24;
    format.cDepthBits = 24;
    format.iLayerType = PFD_MAIN_PLANE;

    int selected = ChoosePixelFormat(deviceContext, &format);
    // selected identifies the closest supported framebuffer format.

    if (!selected || !SetPixelFormat(deviceContext, selected, &format))
        return false;

    renderingContext = wglCreateContext(deviceContext);

    if (!renderingContext
        || !wglMakeCurrent(deviceContext, renderingContext))
        return false;

    glEnable(GL_DEPTH_TEST); // Hide terrain behind nearer terrain.
    glEnable(GL_CULL_FACE);  // Skip back-facing triangles.
    glEnable(GL_NORMALIZE);  // Ensure lighting normals remain unit length.
    glShadeModel(GL_SMOOTH); // Interpolate lighting across triangle vertices.

    // Build one bitmap display list for each byte-sized character.
    fontBase = glGenLists(256);

    HFONT font = CreateFontA(
        -15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY, FIXED_PITCH, "Courier New"
    ); // Temporary Windows font used to generate OpenGL glyphs.

    HGDIOBJ oldFont = SelectObject(deviceContext, font);
    // Preserve the device context's previous font.

    BOOL fontOK = wglUseFontBitmapsA(deviceContext, 0, 256, fontBase);
    // Indicates whether glyph display-list generation succeeded.

    SelectObject(deviceContext, oldFont);
    DeleteObject(font);

    if (!fontOK) return false;

    ShowWindow(windowHandle, SW_SHOW);
    return true;
}

// ===========================================================================
// Application entry point
// ===========================================================================

// Windows GUI entry point.
//
// instance: executable module handle.
// commandLine: arguments after the executable name.
// The unnamed parameters are unused previous-instance and show-state values.
int WINAPI WinMain(HINSTANCE instance, HINSTANCE,
                   LPSTR commandLine, int) {
    // Accept one terrain-folder argument, optionally enclosed in quotes.
    terrainFolder = commandLine;

    while (!terrainFolder.empty() && terrainFolder[0] == ' ')
        terrainFolder.erase(0, 1);

    while (!terrainFolder.empty()
           && terrainFolder[terrainFolder.size()-1] == ' ')
        terrainFolder.erase(terrainFolder.size()-1);

    if (terrainFolder.size() >= 2
        && terrainFolder[0] == '"'
        && terrainFolder[terrainFolder.size()-1] == '"')
        terrainFolder = terrainFolder.substr(1, terrainFolder.size()-2);

    // With no argument, look for a terrain folder in the working directory.
    if (terrainFolder.empty()) terrainFolder = tilesLocation;

    if (!loadIndex(terrainFolder)) {
        MessageBoxA(
            0,
            "Cannot read terrain_index.txt.\n\n"
            "Launch with the prepared terrain folder as an argument.",
            "Terrain viewer", MB_OK | MB_ICONERROR
        );
        return 1;
    }

    int preset = choosePreset(instance); // User's selected graphics level.
    if (preset < 0) return 0;

    // Presets adjust horizontal streaming range, visible range, CPU cache,
    // and minimum detail. These are starting values, not hardware detection.
    if (preset == 0) {
        loadRadius = 450;
        unloadRadius = 600;
        farPlane = 800;
        memoryBudget = 48 * 1024 * 1024;
        lodBias = 2;
    } else if (preset == 1) {
        loadRadius = 750;
        unloadRadius = 950;
        farPlane = 1200;
        memoryBudget = 96 * 1024 * 1024;
        lodBias = 1;
    } else {
        loadRadius = 1100;
        unloadRadius = 1400;
        farPlane = 1700;
        memoryBudget = 192 * 1024 * 1024;
        lodBias = 0;
    }

    if (!initializeOpenGL(instance)) {
        MessageBoxA(
            0, "OpenGL initialization failed.",
            "Terrain viewer", MB_OK | MB_ICONERROR
        );
        return 1;
    }

    // Textures must be created on the main thread with a current GL context.
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

        // Terrain can still run without vegetation.
        vegetation::enabled = false;
    }

    // Match by original terrain basename, not position in the tile vector.
    for (size_t i = 0; i < tiles.size(); ++i) {
        tiles[i].vegetationFilename =
            vegetation::filenameForTerrain(tiles[i].filename);
    }


    // Initialize synchronization before starting the worker.
    InitializeCriticalSection(&loaderLock);
    loaderEvent = CreateEventA(0, FALSE, FALSE, 0);

    if (loaderEvent)
        loaderThread = CreateThread(0, 0, loaderMain, 0, 0, 0);

    if (!loaderEvent || !loaderThread) {
        MessageBoxA(
            0, "Cannot start the background loader.",
            "Terrain viewer", MB_OK | MB_ICONERROR
        );

        if (loaderEvent) CloseHandle(loaderEvent);
        DeleteCriticalSection(&loaderLock);
        return 1;
    }

    homeCamera();

    LARGE_INTEGER frequency; // High-resolution timer ticks per second.
    LARGE_INTEGER previous;  // Time of the previous frame.
    LARGE_INTEGER now;       // Current frame timestamp.

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&previous);

    double fpsTime = 0; // Accumulated time for the FPS display.
    int fpsFrames = 0; // Frames accumulated during that interval.

    while (running) {
        MSG message; // Next queued Windows event.

        // Process all currently queued input/window events without blocking.
        while (PeekMessage(&message, 0, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) running = false;
            TranslateMessage(&message);
            DispatchMessage(&message);
        }

        if (!running) break;

        QueryPerformanceCounter(&now);

        double dt = (double)(now.QuadPart-previous.QuadPart)
                  / frequency.QuadPart; // Frame duration in seconds.

        previous = now;

        // Prevent huge camera jumps after a long stall or debugger pause.
        dt = std::min(dt, 0.1);

        updateCamera(dt);
        updateStreaming();
        render();

        fpsTime += dt;
        ++fpsFrames;

        if (fpsTime >= 0.5) {
            fps = fpsFrames/fpsTime;
            fpsTime = 0;
            fpsFrames = 0;
        }

        // Yield CPU time to Windows and the background loader.
        Sleep(1);
    }

    // Request cooperative cancellation and wake the worker.
    InterlockedExchange(&loaderStop, 1);

    EnterCriticalSection(&loaderLock);
    requests.clear();
    LeaveCriticalSection(&loaderLock);

    SetEvent(loaderEvent);

    // Wait for the worker, but continue servicing Windows messages.
    // Do not free shared data until the worker has actually exited.
    for (;;) {
        DWORD waitResult = MsgWaitForMultipleObjects(
            1,
            &loaderThread,
            FALSE,
            INFINITE,
            QS_ALLINPUT
        );

        if (waitResult == WAIT_OBJECT_0) {
            // Worker has exited; normal cleanup can safely continue.
            break;
        }

        if (waitResult == WAIT_OBJECT_0 + 1) {
            MSG message;

            while (PeekMessage(&message, 0, 0, 0, PM_REMOVE)) {
                // Ignore quit messages here; shutdown is already underway.
                if (message.message != WM_QUIT) {
                    TranslateMessage(&message);
                    DispatchMessage(&message);
                }
            }
        } else {
            // An unexpected wait failure means safe cleanup cannot be
            // guaranteed. End the process instead of freeing live data.
            ExitProcess(1);
        }
    }


    CloseHandle(loaderThread);
    CloseHandle(loaderEvent);

    // Free completed results that were never adopted into the tile cache.
        for (size_t i = 0; i < results.size(); ++i) {
        delete results[i].mesh;
        delete results[i].plants;
    }

    for (size_t i = 0; i < tiles.size(); ++i) {
        delete tiles[i].mesh;
        delete tiles[i].plants;
    }


    DeleteCriticalSection(&loaderLock);

    // Delete OpenGL resources while the context is still current.
    if (fontBase) glDeleteLists(fontBase, 256);


    // Delete shared vegetation textures before releasing OpenGL.
    vegetation::shutdown();

    // Release the context and the Windows drawing resources.
    wglMakeCurrent(0, 0);
    wglDeleteContext(renderingContext);
    ReleaseDC(windowHandle, deviceContext);
    DestroyWindow(windowHandle);

    return 0;
}
