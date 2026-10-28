#define WINVER 0x0501
#define _WIN32_WINNT 0x0501
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <GL/gl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

const double PI = 3.14159265358979323846;

struct Vec {
    double x, y, z;

    Vec(double X = 0, double Y = 0, double Z = 0)
        : x(X), y(Y), z(Z) {}

    Vec operator+(const Vec& b) const {
        return Vec(x + b.x, y + b.y, z + b.z);
    }

    Vec operator-(const Vec& b) const {
        return Vec(x - b.x, y - b.y, z - b.z);
    }

    Vec operator*(double s) const {
        return Vec(x * s, y * s, z * s);
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

Vec unit(const Vec& a) {
    double length = std::sqrt(dot(a, a));
    return length > 1e-12 ? a * (1.0 / length) : Vec(0, 1, 0);
}

bool finiteFloat(float value) {
    // Portable enough for the finite elevations expected in these files.
    return value == value && value > -1e30f && value < 1e30f;
}

// ---------------------------------------------------------------------------
// Terrain structures
// ---------------------------------------------------------------------------

struct Vertex {
    float x, y, z;
    float nx, ny, nz;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    size_t bytes() const {
        return vertices.capacity() * sizeof(Vertex)
             + indices.capacity() * sizeof(unsigned int);
    }
};

struct Tile {
    std::string filename;
    double xmin, ymin, xmax, ymax, zmin, zmax;
    Mesh* mesh;
    int loadedLOD;
    int wantedLOD;
    bool pending;
    bool failed;

    Tile()
        : mesh(0), loadedLOD(-1), wantedLOD(-1),
          pending(false), failed(false) {}
};

struct Request {
    int tile;
    int lod;
};

struct Result {
    int tile;
    int lod;
    Mesh* mesh;
};

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

HWND windowHandle = 0;
HDC deviceContext = 0;
HGLRC renderingContext = 0;
GLuint fontBase = 0;

int viewportWidth = 1100;
int viewportHeight = 750;

bool running = true;
bool helpVisible = true;
bool wireframe = false;
bool orbitMode = true;

bool leftDragging = false;
bool rightDragging = false;
int mouseX = 0;
int mouseY = 0;

bool keys[256] = { false };

Vec camera;
Vec target;
double yaw = 0;
double pitch = -0.5;
double orbitDistance = 800;
double movementSpeed = 80;

std::vector<Tile> tiles;
std::string terrainFolder;
double originX = 0;
double originY = 0;
double originZ = 0;

double loadRadius = 650;
double unloadRadius = 850;
double farPlane = 1200;
size_t memoryBudget = 96 * 1024 * 1024;
int lodBias = 1;

int drawnTiles = 0;
unsigned long drawnTriangles = 0;
double fps = 0;

CRITICAL_SECTION loaderLock;
HANDLE loaderEvent = 0;
HANDLE loaderThread = 0;
volatile LONG loaderStop = 0;

std::vector<Request> requests;
std::vector<Result> results;

// ---------------------------------------------------------------------------
// File loading
// ---------------------------------------------------------------------------

bool readBytes(std::ifstream& file, void* data, size_t bytes) {
    file.read((char*)data, (std::streamsize)bytes);
    return !!file;
}

bool readUInt(std::ifstream& file, unsigned int& value) {
    return readBytes(file, &value, 4);
}

// The format is little-endian. The intended 32-bit Windows target is x86.
Mesh* loadTerrain(const std::string& filename, int requestedLOD) {
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
        || !readBytes(file, elevationBounds, sizeof(elevationBounds))) {
        return 0;
    }

    int selected = std::min(requestedLOD, (int)levels - 1);

    for (unsigned int level = 0; level < levels; ++level) {
        unsigned int columns, rows;

        if (!readUInt(file, columns) || !readUInt(file, rows))
            return 0;

        // Bound input sizes to prevent unreasonable allocations.
        if (columns < 2 || rows < 2
            || columns > 4096 || rows > 4096)
            return 0;

        size_t haloCount = (size_t)(columns + 2) * (rows + 2);
        size_t bytes = ((size_t)columns + rows + haloCount) * 4;

        if ((int)level != selected) {
            file.seekg((std::streamoff)bytes, std::ios::cur);
            if (!file) return 0;
            continue;
        }

        std::vector<float> east(columns);
        std::vector<float> south(rows);
        std::vector<float> heights(haloCount);

        if (!readBytes(file, &east[0], columns * 4)
            || !readBytes(file, &south[0], rows * 4)
            || !readBytes(file, &heights[0], haloCount * 4))
            return 0;

        Mesh* mesh = new Mesh;
        mesh->vertices.resize((size_t)columns * rows);

        // The preprocessor's halo sample distance is the nominal LOD stride.
        double baseSpacing = (east[columns - 1] / (columns - 1));
        if (columns > 2)
            baseSpacing = east[1] - east[0];

        if (baseSpacing <= 0) {
            delete mesh;
            return 0;
        }

        for (unsigned int r = 0; r < rows; ++r) {
            for (unsigned int c = 0; c < columns; ++c) {
                size_t index = (size_t)r * columns + c;
                size_t h = (size_t)(r + 1) * (columns + 2) + c + 1;

                float height = heights[h];

                Vertex& v = mesh->vertices[index];
                v.x = east[c];
                v.y = finiteFloat(height)
                    ? (float)(height - originZ) : 0;
                v.z = south[r];

                float left = heights[h - 1];
                float right = heights[h + 1];
                float north = heights[h - (columns + 2)];
                float southHeight = heights[h + (columns + 2)];

                double dx = (c == 0 ? baseSpacing : east[c] - east[c-1])
                          + (c + 1 == columns
                             ? baseSpacing : east[c+1] - east[c]);

                double dz = (r == 0 ? baseSpacing : south[r] - south[r-1])
                          + (r + 1 == rows
                             ? baseSpacing : south[r+1] - south[r]);

                Vec normal(0, 1, 0);

                if (finiteFloat(left) && finiteFloat(right)
                    && finiteFloat(north) && finiteFloat(southHeight)) {
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

        mesh->indices.reserve(
            (size_t)(columns - 1) * (rows - 1) * 6
        );

        // Only emit triangles whose source elevations are all valid.
        for (unsigned int r = 0; r + 1 < rows; ++r) {
            for (unsigned int c = 0; c + 1 < columns; ++c) {
                unsigned int a = r * columns + c;
                unsigned int b = a + 1;
                unsigned int d = (r + 1) * columns + c + 1;
                unsigned int e = d - 1;

                unsigned int candidates[6] = { a, e, b, b, e, d };

                for (int triangle = 0; triangle < 2; ++triangle) {
                    bool valid = true;

                    for (int k = 0; k < 3; ++k) {
                        unsigned int i = candidates[triangle * 3 + k];
                        unsigned int rr = i / columns;
                        unsigned int cc = i % columns;
                        size_t hi = (size_t)(rr + 1) * (columns + 2)
                                  + cc + 1;

                        if (!finiteFloat(heights[hi])) valid = false;
                    }

                    if (valid) {
                        for (int k = 0; k < 3; ++k)
                            mesh->indices.push_back(
                                candidates[triangle * 3 + k]
                            );
                    }
                }
            }
        }

        if (mesh->indices.empty()) {
            delete mesh;
            return 0;
        }

        return mesh;
    }

    return 0;
}

bool loadIndex(const std::string& folder) {
    std::ifstream file((folder + "\\terrain_index.txt").c_str());
    if (!file) return false;

    std::string magic;
    std::getline(file, magic);

    // Accept a possible CR from CRLF files.
    if (!magic.empty() && magic[magic.size()-1] == '\r')
        magic.erase(magic.size()-1);

    if (magic != "TERRAIN_INDEX_V1") return false;

    unsigned int count;

    if (!(file >> originX >> originY >> originZ >> count)
        || count == 0 || count > 100000)
        return false;

    tiles.reserve(count);

    for (unsigned int i = 0; i < count; ++i) {
        Tile tile;
        std::string relative;

        if (!(file >> relative
             >> tile.xmin >> tile.ymin >> tile.xmax >> tile.ymax
             >> tile.zmin >> tile.zmax))
            return false;

        std::replace(relative.begin(), relative.end(), '/', '\\');
        tile.filename = folder + "\\" + relative;

        if (tile.xmax <= tile.xmin || tile.ymax <= tile.ymin)
            return false;

        tiles.push_back(tile);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Background loader
// ---------------------------------------------------------------------------

DWORD WINAPI loaderMain(void*) {
    while (!InterlockedCompareExchange(&loaderStop, 0, 0)) {
        WaitForSingleObject(loaderEvent, 100);

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
            tiles[request.tile].filename, request.lod
        );

        Result result;
        result.tile = request.tile;
        result.lod = request.lod;
        result.mesh = mesh;

        EnterCriticalSection(&loaderLock);
        results.push_back(result);
        LeaveCriticalSection(&loaderLock);
    }

    return 0;
}

double tileDistance(const Tile& tile) {
    double x0 = tile.xmin - originX;
    double x1 = tile.xmax - originX;
    double z0 = originY - tile.ymax;
    double z1 = originY - tile.ymin;

    double dx = std::max(x0 - camera.x, std::max(0.0, camera.x - x1));
    double dz = std::max(z0 - camera.z, std::max(0.0, camera.z - z1));

    return std::sqrt(dx*dx + dz*dz);
}

size_t cacheBytes() {
    size_t bytes = 0;
    for (size_t i = 0; i < tiles.size(); ++i)
        if (tiles[i].mesh) bytes += tiles[i].mesh->bytes();
    return bytes;
}

void updateStreaming() {
    // Transfer finished meshes. There are no OpenGL calls in the worker.
    std::vector<Result> completed;

    EnterCriticalSection(&loaderLock);
    completed.swap(results);
    LeaveCriticalSection(&loaderLock);

    for (size_t i = 0; i < completed.size(); ++i) {
        Result& result = completed[i];
        Tile& tile = tiles[result.tile];
        tile.pending = false;

        if (!result.mesh) {
            tile.failed = true;
            continue;
        }

        if (tileDistance(tile) > unloadRadius) {
            delete result.mesh;
            continue;
        }

        delete tile.mesh;
        tile.mesh = result.mesh;
        tile.loadedLOD = result.lod;
    }

    // Evict tiles well outside the working area.
    for (size_t i = 0; i < tiles.size(); ++i) {
        Tile& tile = tiles[i];
        if (tile.mesh && tileDistance(tile) > unloadRadius) {
            delete tile.mesh;
            tile.mesh = 0;
            tile.loadedLOD = -1;
        }
    }

    // Enforce the cache budget, evicting farthest tiles first.
    size_t bytes = cacheBytes();

    while (bytes > memoryBudget) {
        int farthest = -1;
        double farthestDistance = -1;

        for (size_t i = 0; i < tiles.size(); ++i) {
            if (tiles[i].mesh) {
                double distance = tileDistance(tiles[i]);
                if (distance > farthestDistance) {
                    farthestDistance = distance;
                    farthest = (int)i;
                }
            }
        }

        if (farthest < 0) break;

        bytes -= tiles[farthest].mesh->bytes();
        delete tiles[farthest].mesh;
        tiles[farthest].mesh = 0;
        tiles[farthest].loadedLOD = -1;
    }

    // Choose one nearest outstanding request at a time.
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

        // Small distance changes should not trigger continuous LOD churn.
        bool needs = !tile.mesh || tile.loadedLOD != lod;

        if (distance <= loadRadius && needs
            && !tile.pending && !tile.failed
            && distance < bestDistance) {
            bestDistance = distance;
            best = (int)i;
        }
    }

    // Keep the queue short: at most one active and one queued request.
    EnterCriticalSection(&loaderLock);

    int pendingCount = 0;
    for (size_t i = 0; i < tiles.size(); ++i)
        if (tiles[i].pending) ++pendingCount;

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
        (xmin+xmax)*0.5 - originX,
        highest - originZ,
        originY - (ymin+ymax)*0.5
    );

    // A local overview rather than loading the entire city.
    orbitDistance = loadRadius * 0.8;
    yaw = 0;
    pitch = -0.55;
    updateOrbitCamera();
}

void switchCamera() {
    if (orbitMode) {
        orbitMode = false;
    } else {
        target = camera + forwardVector() * orbitDistance;
        orbitMode = true;
    }
}

void updateCamera(double dt) {
    double speed = movementSpeed * dt;
    if (keys[VK_SHIFT]) speed *= 4;

    Vec forward = forwardVector();
    Vec right(std::cos(yaw), 0, std::sin(yaw));
    Vec movement;

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

// ---------------------------------------------------------------------------
// Frustum and rendering
// ---------------------------------------------------------------------------

double planes[6][4];

void extractFrustum() {
    double projection[16], view[16], clip[16];
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetDoublev(GL_MODELVIEW_MATRIX, view);

    // clip = projection * view, using OpenGL column-major matrices.
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row) {
            clip[column*4+row] = 0;
            for (int k = 0; k < 4; ++k)
                clip[column*4+row] +=
                    projection[k*4+row] * view[column*4+k];
        }

    for (int p = 0; p < 6; ++p) {
        int axis = p / 2;
        double sign = (p % 2 == 0) ? 1 : -1;

        for (int k = 0; k < 4; ++k)
            planes[p][k] = clip[k*4+3] + sign*clip[k*4+axis];

        double length = std::sqrt(
            planes[p][0]*planes[p][0]
            + planes[p][1]*planes[p][1]
            + planes[p][2]*planes[p][2]
        );

        if (length > 0)
            for (int k = 0; k < 4; ++k)
                planes[p][k] /= length;
    }
}

bool visible(const Tile& tile) {
    // Coordinates relative to the camera, matching the modelview setup.
    double x0 = tile.xmin-originX-camera.x;
    double x1 = tile.xmax-originX-camera.x;
    double y0 = tile.zmin-originZ-camera.y;
    double y1 = tile.zmax-originZ-camera.y;
    double z0 = originY-tile.ymax-camera.z;
    double z1 = originY-tile.ymin-camera.z;

    for (int p = 0; p < 6; ++p) {
        double x = planes[p][0] >= 0 ? x1 : x0;
        double y = planes[p][1] >= 0 ? y1 : y0;
        double z = planes[p][2] >= 0 ? z1 : z0;

        if (planes[p][0]*x + planes[p][1]*y
            + planes[p][2]*z + planes[p][3] < 0)
            return false;
    }

    return true;
}

void text(int x, int y, const char* value) {
    glRasterPos2i(x, y);
    glListBase(fontBase);
    glCallLists((GLsizei)std::strlen(value), GL_UNSIGNED_BYTE, value);
}

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

    char status[256];
    std::sprintf(
        status,
        "%s | %.0f FPS | %d tiles | %lu triangles | %.1f MB cache",
        orbitMode ? "Orbit" : "Flight",
        fps, drawnTiles, drawnTriangles,
        cacheBytes() / (1024.0*1024.0)
    );
    text(12, 22, status);

    if (helpVisible) {
        text(12, 46, "Tab: camera mode    F1: help    F2: wireframe");
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

void render() {
    glViewport(0, 0, viewportWidth, viewportHeight);
    glClearColor(0.25f, 0.32f, 0.40f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    double nearPlane = 1.0;
    double halfHeight = nearPlane * std::tan(60.0 * PI / 360.0);
    double halfWidth = halfHeight * viewportWidth / viewportHeight;

    glFrustum(
        -halfWidth, halfWidth, -halfHeight, halfHeight,
        nearPlane, farPlane
    );

    Vec forward = unit(forwardVector());
    Vec right = unit(cross(forward, Vec(0, 1, 0)));
    Vec up = cross(right, forward);

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
    GLfloat diffuse[] = { 0.85f, 0.85f, 0.80f, 1 };
    GLfloat ambient[] = { 0.30f, 0.30f, 0.32f, 1 };

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
        Tile& tile = tiles[i];
        if (!tile.mesh || !visible(tile)) continue;

        Mesh& mesh = *tile.mesh;

        glPushMatrix();
        glTranslated(
            tile.xmin-originX-camera.x,
            -camera.y,
            originY-tile.ymax-camera.z
        );

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

    // Small orbit target cross.
    if (orbitMode) {
        glDisable(GL_LIGHTING);
        glColor3f(1, 0.5f, 0.1f);

        Vec p = target-camera;
        double size = std::max(1.0, orbitDistance*0.004);

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

// ---------------------------------------------------------------------------
// Window controls
// ---------------------------------------------------------------------------

LRESULT CALLBACK windowProc(HWND hwnd, UINT message,
                            WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CLOSE:
        running = false;
        return 0;

    case WM_SIZE:
        viewportWidth = std::max(1, (int)LOWORD(lParam));
        viewportHeight = std::max(1, (int)HIWORD(lParam));
        return 0;

    case WM_KILLFOCUS:
        for (int i = 0; i < 256; ++i) keys[i] = false;
        leftDragging = rightDragging = false;
        ReleaseCapture();
        return 0;

    case WM_KEYDOWN:
        if (wParam < 256) keys[wParam] = true;

        // Ignore auto-repeat for toggles.
        if (!(lParam & (1L << 30))) {
            if (wParam == VK_ESCAPE) running = false;
            if (wParam == VK_F1) helpVisible = !helpVisible;
            if (wParam == VK_F2) wireframe = !wireframe;
            if (wParam == VK_TAB) switchCamera();
            if (wParam == VK_HOME) homeCamera();
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
        SetCapture(hwnd);
        return 0;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
        if (message == WM_LBUTTONUP) leftDragging = false;
        else rightDragging = false;

        if (!leftDragging && !rightDragging) ReleaseCapture();
        return 0;

    case WM_MOUSEMOVE: {
        int x = (short)LOWORD(lParam);
        int y = (short)HIWORD(lParam);
        int dx = x-mouseX;
        int dy = y-mouseY;
        mouseX = x;
        mouseY = y;

        if ((!orbitMode && rightDragging)
            || (orbitMode && leftDragging)) {
            yaw += dx*0.005;
            pitch -= dy*0.005;
            pitch = std::max(-1.50, std::min(1.50, pitch));
        } else if (orbitMode && rightDragging) {
            Vec right(std::cos(yaw), 0, std::sin(yaw));
            Vec up = unit(cross(right, forwardVector()));
            double scale = orbitDistance*0.0015;
            target = target-right*(dx*scale)+up*(dy*scale);
        }

        if (orbitMode) updateOrbitCamera();
        return 0;
    }

    case WM_MOUSEWHEEL: {
        double steps = (short)HIWORD(wParam)/120.0;

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
        return 1;
    }

    return DefWindowProcA(hwnd, message, wParam, lParam);
}

// ---------------------------------------------------------------------------
// Simple startup graphics selector
// ---------------------------------------------------------------------------

int selectedPreset = -1;

LRESULT CALLBACK launcherProc(HWND hwnd, UINT message,
                              WPARAM wParam, LPARAM lParam) {
    if (message == WM_COMMAND) {
        int id = LOWORD(wParam);
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

int choosePreset(HINSTANCE instance) {
    WNDCLASSA wc;
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
    );

    if (!launcher) return -2;

    CreateWindowA(
        "STATIC", "Choose a preset. Start with Low on older hardware.",
        WS_CHILD | WS_VISIBLE,
        20, 20, 420, 25, launcher, 0, instance, 0
    );

    const char* labels[] = { "Low", "Medium", "High" };

    for (int i = 0; i < 3; ++i)
        CreateWindowA(
            "BUTTON", labels[i], WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            20+i*145, 65, 130, 35,
            launcher, (HMENU)(INT_PTR)(101+i), instance, 0
        );

    ShowWindow(launcher, SW_SHOW);

    MSG message;
    while (selectedPreset == -1 && GetMessage(&message, 0, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessage(&message);
    }

    return selectedPreset;
}

bool initializeOpenGL(HINSTANCE instance) {
    WNDCLASSA wc;
    ZeroMemory(&wc, sizeof(wc));
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

    PIXELFORMATDESCRIPTOR format;
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
    if (!selected || !SetPixelFormat(deviceContext, selected, &format))
        return false;

    renderingContext = wglCreateContext(deviceContext);
    if (!renderingContext
        || !wglMakeCurrent(deviceContext, renderingContext))
        return false;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_SMOOTH);

    // Bitmap font: no texture atlas or third-party UI dependency.
    fontBase = glGenLists(256);

    HFONT font = CreateFontA(
        -15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY, FIXED_PITCH, "Courier New"
    );

    HGDIOBJ oldFont = SelectObject(deviceContext, font);
    BOOL fontOK = wglUseFontBitmapsA(deviceContext, 0, 256, fontBase);
    SelectObject(deviceContext, oldFont);
    DeleteObject(font);

    if (!fontOK) return false;

    ShowWindow(windowHandle, SW_SHOW);
    return true;
}

// ---------------------------------------------------------------------------
// Entry point and cleanup
// ---------------------------------------------------------------------------

int WINAPI WinMain(HINSTANCE instance, HINSTANCE,
                   LPSTR commandLine, int) {
    // Accept one folder argument, optionally surrounded by quotes.
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

    if (terrainFolder.empty()) terrainFolder = "terrain";

    if (!loadIndex(terrainFolder)) {
        MessageBoxA(
            0,
            "Cannot read terrain_index.txt.\n\n"
            "Launch with the prepared terrain folder as an argument.",
            "Terrain viewer", MB_OK | MB_ICONERROR
        );
        return 1;
    }

    int preset = choosePreset(instance);
    if (preset < 0) return 0;

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

    LARGE_INTEGER frequency, previous, now;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&previous);

    double fpsTime = 0;
    int fpsFrames = 0;

    while (running) {
        MSG message;
        while (PeekMessage(&message, 0, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) running = false;
            TranslateMessage(&message);
            DispatchMessage(&message);
        }

        if (!running) break;

        QueryPerformanceCounter(&now);
        double dt = (double)(now.QuadPart-previous.QuadPart)
                  / frequency.QuadPart;
        previous = now;

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

        // Yield a little CPU time to Windows and the loader.
        Sleep(1);
    }

    // Join the worker before freeing tile data or synchronization objects.
    InterlockedExchange(&loaderStop, 1);
    SetEvent(loaderEvent);
    WaitForSingleObject(loaderThread, INFINITE);

    CloseHandle(loaderThread);
    CloseHandle(loaderEvent);

    for (size_t i = 0; i < results.size(); ++i)
        delete results[i].mesh;

    for (size_t i = 0; i < tiles.size(); ++i)
        delete tiles[i].mesh;

    DeleteCriticalSection(&loaderLock);

    if (fontBase) glDeleteLists(fontBase, 256);

    wglMakeCurrent(0, 0);
    wglDeleteContext(renderingContext);
    ReleaseDC(windowHandle, deviceContext);
    DestroyWindow(windowHandle);

    return 0;
}
