#ifndef BUILDINGS_H_INCLUDED
#define BUILDINGS_H_INCLUDED

#include <string>
#include <vector>
#include <cstddef>

namespace buildings {

// Exact vertex layout written by generate_buildings.py:
// position, normal, texture coordinates = eight float32 values.
struct Vertex {
    float x, y, z;    // Building-local east/up/south position.
    float nx, ny, nz; // Surface normal.
    float u, v;       // Repeating texture coordinates.
};

// One wall, roof, or floor group.
struct Group {
    unsigned int category; // 1 wall, 2 roof, 3 floor.
    unsigned int material; // Texture ID, or 0xFFFFFFFF for untextured.
    std::vector<Vertex> vertices; // Three consecutive vertices per triangle.
};

// One complete building, stored only in its owner tile.
struct Building {
    std::string id; // Persistent project building ID.

    double easting;   // Absolute anchor easting.
    double northing;  // Absolute anchor northing.
    double elevation; // Absolute anchor elevation.

    float minimum[3]; // Full local geometry bounds: X,Y,Z.
    float maximum[3];

    std::vector<Group> groups;
};

// Resident building data associated with one terrain tile.
struct Data {
    std::vector<Building> items;

    // Approximate resident CPU memory.
    std::size_t bytes() const;
};

// Initialize textures and terrain-file mapping.
// Call on the main thread after creating the OpenGL context.
bool initialize(const std::string& folder, std::string& error);

// Find a building file by terrain basename.
// Empty string means that terrain tile has no building data.
std::string filenameForTerrain(const std::string& terrainFilename);

// Load a BLD2 file without any OpenGL calls.
// Suitable for the existing background thread.
// Empty filename is normal and returns null.
Data* load(
    const std::string& filename,
    bool (*cancelled)()
);

// Set maximum rendering distance for Low / Medium / High.
void setPreset(int preset);

// Reset rendering statistics at the beginning of a frame.
void beginFrame();

// Cull and draw buildings from one resident tile.
//
// The current modelview matrix must be the same rotation-only camera matrix
// used by the terrain renderer. This function handles translations itself.
void render(
    const Data* data,
    double cameraX, double cameraY, double cameraZ,
    double originX, double originY, double originZ,
    const double planes[6][4],
    bool wireframe
);

// Release textures before destroying the OpenGL context.
// Resident Data objects remain owned by main.cpp.
void shutdown();

// Information for diagnostics.
unsigned int indexedTileCount();
long failedLoadCount();

extern bool enabled;
extern unsigned long drawnBuildings;
extern unsigned long drawnTriangles;

} // namespace buildings

#endif

