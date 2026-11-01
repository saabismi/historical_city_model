#ifndef ROADS_H_INCLUDED
#define ROADS_H_INCLUDED

#include <string>
#include <vector>
#include <cstddef>

namespace roads {

// ROAD1 vertex layout: position, normal, UV.
struct Vertex {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
};

// One mesh group of a road piece.
struct Group {
    unsigned int category; // 1 road surface, 2 shoulder
    unsigned int material;
    std::vector<Vertex> vertices;
};

// One road piece stored in one terrain tile.
struct Road {
    std::string id;

    double easting;
    double northing;
    double elevation;

    float minimum[3];
    float maximum[3];

    int luokka;
    int versuh;

    std::vector<Group> groups;
};

// Resident road data attached to one terrain tile.
struct Data {
    std::vector<Road> items;
    std::size_t bytes() const;
};

// Initialize shared textures and terrain-file mapping.
// Call after creating the OpenGL context.
bool initialize(const std::string& folder, std::string& error);

// Find a road file by terrain basename.
// Empty result means that terrain tile has no road data.
std::string filenameForTerrain(const std::string& terrainFilename);

// Load one ROAD1 file on the background thread.
Data* load(
    const std::string& filename,
    bool (*cancelled)()
);

// Preset-specific render range.
void setPreset(int preset);

// Reset per-frame statistics.
void beginFrame();

// Render one resident road tile.
void render(
    const Data* data,
    double cameraX, double cameraY, double cameraZ,
    double originX, double originY, double originZ,
    const double planes[6][4],
    bool wireframe
);

// Release shared textures before destroying the OpenGL context.
void shutdown();

unsigned int indexedTileCount();
long failedLoadCount();

extern bool enabled;
extern unsigned long drawnRoads;
extern unsigned long drawnTriangles;

} // namespace roads

#endif

