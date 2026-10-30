#ifndef VEGETATION_H_INCLUDED
#define VEGETATION_H_INCLUDED

#include <string>
#include <vector>
#include <cstddef>

using namespace std;

namespace vegetation {

// One plant instance from a VEG1 file.
struct Plant {
    unsigned int asset; // Numeric species-variant identifier.
    float east;         // East offset from the vegetation tile origin.
    float south;        // South offset from the vegetation tile origin.
    float elevation;    // Absolute source elevation.
    float heightScale;  // Multiplier for asset card height.
    float widthScale;   // Multiplier for asset card width.
    float yaw;          // Crossed-card orientation in radians.
    float tint;         // RGB brightness multiplier.
    float priority;     // Stable density priority.
};

// One spatially grouped vegetation cell.
struct Cell {
    std::vector<Plant> plants; // Instances belonging to this cell.

    // Absolute bounds in easting, elevation, and south-coordinate space.
    // South coordinate here is negative absolute northing.
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
};

// Resident placement data for one tile.
struct Data {
    double xmin;             // Absolute tile easting origin.
    double ymax;             // Absolute tile northing origin.
    std::vector<Cell> cells; // Occupied viewer cells.

    std::size_t bytes() const; // Approximate resident CPU storage.
};

// Initialize shared textures and indexes.
// Call on the main thread AFTER creating the OpenGL context.
bool initialize(const std::string& folder, std::string& error);

// Return the VEG filename associated with a terrain filename.
// An empty result means there is no corresponding vegetation tile.
std::string filenameForTerrain(const std::string& terrainFilename);

// Load placement data on the background thread.
// Cancellation callback is your existing loaderShouldStop().
Data* load(const std::string& filename, bool (*cancelled)());

// Configure range, density, and available LODs for Low/Medium/High.
void setPreset(int preset);

// Begin one frame's vegetation collection.
void beginFrame();

// Cull cells and collect visible billboard vertices.
// Camera and origin use your existing viewer coordinate system.
void collect(
    const Data* data,
    double cameraX, double cameraY, double cameraZ,
    double originX, double originY, double originZ,
    const double planes[6][4]
);

// Draw texture-grouped billboard batches and restore OpenGL state.
void draw();

// Delete shared OpenGL textures. Call before destroying the context.
void shutdown();

// On-screen statistics/toggle.
extern bool enabled;
extern unsigned long drawnPlants;

} // namespace vegetation

#endif
