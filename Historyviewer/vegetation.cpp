// XP-compatible Windows declarations.
#define WINVER 0x0501
#define _WIN32_WINNT 0x0501
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <GL/gl.h>

#include "vegetation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

using namespace std;

namespace vegetation {

// Shared appearance data for one species variant.
struct Asset {
    float width;       // Unscaled card width in metres.
    float height;      // Unscaled card height in metres.
    float pivotU;      // Horizontal trunk-base image coordinate.
    float pivotV;      // Vertical trunk-base image coordinate.
    GLuint texture[5]; // Near front/side, medium front/side, far.
};

// Interleaved vertex format for texture-grouped CPU arrays.
struct BillboardVertex {
    float x, y, z; // Camera-relative position.
    float u, v;    // Texture coordinate.
    float r, g, b; // Instance tint.
};

// Shared assets are initialized before the background loader starts.
// They remain read-only until shutdown.
static std::vector<Asset> assets;

// Terrain basename -> full vegetation-file path.
static std::map<std::string, std::string> tileFiles;

// One temporary vertex batch for each asset texture.
static std::vector< std::vector<BillboardVertex> > batches;

static double maximumDistance = 1000; // Vegetation render range.
static float density = 0.65f;        // Stable instance retention threshold.
static int minimumLOD = 0;          // Low preset can omit near detail.
static unsigned long plantBudget = 15000; // Per-frame draw cap.

bool enabled = true;
unsigned long drawnPlants = 0;


// Read a fixed number of bytes; false indicates a truncated file.
static bool readBytes(std::ifstream& file, void* target, std::size_t bytes) {
    file.read((char*)target, (std::streamsize)bytes);
    return !!file;
}


// Read one metadata line, accepting CRLF and LF.
static bool readLine(std::ifstream& file, std::string& line) {
    if (!std::getline(file, line)) return false;
    if (!line.empty() && line[line.size()-1] == '\r')
        line.erase(line.size()-1);
    return true;
}


// Convert portable relative paths into Windows paths.
static std::string joinPath(
    const std::string& folder,
    std::string relative
) {
    std::replace(relative.begin(), relative.end(), '/', '\\');
    return folder + "\\" + relative;
}


// Get a filename independently of its parent directory.
static std::string basename(const std::string& filename) {
    std::string::size_type position = filename.find_last_of("\\/");
    return position == std::string::npos
        ? filename : filename.substr(position + 1);
}


// Load the converter's uncompressed 32-bit bottom-origin TGA.
static GLuint loadTexture(const std::string& filename) {
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file) return 0;

    unsigned char header[18];
    if (!readBytes(file, header, sizeof(header))) return 0;

    unsigned int width = header[12] | (header[13] << 8);
    unsigned int height = header[14] | (header[15] << 8);

    // Only accept the exact simple format produced by the converter.
    if (header[0] != 0 || header[1] != 0 || header[2] != 2
        || header[16] != 32 || header[17] != 8
        || width == 0 || height == 0
        || width > 2048 || height > 2048)
        return 0;

    GLint supportedSize = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &supportedSize);
    if (width > (unsigned int)supportedSize
        || height > (unsigned int)supportedSize)
        return 0;

    std::vector<unsigned char> pixels(
        (std::size_t)width * height * 4
    );

    if (!readBytes(file, &pixels[0], pixels.size())) return 0;

    // Convert BGRA from TGA into RGBA for OpenGL 1.1.
    for (std::size_t i = 0; i < pixels.size(); i += 4)
        std::swap(pixels[i], pixels[i+2]);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    if (!texture) return 0;

    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Separate distance LOD textures are already supplied.
    // This prototype does not create an additional mipmap chain.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // Clear prior errors so this upload's result can be checked.
    while (glGetError() != GL_NO_ERROR) {}

    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA,
        width, height, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, &pixels[0]
    );

    if (glGetError() != GL_NO_ERROR) {
        glDeleteTextures(1, &texture);
        return 0;
    }

    return texture;
}


// Initialize assets and the filename-based terrain/vegetation mapping.
bool initialize(const std::string& folder, std::string& error) {
    shutdown();

    std::ifstream file((folder + "\\assets.txt").c_str());
    std::string line;

    if (!readLine(file, line) || line != "VEGETATION_ASSETS_V1") {
        error = "Cannot read vegetation assets.txt.";
        return false;
    }

    unsigned int count = 0;
    if (!readLine(file, line)) return false;
    {
        std::istringstream input(line);
        if (!(input >> count) || count == 0 || count > 1000) {
            error = "Invalid vegetation asset count.";
            return false;
        }
    }

    assets.resize(count);

    // Initialize texture IDs before any possible failure cleanup.
    for (unsigned int i = 0; i < count; ++i)
        std::memset(assets[i].texture, 0, sizeof(assets[i].texture));

    for (unsigned int i = 0; i < count; ++i) {
        unsigned int identifier = 0;
        Asset& asset = assets[i];

        if (!readLine(file, line)) {
            error = "Truncated assets.txt.";
            shutdown();
            return false;
        }

        std::istringstream input(line);
        if (!(input >> identifier >> asset.width >> asset.height
                    >> asset.pivotU >> asset.pivotV)
            || identifier != i
            || asset.width <= 0 || asset.height <= 0
            || asset.pivotU < 0 || asset.pivotU > 1
            || asset.pivotV < 0 || asset.pivotV > 1) {
            error = "Invalid vegetation asset dimensions or ID.";
            shutdown();
            return false;
        }

        for (int texture = 0; texture < 5; ++texture) {
            if (!readLine(file, line)) {
                error = "Missing vegetation texture path.";
                shutdown();
                return false;
            }

            std::string path = joinPath(folder, line);
            asset.texture[texture] = loadTexture(path);

            if (!asset.texture[texture]) {
                error = "Cannot load vegetation texture:\n" + path;
                shutdown();
                return false;
            }
        }
    }

    std::ifstream index(
        (folder + "\\vegetation_index.txt").c_str()
    );

    if (!readLine(index, line) || line != "VEGETATION_INDEX_V1") {
        error = "Cannot read vegetation_index.txt.";
        shutdown();
        return false;
    }

    if (!readLine(index, line)) {
        error = "Missing vegetation tile count.";
        shutdown();
        return false;
    }

    {
        std::istringstream input(line);
        if (!(input >> count) || count > 100000) {
            error = "Invalid vegetation tile count.";
            shutdown();
            return false;
        }
    }

    for (unsigned int i = 0; i < count; ++i) {
        std::string terrainName;
        std::string vegetationName;

        if (!readLine(index, terrainName)
            || !readLine(index, vegetationName)) {
            error = "Truncated vegetation_index.txt.";
            shutdown();
            return false;
        }

        tileFiles[basename(terrainName)] = joinPath(
            folder, vegetationName
        );
    }

    batches.resize(assets.size() * 5);
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}


std::string filenameForTerrain(const std::string& terrainFilename) {
    std::map<std::string, std::string>::const_iterator found =
        tileFiles.find(basename(terrainFilename));

    return found == tileFiles.end() ? "" : found->second;
}


// Approximate resident placement memory.
std::size_t Data::bytes() const {
    std::size_t total = sizeof(Data) + cells.capacity() * sizeof(Cell);
    for (std::size_t i = 0; i < cells.size(); ++i)
        total += cells[i].plants.capacity() * sizeof(Plant);
    return total;
}


// Read one VEG1 file and compute conservative cell bounds.
// Reads fields individually rather than relying on C++ struct packing.
Data* load(const std::string& filename, bool (*cancelled)()) {
    if (filename.empty() || cancelled()) return 0;

    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file) return 0;

    char magic[4];
    double xmin, ymax;
    float cellSize;
    unsigned int cellCount, plantCount;

    if (!readBytes(file, magic, 4)
        || std::memcmp(magic, "VEG1", 4) != 0
        || !readBytes(file, &xmin, 8)
        || !readBytes(file, &ymax, 8)
        || !readBytes(file, &cellSize, 4)
        || !readBytes(file, &cellCount, 4)
        || !readBytes(file, &plantCount, 4)
        || cellSize <= 0
        || cellCount > 100000
        || plantCount > 2000000)
        return 0;

    Data* data = new Data;
    data->xmin = xmin;
    data->ymax = ymax;
    data->cells.reserve(cellCount);

    unsigned int totalRead = 0;

    for (unsigned int i = 0; i < cellCount; ++i) {
        if (cancelled()) {
            delete data;
            return 0;
        }

        unsigned int column, row, count;
        if (!readBytes(file, &column, 4)
            || !readBytes(file, &row, 4)
            || !readBytes(file, &count, 4)
            || count > plantCount - totalRead) {
            delete data;
            return 0;
        }

        totalRead += count;

        Cell cell;
        const double infinity = std::numeric_limits<double>::max();
        cell.xmin = cell.ymin = cell.zmin = infinity;
        cell.xmax = cell.ymax = cell.zmax = -infinity;
        cell.plants.reserve(count);

        for (unsigned int j = 0; j < count; ++j) {
            if ((j % 256) == 0 && cancelled()) {
                delete data;
                return 0;
            }

            Plant plant;
            float values[8];

            if (!readBytes(file, &plant.asset, 4)
                || !readBytes(file, values, sizeof(values))) {
                delete data;
                return 0;
            }

            bool valid = plant.asset < assets.size();
            for (int k = 0; k < 8; ++k)
                if (!(values[k] == values[k])
                    || values[k] < -1e20f || values[k] > 1e20f)
                    valid = false;

            if (!valid || values[3] <= 0 || values[4] <= 0) {
                delete data;
                return 0;
            }

            plant.east = values[0];
            plant.south = values[1];
            plant.elevation = values[2];
            plant.heightScale = values[3];
            plant.widthScale = values[4];
            plant.yaw = values[5];
            plant.tint = values[6];
            plant.priority = values[7];

            const Asset& asset = assets[plant.asset];
            double width = asset.width * plant.widthScale;
            double height = asset.height * plant.heightScale;

            // A fully facing card may tilt in any direction.
            // Bound it by a sphere around its trunk-base pivot.
            double horizontalExtent = width * std::max(
                asset.pivotU, 1.0f - asset.pivotU
            );

            double verticalExtent = height * std::max(
                asset.pivotV, 1.0f - asset.pivotV
            );

            double radius = std::sqrt(
                horizontalExtent * horizontalExtent
                + verticalExtent * verticalExtent
            );

            double x = xmin + plant.east;
            double y = plant.elevation;
            double z = -ymax + plant.south;

            cell.xmin = std::min(cell.xmin, x - radius);
            cell.xmax = std::max(cell.xmax, x + radius);
            cell.ymin = std::min(cell.ymin, y - radius);
            cell.ymax = std::max(cell.ymax, y + radius);
            cell.zmin = std::min(cell.zmin, z - radius);
            cell.zmax = std::max(cell.zmax, z + radius);


            cell.plants.push_back(plant);
        }

        if (!cell.plants.empty())
            data->cells.push_back(cell);
    }

    if (totalRead != plantCount || cancelled()) {
        delete data;
        return 0;
    }

    return data;
}


// Choose practical starting settings; these are not hardware benchmarks.
void setPreset(int preset) {
    if (preset == 0) {
        maximumDistance = 350;
        density = 0.5f;
        minimumLOD = 1;
        plantBudget = 6000;
    } else if (preset == 1) {
        maximumDistance = 1000;
        density = 0.9f;
        minimumLOD = 0;
        plantBudget = 15000;
    } else {
        maximumDistance = 10000;
        density = 1.0f;
        minimumLOD = 0;
        plantBudget = 30000;
    }
}


// Clear frame geometry but retain vector capacity for reuse.
void beginFrame() {
    drawnPlants = 0;
    for (std::size_t i = 0; i < batches.size(); ++i)
        batches[i].clear();
}


// Conservative camera-relative AABB/frustum test.
static bool visibleBox(
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    const double planes[6][4]
) {
    for (int p = 0; p < 6; ++p) {
        double x = planes[p][0] >= 0 ? xmax : xmin;
        double y = planes[p][1] >= 0 ? ymax : ymin;
        double z = planes[p][2] >= 0 ? zmax : zmin;

        if (planes[p][0]*x + planes[p][1]*y
            + planes[p][2]*z + planes[p][3] < 0)
            return false;
    }
    return true;
}


// Append one vertical card to its texture batch.
// directionX/directionZ define the card's horizontal right direction.
static void appendCard(
    std::vector<BillboardVertex>& batch,
    const Asset& asset,
    const Plant& plant,
    double x, double y, double z,
    double directionX, double directionZ
) {
    double width = asset.width * plant.widthScale;
    double height = asset.height * plant.heightScale;

    double left = -asset.pivotU * width;
    double right = (1-asset.pivotU) * width;
    double bottom = -asset.pivotV * height;
    double top = (1-asset.pivotV) * height;

    double offsets[4] = { left, right, right, left };
    double elevations[4] = { bottom, bottom, top, top };
    float u[4] = { 0, 1, 1, 0 };
    float v[4] = { 0, 0, 1, 1 };

    for (int i = 0; i < 4; ++i) {
        BillboardVertex vertex;
        vertex.x = (float)(x + directionX * offsets[i]);
        vertex.y = (float)(y + elevations[i]);
        vertex.z = (float)(z + directionZ * offsets[i]);
        vertex.u = u[i];
        vertex.v = v[i];
        vertex.r = vertex.g = vertex.b = plant.tint;
        batch.push_back(vertex);
    }
}

// Append a card facing the camera in both horizontal and vertical directions.
//
// Its pivot remains at the plant's base position. Unlike an axial billboard,
// its top can tilt away from world-up to remain visible from high altitude.
static void appendFacingCard(
    std::vector<BillboardVertex>& batch,
    const Asset& asset,
    const Plant& plant,
    double x, double y, double z
) {
    double width = asset.width * plant.widthScale;
    double height = asset.height * plant.heightScale;

    // Direction from the plant base toward the camera.
    // Positions passed to this function are already camera-relative.
    double nx = -x;
    double ny = -y;
    double nz = -z;

    double length = std::sqrt(nx*nx + ny*ny + nz*nz);

    if (length < 1e-6) {
        nx = 0;
        ny = 0;
        nz = 1;
    } else {
        nx /= length;
        ny /= length;
        nz /= length;
    }

    // Horizontal right vector perpendicular to the facing direction.
    double rx = nz;
    double ry = 0;
    double rz = -nx;

    double rightLength = std::sqrt(rx*rx + rz*rz);

    // Directly overhead, the horizontal facing direction is undefined.
    if (rightLength < 1e-6) {
        rx = 1;
        ry = 0;
        rz = 0;
    } else {
        rx /= rightLength;
        rz /= rightLength;
    }

    // Card-up = facing direction cross card-right.
    // This tilts the card toward an elevated camera.
    double ux = ny*rz - nz*ry;
    double uy = nz*rx - nx*rz;
    double uz = nx*ry - ny*rx;

    double left = -asset.pivotU * width;
    double right = (1.0 - asset.pivotU) * width;
    double bottom = -asset.pivotV * height;
    double top = (1.0 - asset.pivotV) * height;

    double horizontal[4] = { left, right, right, left };
    double vertical[4] = { bottom, bottom, top, top };

    float textureU[4] = { 0, 1, 1, 0 };
    float textureV[4] = { 0, 0, 1, 1 };

    for (int i = 0; i < 4; ++i) {
        BillboardVertex vertex;

        vertex.x = (float)(
            x + rx*horizontal[i] + ux*vertical[i]
        );
        vertex.y = (float)(
            y + ry*horizontal[i] + uy*vertical[i]
        );
        vertex.z = (float)(
            z + rz*horizontal[i] + uz*vertical[i]
        );

        vertex.u = textureU[i];
        vertex.v = textureV[i];
        vertex.r = vertex.g = vertex.b = plant.tint;

        batch.push_back(vertex);
    }
}


// Cull spatial cells and choose each plant's LOD by actual 3D distance.
void collect(
    const Data* data,
    double cameraX, double cameraY, double cameraZ,
    double originX, double originY, double originZ,
    const double planes[6][4]
) {
    if (!enabled || !data || drawnPlants >= plantBudget) return;

    for (std::size_t i = 0; i < data->cells.size(); ++i) {
        const Cell& cell = data->cells[i];

        double x0 = cell.xmin-originX-cameraX;
        double x1 = cell.xmax-originX-cameraX;
        double y0 = cell.ymin-originZ-cameraY;
        double y1 = cell.ymax-originZ-cameraY;
        double z0 = cell.zmin+originY-cameraZ;
        double z1 = cell.zmax+originY-cameraZ;

        if (!visibleBox(x0,y0,z0,x1,y1,z1,planes)) continue;

        // Distance to nearest point of the cell's expanded bounding box.
        double dx = std::max(x0, std::max(0.0, -x1));
        double dy = std::max(y0, std::max(0.0, -y1));
        double dz = std::max(z0, std::max(0.0, -z1));

        if (dx*dx+dy*dy+dz*dz > maximumDistance*maximumDistance)
            continue;

        for (std::size_t j = 0; j < cell.plants.size(); ++j) {
            if (drawnPlants >= plantBudget) return;

            const Plant& plant = cell.plants[j];
            if (plant.priority > density) continue;

            const Asset& asset = assets[plant.asset];

            double x = data->xmin+plant.east-originX-cameraX;
            double y = plant.elevation-originZ-cameraY;
            double z = originY-data->ymax+plant.south-cameraZ;

            // Measure distance to the approximate crown/card center.
            double centerY = y + (
                0.5-asset.pivotV
            ) * asset.height * plant.heightScale;

            double distance = std::sqrt(
                x*x + centerY*centerY + z*z
            );
            if (distance > maximumDistance) continue;

            int lod = distance < 60 ? 0 : (distance < 150 ? 1 : 2);
            lod = std::max(lod, minimumLOD);

            // Switch to a facing billboard when looking steeply downward
            // at a plant, even if it is within near/medium LOD distance.
            double horizontalDistance = std::sqrt(x*x + z*z);
            double cameraHeightAbovePlant = -y;

            if (cameraHeightAbovePlant > 0
                && cameraHeightAbovePlant > horizontalDistance) {
                // More than approximately 45 degrees above the plant.
                lod = 2;
            }


            std::size_t firstTexture = plant.asset * 5;

            if (lod < 2) {
                std::size_t slot = firstTexture + (lod == 0 ? 0 : 2);
                double cosine = std::cos(plant.yaw);
                double sine = std::sin(plant.yaw);

                appendCard(
                    batches[slot], asset, plant,
                    x,y,z, cosine,sine
                );
                appendCard(
                    batches[slot+1], asset, plant,
                    x,y,z, -sine,cosine
                );
            } else {
                // Fully camera-facing far LOD, visible from above.
                appendFacingCard(
                    batches[firstTexture + 4],
                    asset,
                    plant,
                    x, y, z
                );
            }


            ++drawnPlants;
        }
    }
}


// Draw batches with alpha testing, then restore terrain/overlay state.
void draw() {
    if (!enabled) return;

    glPushAttrib(
        GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_COLOR_BUFFER_BIT
        | GL_DEPTH_BUFFER_BIT | GL_POLYGON_BIT | GL_CURRENT_BIT
    );
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);

    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.5f);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    glDisableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    for (std::size_t i = 0; i < batches.size(); ++i) {
        std::vector<BillboardVertex>& batch = batches[i];
        if (batch.empty()) continue;

        GLuint texture = assets[i/5].texture[i%5];
        glBindTexture(GL_TEXTURE_2D, texture);

        glVertexPointer(
            3, GL_FLOAT, sizeof(BillboardVertex), &batch[0].x
        );
        glTexCoordPointer(
            2, GL_FLOAT, sizeof(BillboardVertex), &batch[0].u
        );
        glColorPointer(
            3, GL_FLOAT, sizeof(BillboardVertex), &batch[0].r
        );

        glDrawArrays(GL_QUADS, 0, (GLsizei)batch.size());
    }

    glPopClientAttrib();
    glPopAttrib();
}


// Release shared textures while the OpenGL context is still current.
// Resident Data objects are owned and deleted by main.cpp.
void shutdown() {
    for (std::size_t i = 0; i < assets.size(); ++i)
        for (int j = 0; j < 5; ++j)
            if (assets[i].texture[j])
                glDeleteTextures(1, &assets[i].texture[j]);

    assets.clear();
    tileFiles.clear();
    batches.clear();
}

} // namespace vegetation

