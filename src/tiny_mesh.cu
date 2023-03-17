#include <rta/tiny_mesh.h>

#include <iostream>
#include <utility>

#include <rta/debug.h>
#include "neural-graphics-primitives/nerf_loader.h"

#include "tinylogger/tinylogger.h"

#ifdef NDEBUG
constexpr int N_PRIMITIVES_LEAF = 2;
#else
constexpr int N_PRIMITIVES_LEAF = 4;
#endif

using namespace Eigen;

//std::default_random_engine generator(42);
//std::uniform_real_distribution<float> distribution{0.f, 1.f};

rta::TinyMesh::TinyMesh(const std::string &path, cudaStream_t inference_stream, std::shared_ptr<Masking> masking, std::shared_ptr<TinyMesh> canonical) : m_inference_stream(inference_stream), m_masking(masking) {
    load(path);

    mesh_total_count++;

    build_triangles(canonical);
    calculate_normals();
    build_tries();

//    build_flann();

    find_bounding_sphere();
}

void rta::TinyMesh::load(const std::string &path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) {
        throw std::runtime_error(warn + err);
    }

    Array3f offset = {0.5f, 0.5f, 0.5f};
    Vector3f center = {0, 0, 0};
    Array3f flip = {1.f, 1.f, 1.f};

    auto num_vertices = attrib.vertices.size();
    for (int i = 0; i < num_vertices; i += 3) {
        Array3f vertex = {attrib.vertices[i + 0], attrib.vertices[i + 1], attrib.vertices[i + 2]};
        vertex *= ngp::MESH_SCALE;
        vertex += offset;
        center += vertex.matrix();
        vertices.emplace_back(vertex.matrix());
    }

    for (size_t s = 0; s < shapes.size(); s++) {
        auto num_faces = shapes[s].mesh.indices.size();
        for (int i = 0; i < num_faces; i += 3) {
            uint32_t a = shapes[s].mesh.indices[i + 0].vertex_index;
            uint32_t b = shapes[s].mesh.indices[i + 1].vertex_index;
            uint32_t c = shapes[s].mesh.indices[i + 2].vertex_index;
            faces.emplace_back(a, b, c);
        }
    }

    m_centorid = center / float(vertices.size());
}

void rta::TinyMesh::print() const {
//    tlog::success() << "Mesh loaded with faces: " << faces.size() << " vertices: " << vertices.size();
}

void rta::TinyMesh::save(const std::string &path) const {
    std::ofstream out;
    out.open(path);

    for (auto vert: vertices) {
        out << "v " << vert.x() << " " << vert.y() << " " << vert.z() << std::endl;
    }

    for (auto n: normals) {
        out << "vn " << n.x() << " " << n.y() << " " << n.z() << std::endl;
    }

    for (auto face: faces) {
        out << "f " << face.x() + 1 << " " << face.y() + 1 << " " << face.z() + 1 << std::endl;
    }

    out.close();
}

void rta::TinyMesh::save_xyz(const std::string &path) const {
    std::ofstream out;
    out.open(path);
    for (auto vert: vertices) {
        out << vert.x() << " " << vert.y() << " " << vert.z() << std::endl;
    }
    out.close();
}

void save_triangles(const std::string &path, const std::vector<ngp::Triangle> &triangles) {
    std::ofstream out;
    out.open(path);
    for (auto &t: triangles) {
        auto vert = t.centroid();
        out << vert.x() << " " << vert.y() << " " << vert.z() << std::endl;
//        for (auto vert : {t.a, t.b, t.b}) {
//            out << vert.x() << " " << vert.y() << " " << vert.z() << std::endl;
//        }
    }
    out.close();
}

void rta::TinyMesh::build_triangles(const std::shared_ptr<TinyMesh>& canonical) {
    for (const auto &face: faces) {
        auto x = vertices[face[0]];
        auto y = vertices[face[1]];
        auto z = vertices[face[2]];
        ngp::Triangle t = {x, y, z};
        t.id = triangles_cpu.size();
        t.mask_id = m_masking->get_triangle_mask_id(t.id);
//        t.flame_mask_id = m_masking->get_flame_triangle_mask_id(t.id); // Problem with topology

        float s = 1;
        if (canonical) {
            auto canonical_tri = canonical->triangles_cpu[t.id];
            float relative = t.surface_area() / canonical_tri.surface_area();
            s *= relative;
//            s = t.scale().cwiseQuotient(canonical_tri.scale());
//            s = t.scale();
        }

        t.scaling = s;
        triangles_cpu.push_back(t);
    }

    m_aabb.min = Vector3f::Constant(std::numeric_limits<float>::infinity());
    m_aabb.max = Vector3f::Constant(-std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < vertices.size(); ++i) {
        m_aabb.enlarge(vertices[i]);
    }
    m_aabb.inflate(m_aabb.diag().norm() * 0.15f);
//    m_aabb.min.z() += 0.1;
}

void rta::TinyMesh::build_tries() {
    triangle_bvh = ngp::TriangleBvh::make();
    auto copy = triangles_cpu;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    triangles_gpu_orig_order.resize_and_copy_from_host(triangles_cpu);
//    save_triangles("triangles.xyz",triangles_cpu);
    triangle_bvh->build(triangles_cpu, N_PRIMITIVES_LEAF); // build modifies the triangles
    triangles_gpu.resize_and_copy_from_host(triangles_cpu);
    triangles_cpu = copy;
//    triangle_bvh->build_optix(triangles_gpu, m_inference_stream); // will not work for GAS because of the id prop in ngp::Triangle
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    tlog::success() << "[RTA] (" << mesh_total_count << ") BVH creation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms] for v: " << vertices.size() << " f: " << faces.size();
}

void rta::TinyMesh::build_flann() {
    sample_points_nn();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//    tree = m_nn.build(samples);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    tlog::success() << "[RTA] (" << mesh_total_count << ") Flann creation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms] for " << samples.size() << " samples";
}

void rta::TinyMesh::sample_points_triangle(const ngp::Triangle &t, uint32_t faceId) {
    for (int i = 0; i < m_max_samples_triangle; ++i) {
//        auto x = distribution(generator);
//        auto y = distribution(generator);
//        samples.push_back(t.sample_uniform_position({x, y}));
//        samples_ids.push_back(faceId);
    }
}

void rta::TinyMesh::sample_points_nn() {
    for (auto &triangle: triangles_cpu) {
        auto faceId = samples.size();
        samples.push_back(triangle.centroid());
        samples_ids.push_back(faceId);
//        sample_points_triangle(t, faceId);
    }
}

void rta::TinyMesh::find_bounding_sphere() {
    // https://github.com/erich666/GraphicsGems/blob/master/gems/BoundSphere.c
    int i;
    float dx, dy, dz;
    float rad_sq, xspan, yspan, zspan, maxspan;
    float old_to_p, old_to_p_sq, old_to_new;
    Vector3f xmin, xmax, ymin, ymax, zmin, zmax, dia1, dia2;
    auto BIGNUMBER = std::numeric_limits<float>::max();

    Vector3f cen;
    float rad;

    /* FIRST PASS: find 6 minima/maxima points */
    xmin.x() = ymin.y() = zmin.z() = BIGNUMBER; /* initialize for min/max compare */
    xmax.x() = ymax.y() = zmax.z() = -BIGNUMBER;
    for (i = 0; i < vertices.size(); i++) {
        auto caller_p = vertices[i];
        /* his ith point. */
        if (caller_p.x() < xmin.x())
            xmin = caller_p; /* New xminimum point */
        if (caller_p.x() > xmax.x())
            xmax = caller_p;
        if (caller_p.y() < ymin.y())
            ymin = caller_p;
        if (caller_p.y() > ymax.y())
            ymax = caller_p;
        if (caller_p.z() < zmin.z())
            zmin = caller_p;
        if (caller_p.z() > zmax.z())
            zmax = caller_p;
    }
    /* Set xspan = distance between the 2 points xmin & xmax (squared) */
    dx = xmax.x() - xmin.x();
    dy = xmax.y() - xmin.y();
    dz = xmax.z() - xmin.z();
    xspan = dx * dx + dy * dy + dz * dz;

    /* Same for y & z spans */
    dx = ymax.x() - ymin.x();
    dy = ymax.y() - ymin.y();
    dz = ymax.z() - ymin.z();
    yspan = dx * dx + dy * dy + dz * dz;

    dx = zmax.x() - zmin.x();
    dy = zmax.y() - zmin.y();
    dz = zmax.z() - zmin.z();
    zspan = dx * dx + dy * dy + dz * dz;

    /* Set points dia1 & dia2 to the maximally separated pair */
    dia1 = xmin;
    dia2 = xmax; /* assume xspan biggest */
    maxspan = xspan;
    if (yspan > maxspan) {
        maxspan = yspan;
        dia1 = ymin;
        dia2 = ymax;
    }
    if (zspan > maxspan) {
        dia1 = zmin;
        dia2 = zmax;
    }


    /* dia1,dia2 is a diameter of initial sphere */
    /* calc initial center */
    cen.x() = (dia1.x() + dia2.x()) / 2.0;
    cen.y() = (dia1.y() + dia2.y()) / 2.0;
    cen.z() = (dia1.z() + dia2.z()) / 2.0;
    /* calculate initial radius**2 and radius */
    dx = dia2.x() - cen.x(); /* x component of radius vector */
    dy = dia2.y() - cen.y(); /* y component of radius vector */
    dz = dia2.z() - cen.z(); /* z component of radius vector */
    rad_sq = dx * dx + dy * dy + dz * dz;
    rad = sqrt(rad_sq);

    /* SECOND PASS: increment current sphere */

    for (i = 0; i < vertices.size(); i++) {
        auto caller_p = vertices[i];
        /* with his ith point. */
        dx = caller_p.x() - cen.x();
        dy = caller_p.y() - cen.y();
        dz = caller_p.z() - cen.z();
        old_to_p_sq = dx * dx + dy * dy + dz * dz;
        if (old_to_p_sq > rad_sq)    /* do r**2 test first */
        {    /* this point is outside of current sphere */
            old_to_p = sqrt(old_to_p_sq);
            /* calc radius of new sphere */
            rad = (rad + old_to_p) / 2.0;
            rad_sq = rad * rad;    /* for next r**2 compare */
            old_to_new = old_to_p - rad;
            /* calc center of new sphere */
            cen.x() = (rad * cen.x() + old_to_new * caller_p.x()) / old_to_p;
            cen.y() = (rad * cen.y() + old_to_new * caller_p.y()) / old_to_p;
            cen.z() = (rad * cen.z() + old_to_new * caller_p.z()) / old_to_p;
//            /* Suppress if desired */
//            printf("\n New sphere: cen,rad = %f %f %f   %f",
//                   cen.x(), cen.y(), cen.z(), rad);
        }
    }

    m_bsphere = ngp::BoundingSphere(rad, cen);
}

rta::TinyMesh::~TinyMesh() {
    triangles_gpu.free_memory();
    triangles_gpu_orig_order.free_memory();
}

std::vector<int> rta::TinyMesh::get_triangle_3_neighbours() {
    std::vector<std::set<int>> neighbours(vertices.size());
    int faceid = 0;
    for (const auto &face: faces) {
        auto x = face[0];
        auto y = face[1];
        auto z = face[2];
        neighbours[x].insert(faceid);
        neighbours[y].insert(faceid);
        neighbours[z].insert(faceid);
        ++faceid;
    }

    int i = 0;
    std::vector<int> neighbours3(faces.size() * 3, -1);
    for (const auto &face: faces) {
        auto a = face[0];
        auto b = face[1];
        auto c = face[2];

        auto nn_a = neighbours[a];
        auto nn_b = neighbours[b];
        auto nn_c = neighbours[c];

        std::vector<int> egde0;
        std::set_intersection(nn_a.begin(), nn_a.end(), nn_b.begin(), nn_b.end(), std::back_inserter(egde0));

        std::vector<int> egde1;
        std::set_intersection(nn_a.begin(), nn_a.end(), nn_c.begin(), nn_c.end(), std::back_inserter(egde1));

        std::vector<int> egde2;
        std::set_intersection(nn_b.begin(), nn_b.end(), nn_c.begin(), nn_c.end(), std::back_inserter(egde2));

        std::vector<int> ids;
        ids.insert(ids.end(), egde0.begin(), egde0.end());
        ids.insert(ids.end(), egde1.begin(), egde1.end());
        ids.insert(ids.end(), egde2.begin(), egde2.end());
        std::set<int> unique_ids(ids.begin(), ids.end());
        ids.assign(unique_ids.begin(), unique_ids.end());

        auto d = 4 - ids.size();
        if (d > 0)
            for (int p = 0; p < d; ++p) ids.push_back(-1);

        int k = 0;
        for (int n: ids) {
            if (n != i) {
                neighbours3[i * 3 + k] = n;
                ++k;
            }
        }
        ++i;
    }

    return neighbours3;
}

std::vector<int> rta::TinyMesh::get_triangle_all_neighbours() {
    int MAX_N = 10;
    std::vector<std::set<int>> neighbours(vertices.size());
    int i = 0;
    for (const auto &face: faces) {
        auto x = face[0];
        auto y = face[1];
        auto z = face[2];
        neighbours[x].insert(i);
        neighbours[y].insert(i);
        neighbours[z].insert(i);
        ++i;
    }

    i = 0;
    std::vector<int> neighbours_all(faces.size() * MAX_N, -1);
    for (const auto &face: faces) {
        auto a = face[0];
        auto b = face[1];
        auto c = face[2];

        auto nn_a = neighbours[a];
        auto nn_b = neighbours[b];
        auto nn_c = neighbours[c];

        auto ids = nn_a;
        ids.insert(nn_b.begin(), nn_b.end());
        ids.insert(nn_c.begin(), nn_c.end());

        int k = 0;
        for (int n: ids) {
            if (k >= MAX_N) break;
            neighbours_all[i * MAX_N + k] = n;
            ++k;
        }
        ++i;
    }

    return neighbours_all;
}

void rta::TinyMesh::calculate_normals() {
    std::vector<std::set<int>> neighbours(vertices.size());
    int i = 0;
    for (const auto &face: faces) {
        auto x = face[0];
        auto y = face[1];
        auto z = face[2];
        neighbours[x].insert(i);
        neighbours[y].insert(i);
        neighbours[z].insert(i);
        ++i;
    }

    auto fun = [&](const std::set<int>& fs) {
        Eigen::Vector3f normal = Eigen::Vector3f::Zero();
        for (auto f : fs){
            ngp::Triangle t = triangles_cpu[f];
            Vector3f n = (t.b - t.a).cross(t.c - t.a);
            normal += n;
        }
        return normal.normalized();
    };

    normals.clear();
    normals.resize(vertices.size(), Eigen::Vector3f::Zero());

    i = 0;
    for (const auto &face: faces) {
        auto a = face[0];
        auto b = face[1];
        auto c = face[2];

//        auto nn_a = neighbours[a];
//        triangles_cpu[i].na = fun(nn_a);
//
//        auto nn_b = neighbours[b];
//        triangles_cpu[i].nb = fun(nn_b);
//
//        auto nn_c = neighbours[c];
//        triangles_cpu[i].nc = fun(nn_c);
//
//        normals[a] = triangles_cpu[i].na;
//        normals[b] = triangles_cpu[i].nb;
//        normals[c] = triangles_cpu[i].nc;

        i++;
    }
}
