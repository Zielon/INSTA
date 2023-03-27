#include "Eigen/Core"

using namespace Eigen;

constexpr float MAX_FLOAT = std::numeric_limits<float>::max();
constexpr int BRANCHING_FACTOR = 4;

struct DistAndIdx {
    float dist;
    uint32_t idx;

    // Sort in descending order!
    __host__ __device__ bool operator<(const DistAndIdx &other) {
        return dist < other.dist;
    }
};

template<typename T>
__host__ __device__ void inline compare_and_swap(T &t1, T &t2) {
    if (t1 < t2) {
        T tmp{t1};
        t1 = t2;
        t2 = tmp;
    }
}

// Sorting networks from http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N4L5D3
template<uint32_t N, typename T>
__host__ __device__ void sorting_network(T values[N]) {
    static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
    if (N <= 1) {
        return;
    } else if (N == 2) {
        compare_and_swap(values[0], values[1]);
    } else if (N == 3) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 4) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[1], values[2]);
    }
}

__host__ __device__ static std::pair<int, float> closest_triangle(const Vector3f &point, const ngp::TriangleBvhNode *__restrict__ bvhnodes, const ngp::Triangle *__restrict__ triangles, float max_distance_sq = MAX_FLOAT) {
    ngp::FixedIntStack query_stack;
    query_stack.push(0);

    float shortest_distance_sq = max_distance_sq;
    int shortest_idx = -1;

    while (!query_stack.empty()) {
        int idx = query_stack.pop();

        const ngp::TriangleBvhNode &node = bvhnodes[idx];

        if (node.left_idx < 0) {
            int end = -node.right_idx - 1;
            for (int i = -node.left_idx - 1; i < end; ++i) {
                float dist_sq = triangles[i].distance_sq(point);
                if (dist_sq <= shortest_distance_sq) {
                    shortest_distance_sq = dist_sq;
                    shortest_idx = i;
                }
            }
        } else {
            DistAndIdx children[BRANCHING_FACTOR];

            uint32_t first_child = node.left_idx;

            NGP_PRAGMA_UNROLL
            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                children[i] = {bvhnodes[i + first_child].bb.distance_sq(point), i + first_child};
            }

            sorting_network<BRANCHING_FACTOR>(children);

            NGP_PRAGMA_UNROLL
            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                if (children[i].dist <= shortest_distance_sq) {
                    query_stack.push(children[i].idx);
                }
            }
        }
    }

    if (shortest_idx == -1) {
        shortest_idx = -1;
        shortest_distance_sq = 0.0f;
    }

    return {shortest_idx, std::sqrt(shortest_distance_sq)};
}

__host__ __device__ static std::pair<int, float> ray_intersect(const Vector3f &ro, const Vector3f &rd, const ngp::TriangleBvhNode *__restrict__ bvhnodes, const ngp::Triangle *__restrict__ triangles, float max_distance_sq = MAX_FLOAT) {
    ngp::FixedIntStack query_stack;
    query_stack.push(0);

    float mint = max_distance_sq;
    int shortest_idx = -1;

    while (!query_stack.empty()) {
        int idx = query_stack.pop();

        const ngp::TriangleBvhNode &node = bvhnodes[idx];

        if (node.left_idx < 0) {
            int end = -node.right_idx - 1;
            for (int i = -node.left_idx - 1; i < end; ++i) {
                float t = triangles[i].ray_intersect(ro, rd);
                if (t < mint) {
                    mint = t;
                    shortest_idx = i;
                }
            }
        } else {
            DistAndIdx children[BRANCHING_FACTOR];

            uint32_t first_child = node.left_idx;

            NGP_PRAGMA_UNROLL
            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                children[i] = {bvhnodes[i + first_child].bb.ray_intersect(ro, rd).x(), i + first_child};
            }

            sorting_network<BRANCHING_FACTOR>(children);

            NGP_PRAGMA_UNROLL
            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                if (children[i].dist < mint) {
                    query_stack.push(children[i].idx);
                }
            }
        }
    }

    return {shortest_idx, mint};
}

