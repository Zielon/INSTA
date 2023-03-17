//#include <rta/nn.h>
//
//rta::NearsetNeighbours::NearsetNeighbours() {
//
//}
//
//std::shared_ptr<rta::FlannTree> rta::NearsetNeighbours::build(const std::vector<Vector3f>& vertices) {
//    auto flat_points = new float[vertices.size() * 3];
//    for (size_t i = 0; i < vertices.size(); i++) {
//        for (size_t dim = 0; dim < 3; dim++) {
//            flat_points[i * 3 + dim] = vertices[i][dim];
//        }
//    }
//
//    flann::Matrix<float> dataset(flat_points, vertices.size(), 3);
//    auto index = std::make_shared<rta::FlannTree>(dataset, flann::KDTreeIndexParams(1));
//    index->buildIndex();
//
//    return index;
//}
