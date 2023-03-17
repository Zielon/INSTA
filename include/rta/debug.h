#ifndef REAL_TIME_AVATAR_DEBUG_H
#define REAL_TIME_AVATAR_DEBUG_H

#include <string>
#include <iostream>
#include "Eigen/Core"

#define SHAPE(X) \
( \
(std::cout << "Tensor shape: " << X.sizes() << " " << X.options() << std::endl), \
(void)0 \
)

#define PRINT(X) \
( \
(std::cout << X << std::endl), \
(void)0 \
)

inline void PRINT_E(const Eigen::MatrixXf &vector) {
    Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
    std::cout << vector.format(OctaveFmt) << std::endl;
}

#endif //REAL_TIME_AVATAR_DEBUG_H
