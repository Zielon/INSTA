/*
 -*- coding: utf-8 -*-
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
You can only use this computer program if you have closed
a license agreement with MPG or you get the right to use the computer
program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and
liable to prosecution.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Contact: insta@tue.mpg.de
*/

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
