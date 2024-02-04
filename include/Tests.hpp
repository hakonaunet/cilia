// Tests.hpp

#pragma once

#include <vector>

#include <Eigen/Dense>
#include <implot.h>

#include <Oseen.hpp>
#include <SharedDataOseen.hpp>

void test1();
void plotTest1(std::vector<Eigen::Vector3d> test_points, std::vector<Eigen::Vector3d> velocities);