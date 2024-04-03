// Tests.hpp

#pragma once

#include <vector>

#include <Eigen/Dense>
#include <implot.h>

#include <Oseen.hpp>
#include <SharedDataOseen.hpp>
#include <pybind11/embed.h>

namespace py = pybind11;

void test1();
void plotTest1(std::vector<Eigen::Vector3f> test_points, std::vector<Eigen::Vector3f> velocities);
void test2();
void plotTest2(std::vector<Eigen::Vector3f> velocities);