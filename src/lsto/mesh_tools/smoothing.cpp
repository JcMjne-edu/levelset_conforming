#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <omp.h>

using namespace Eigen;
namespace py = pybind11;

MatrixXd get_norm(const MatrixXi &connect, const MatrixXd &coord) {
    MatrixXd vs1 = coord.row(connect.col(1)) - coord.row(connect.col(0));
    MatrixXd vs2 = coord.row(connect.col(2)) - coord.row(connect.col(0));
    MatrixXd n = vs1.cross(vs2);
    return n.rowwise().normalized();
}

MatrixXd get_aspect_ratio(const MatrixXi &connect, const MatrixXd &coord) {
    // Calculate the aspect ratio of each element
    // connect: (n_elem, 3) matrix containing the indices of the nodes of each element
    // coord: (n_nodes, 3) matrix containing the coordinates of the nodes
    
    // Initialize the vector to store the aspect ratios wit size n_elem
    VectorXd aspect_ratios(connect.rows());
    #pragma omp parallel for
    for (int i = 0; i < connect.rows(); ++i) {
        MatrixXd vs1 = coord.row(connect(i, 1)) - coord.row(connect(i, 0));
        MatrixXd vs2 = coord.row(connect(i, 2)) - coord.row(connect(i, 0));
        MatrixXd vs3 = coord.row(connect(i, 2)) - coord.row(connect(i, 1));
        double l1 = vs1.norm();
        double l2 = vs2.norm();
        double l3 = vs3.norm();
        double s = (l1 + l2 + l3) / 2;
        double area = sqrt(s * (s - l1) * (s - l2) * (s - l3));
        aspect_ratios(i) = area/std::max({l1, l2, l3});
    }
    return aspect_ratios;
}