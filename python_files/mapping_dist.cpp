#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <omp.h> 

namespace py = pybind11;

std::tuple<Eigen::VectorXi, Eigen::VectorXd>
calc_dist(Eigen::MatrixXd coords1, Eigen::MatrixXd coords2) {
  const int m = coords1.rows();
  const int n = coords2.rows();

  Eigen::VectorXi nid_identical(m);
  Eigen::VectorXd min_dist(m);
  min_dist.setConstant(std::numeric_limits<double>::max()); // Initialize with a large value
  
  // Calculate the distance between two sets of coordinates
  #pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double dist_ij = 
        (coords1(i, 0)-coords2(j, 0))*(coords1(i, 0)-coords2(j, 0))+
        (coords1(i, 1)-coords2(j, 1))*(coords1(i, 1)-coords2(j, 1))+
        (coords1(i, 2)-coords2(j, 2))*(coords1(i, 2)-coords2(j, 2));
      if (dist_ij < min_dist(i)) {
        min_dist(i) = dist_ij;
        nid_identical(i) = j;
      }
    }
  }
  min_dist=min_dist.array().sqrt();  // Take square root only once per point
  return std::make_tuple(nid_identical, min_dist);
}

PYBIND11_MODULE(mapping_dist, m) {
  m.def("calc_dist", &calc_dist,
  "Calculate distances between two sets of coordinates",
  py::arg("coords1"), py::arg("coords2"));
}
