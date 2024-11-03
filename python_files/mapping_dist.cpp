#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <omp.h> 

namespace py = pybind11;

std::tuple<py::array_t<int>, py::array_t<double>>
calc_dist(py::array_t<double> coords1, py::array_t<double> coords2) {
    // Extract input data from NumPy arrays
    auto coords1_buf = coords1.unchecked<2>(); // (m, 3)
    auto coords2_buf = coords2.unchecked<2>(); // (n, 3)

    const int m = coords1_buf.shape(0);
    const int n = coords2_buf.shape(0);

    // Initialize variables
    Eigen::MatrixXd dist(m, n);
    Eigen::VectorXi nid_identical(m);
    Eigen::VectorXd min_dist(m);

    // Calculate the distance between two sets of coordinates
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            dist(i,j)=std::sqrt((coords1_buf(i, 0)-coords2_buf(j, 0)) * (coords1_buf(i, 0) - coords2_buf(j, 0)) +
                          (coords1_buf(i, 1)-coords2_buf(j, 1)) * (coords1_buf(i, 1) - coords2_buf(j, 1)) +
                          (coords1_buf(i, 2)-coords2_buf(j, 2)) * (coords1_buf(i, 2) - coords2_buf(j, 2)));
        }
    }

    // Calculate the nearest point
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        min_dist(i) = dist.row(i).minCoeff(&nid_identical(i));
    }

    // Convert Eigen::Matrix to NumPy array
    //py::array_t<double> dist_array = py::array_t<double>({m, n}); // (m, n) の空の配列を作成
    py::array_t<int> nid_identical_array = py::array_t<int>(m); // Create an empty array of shape (m,)
    py::array_t<double> min_dist_array = py::array_t<double>(m); // Create an empty array of shape (m,)

    //auto dist_array_buf = dist_array.mutable_unchecked<2>(); // mutable buffer for dist_array
    //auto nid_identical_array_buf = nid_identical_array.mutable_unchecked<1>(); // mutable buffer for nid_identical_array
    //auto min_dist_array_buf = min_dist_array.mutable_unchecked<1>(); // mutable buffer for min_dist_array
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        nid_identical_array.mutable_unchecked<1>()(i) = nid_identical(i);
        min_dist_array.mutable_unchecked<1>()(i) = min_dist(i);
    }
    //for (int i = 0; i < m; ++i) {
    //    for (int j = 0; j < n; ++j) {
    //        dist_array_buf(i, j) = dist(i, j);
    //    }
    //    nid_identical_array_buf(i) = nid_identical(i);
    //    min_dist_array_buf(i) = min_dist(i);
    //}
    
    return std::make_tuple(nid_identical_array, min_dist_array);
}

PYBIND11_MODULE(mapping_dist, m) {
    m.def("calc_dist", &calc_dist,
    "Calculate distances between two sets of coordinates",
    py::arg("coords1"), py::arg("coords2"));
}
