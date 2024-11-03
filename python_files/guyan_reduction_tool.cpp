#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <omp.h>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;

VectorXd get_grad_C(const MatrixXd& g, const MatrixXi& c_indices, const MatrixXd& invC_B) {
    int nnz = c_indices.rows();
    VectorXd grad_C(nnz);      

    #pragma omp parallel for
    for (int i = 0; i < nnz; ++i) {
        int idx1 = c_indices(i, 0);
        int idx2 = c_indices(i, 1);

        //Calculate grad_C for each non-zero element
        grad_C(i) = (invC_B.row(idx1) * g * invC_B.row(idx2).transpose()).sum();
    }
    return grad_C;
}

PYBIND11_MODULE(guyan_reduction_tool, m) {
    m.def("get_grad_C", &get_grad_C, "Calculate grad_C",
          py::arg("g"), py::arg("c_indices"), py::arg("invC_B"));
}