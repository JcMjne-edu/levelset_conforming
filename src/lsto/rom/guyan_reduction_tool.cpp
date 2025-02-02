#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <omp.h>
#include <cholmod.h>

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

MatrixXd solve_cholmod(const Eigen::SparseMatrix<double>& a, const MatrixXd& b) {
    // initialize CHOLMOD
    cholmod_common c;
    cholmod_start(&c);

    c.supernodal = CHOLMOD_SUPERNODAL;

    // Convert Eigen::SparseMatrix to CHOLMOD form
    cholmod_sparse chol_a;
    chol_a.nrow = a.rows();
    chol_a.ncol = a.cols();
    chol_a.nzmax = a.nonZeros();
    chol_a.p = const_cast<int*>(a.outerIndexPtr());
    chol_a.i = const_cast<int*>(a.innerIndexPtr());
    chol_a.x = const_cast<double*>(a.valuePtr());
    chol_a.stype = 1;  // symmetric
    chol_a.itype = CHOLMOD_INT;
    chol_a.xtype = CHOLMOD_REAL;
    chol_a.dtype = CHOLMOD_DOUBLE;
    chol_a.sorted = 1;
    chol_a.packed = 1;

    // Cholesky factorization
    cholmod_factor* L = cholmod_analyze(&chol_a, &c);
    if (!L || !cholmod_factorize(&chol_a, L, &c)) {
        throw std::runtime_error("CHOLMOD factorization failed!!");
    }

    // Allocate the solution matrix
    Eigen::MatrixXd x(b.rows(), b.cols());

    // Solve for each column of b
    #pragma omp parallel for
    for (int i = 0; i < b.cols(); ++i) {
        // Convert Eigen::MatrixXd to CHOLMOD form
        cholmod_dense chol_b;
        chol_b.nrow = b.rows();
        chol_b.ncol = 1;
        chol_b.nzmax = b.rows();
        chol_b.d = b.rows();
        chol_b.x = const_cast<double*>(b.col(i).data());
        chol_b.xtype = CHOLMOD_REAL;
        chol_b.dtype = CHOLMOD_DOUBLE;

        // Solve
        cholmod_dense* chol_x = cholmod_solve(CHOLMOD_A, L, &chol_b, &c);
        if (!chol_x) {
            #pragma omp critical
            {
                cholmod_free_factor(&L, &c);
                cholmod_finish(&c);
            }
            throw std::runtime_error("CHOLMOD solve failed.");
        }

        // Copy the solution to Eigen::MatrixXd
        std::copy(static_cast<double*>(chol_x->x), 
                  static_cast<double*>(chol_x->x) + b.rows(), 
                  x.col(i).data());

        // Free the solution
        cholmod_free_dense(&chol_x, &c);
    }

    // Free the factor
    cholmod_free_factor(&L, &c);
    cholmod_finish(&c);

    return x;
}


PYBIND11_MODULE(guyan_reduction_tool, m) {
    m.def("get_grad_C", &get_grad_C, "Calculate grad_C",
          py::arg("g"), py::arg("c_indices"), py::arg("invC_B"));
    m.def("solve_cholmod", &solve_cholmod, "Solve a linear system using CHOLMOD",
          py::arg("a"), py::arg("b"));
}