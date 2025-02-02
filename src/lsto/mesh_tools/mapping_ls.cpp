#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

const double THREASHOLD = 1e-8;

// 点が三角形内にあるかを判定し、距離を計算する関数
std::pair<bool, double> is_inside(const Eigen::Vector3d& p, const Eigen::Vector3d& t1, const Eigen::Vector3d& t2, const Eigen::Vector3d& t3, const Eigen::Vector3d& norm) {
    double s1 = (t1 - p).cross(t2 - p).dot(norm);
    double s2 = (t2 - p).cross(t3 - p).dot(norm);
    double s3 = (t3 - p).cross(t1 - p).dot(norm);
    double dist = std::abs((t1 - p).dot(norm));
    bool flag = ((s1 > -THREASHOLD && s2 > -THREASHOLD && s3 > -THREASHOLD) || (s1 < THREASHOLD && s2 < THREASHOLD && s3 < THREASHOLD));
    return {flag, dist};
}

// 各点に最も近い三角形のインデックスを取得
Eigen::VectorXi tri_idx(const Eigen::MatrixXd& v, const Eigen::MatrixXi& connect, const Eigen::MatrixXd& coord) {
    Eigen::VectorXi idx(v.rows());
    idx.setZero();

    std::vector<Eigen::Vector3d> normals(connect.rows());

    // 各三角形の法線ベクトルを事前計算
    #pragma omp parallel for
    for (int j = 0; j < connect.rows(); ++j) {
        Eigen::Vector3d t1 = coord.row(connect(j, 0));
        Eigen::Vector3d t2 = coord.row(connect(j, 1));
        Eigen::Vector3d t3 = coord.row(connect(j, 2));
        normals[j] = (t2 - t1).cross(t3 - t2).normalized();
    }

    // 各点について最も近い三角形を探索
    #pragma omp parallel for
    for (int i = 0; i < v.rows(); ++i) {
        const Eigen::Vector3d& p = v.row(i).transpose();
        double min_dist = std::numeric_limits<double>::infinity();
        int closest_idx = -1;

        for (int j = 0; j < connect.rows(); ++j) {
            Eigen::Vector3d t1 = coord.row(connect(j, 0));
            Eigen::Vector3d t2 = coord.row(connect(j, 1));
            Eigen::Vector3d t3 = coord.row(connect(j, 2));

            auto [flag, dist] = is_inside(p, t1, t2, t3, normals[j]);
            if (flag && dist < min_dist) {
                min_dist = dist;
                closest_idx = j;
            }
        }
        idx[i] = closest_idx;
    }

    return idx;
}

PYBIND11_MODULE(mapping_ls, m) {
    m.def("tri_idx", &tri_idx, "Get triangle index closest to each point",
          py::arg("v"), py::arg("connect"), py::arg("coord"));
}
