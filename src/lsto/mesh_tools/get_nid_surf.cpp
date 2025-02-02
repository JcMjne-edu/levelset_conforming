#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;

struct HashTuple {
    size_t operator()(const std::tuple<int, int, int>& key) const {
        return std::hash<int>()(std::get<0>(key)) ^ std::hash<int>()(std::get<1>(key)) ^ std::hash<int>()(std::get<2>(key));
    }
};

std::pair<Eigen::VectorXi, Eigen::MatrixXi> get_nid_surf(const Eigen::MatrixXi &elems_tet) {
    int num_elems = elems_tet.rows();

    // 面の出現回数をカウントするためのハッシュマップ
    std::unordered_map<std::tuple<int, int, int>, int, HashTuple> face_count;

    // 各四面体の面をカウント
    for (int i = 0; i < num_elems; i++) {
        // 各面を生成
        auto f1 = std::make_tuple(elems_tet(i, 0), elems_tet(i, 1), elems_tet(i, 2));
        auto f2 = std::make_tuple(elems_tet(i, 0), elems_tet(i, 1), elems_tet(i, 3));
        auto f3 = std::make_tuple(elems_tet(i, 0), elems_tet(i, 2), elems_tet(i, 3));
        auto f4 = std::make_tuple(elems_tet(i, 1), elems_tet(i, 2), elems_tet(i, 3));

        // 各面をソートして順不同にする
        std::vector<int> faces[4] = {
            { std::get<0>(f1), std::get<1>(f1), std::get<2>(f1) },
            { std::get<0>(f2), std::get<1>(f2), std::get<2>(f2) },
            { std::get<0>(f3), std::get<1>(f3), std::get<2>(f3) },
            { std::get<0>(f4), std::get<1>(f4), std::get<2>(f4) }
        };

        for (int j = 0; j < 4; j++) {
            std::sort(faces[j].begin(), faces[j].end()); // 面をソート
            auto face_tuple = std::make_tuple(faces[j][0], faces[j][1], faces[j][2]);
            face_count[face_tuple]++;
        }
    }

    // 外部表面の面のみを抽出
    std::set<int> nid_surf_set;  // 重複を排除するためのセット
    std::set<std::pair<int, int>> edge_surf_set; // 重複を排除するためのセット

    for (const auto& [face, count] : face_count) {
        if (count == 1) {  // 一度しか出現しない面が外部表面
            int a, b, c;
            std::tie(a, b, c) = face;

            // nid_surfに頂点を追加（重複を排除）
            nid_surf_set.insert(a);
            nid_surf_set.insert(b);
            nid_surf_set.insert(c);

            // edge_surfにエッジを追加（重複を排除）
            edge_surf_set.emplace(std::min(a, b), std::max(a, b));
            edge_surf_set.emplace(std::min(b, c), std::max(b, c));
            edge_surf_set.emplace(std::min(c, a), std::max(c, a));
        }
    }

    // 出力の変換
    Eigen::VectorXi nid_surf_eigen(nid_surf_set.size());
    int index = 0;
    for (const auto& nid : nid_surf_set) {
        nid_surf_eigen(index++) = nid;
    }

    Eigen::MatrixXi edge_surf_eigen(edge_surf_set.size(), 2);
    index = 0;
    for (const auto& edge : edge_surf_set) {
        edge_surf_eigen(index, 0) = edge.first;
        edge_surf_eigen(index, 1) = edge.second;
        index++;
    }

    return std::make_pair(nid_surf_eigen, edge_surf_eigen);
}

PYBIND11_MODULE(get_nid_surf, m) {
    m.def("get_nid_surf", &get_nid_surf, "Calculate the surface node id and edge id of the tetrahedral mesh");
}