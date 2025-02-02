#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/boost/graph/selection.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
typedef CGAL::Surface_mesh<K::Point_3>                        Mesh;
typedef boost::graph_traits<Mesh>::halfedge_descriptor        halfedge_descriptor;
typedef boost::graph_traits<Mesh>::edge_descriptor            edge_descriptor;
typedef boost::graph_traits<Mesh>::face_descriptor            face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;
namespace params = PMP::parameters;
namespace py = pybind11;

double compute_aspect_ratio(face_descriptor f, const Mesh& mesh) {
  auto h = halfedge(f, mesh);
  const auto& p0 = mesh.point(target(h, mesh));
  const auto& p1 = mesh.point(target(next(h, mesh), mesh));
  const auto& p2 = mesh.point(target(next(next(h, mesh), mesh), mesh));

  double a = std::sqrt(CGAL::squared_distance(p0, p1));
  double b = std::sqrt(CGAL::squared_distance(p1, p2));
  double c = std::sqrt(CGAL::squared_distance(p2, p0));

  double s = (a + b + c) / 2.0;
  double area = std::sqrt(s * (s - a) * (s - b) * (s - c));

  // circumradius = a*b*c / 4*area
  return (a * b * c) / (4.0 * area);
}

void selective_remeshing(Mesh& mesh, double max_aspect_ratio, double target_edge_length) {
  std::vector<face_descriptor> selected_faces;
  for (auto f : faces(mesh)) {
    if (compute_aspect_ratio(f, mesh) > max_aspect_ratio) {
      selected_faces.push_back(f);
    }
  }

  PMP::isotropic_remeshing(
      selected_faces,
      target_edge_length,
      mesh,
      CGAL::Polygon_mesh_processing::parameters::all_default());
}

struct Vector_pmap_wrapper
{
  std::vector<bool>& vect;
  Vector_pmap_wrapper(std::vector<bool>& v) : vect(v) {}
  friend bool get(const Vector_pmap_wrapper& m, face_descriptor f)
  {
    return m.vect[f];
  }
  friend void put(const Vector_pmap_wrapper& m, face_descriptor f, bool b)
  {
    m.vect[f]=b;
  }
};

//int main(int argc, char* argv[])
py::tuple mesh_smoothing(const Eigen::MatrixXd& coords,const Eigen::MatrixXi& connects,
                         double target_edge_length = 1.0) {
  
  Mesh mesh1;

  // Load vertices and faces into mesh1
  #pragma omp parallel for
  for (int i = 0; i < coords.rows(); ++i) {
    mesh1.add_vertex(K::Point_3(coords(i, 0), coords(i, 1), coords(i, 2)));
  }
  #pragma omp parallel for
  for (int i = 0; i < connects.rows(); ++i) {
    std::vector<Mesh::Vertex_index> face_vertices;
    for (int j = 0; j < 3; ++j) {
      face_vertices.push_back(Mesh::Vertex_index(connects(i, j)));
    }
    mesh1.add_face(face_vertices);
  }

  PMP::isotropic_remeshing(faces(mesh1), target_edge_length, mesh1,
    CGAL::Polygon_mesh_processing::parameters::all_default());
  
  // Collect vertices into an Eigen matrix
  Eigen::MatrixXd verts(mesh1.number_of_vertices(), 3);
  int idx = 0;
  for (auto v : vertices(mesh1)) {
    K::Point_3 p = mesh1.point(v);
    verts(idx, 0) = p.x();
    verts(idx, 1) = p.y();
    verts(idx, 2) = p.z();
    idx++;
  }

  // Collect faces into an Eigen matrix
  Eigen::MatrixXi connectivities(mesh1.number_of_faces(), 3);
  idx = 0;
  for (auto f : faces(mesh1)) {
    auto h = halfedge(f, mesh1);
    for (int j = 0; j < 3; ++j) {
      connectivities(idx, j) = static_cast<int>(target(h, mesh1));
      h = next(h, mesh1);
    }
    idx++;
  }

  // Return as a tuple
  return py::make_tuple(verts, connectivities);
}

// Pybind11 module
PYBIND11_MODULE(mesh_smoothing, m) {
    m.def("mesh_smoothing", &mesh_smoothing, 
          "Smooth a mesh using isotropic remeshing",
          py::arg("coords"),py::arg("connects"),py::arg("target_edge_length")=1.0);
}
