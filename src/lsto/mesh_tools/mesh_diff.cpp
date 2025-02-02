#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/boost/graph/selection.h>
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
py::tuple mesh_diff(const Eigen::MatrixXd& coords1,const Eigen::MatrixXi& connects1,
                    const Eigen::MatrixXd& coords2,const Eigen::MatrixXi& connects2,
                    bool remesh = false, double target_edge_length = 1.0) {
  
  Mesh mesh1, mesh2;

  // Load vertices and faces into mesh1
  for (int i = 0; i < coords1.rows(); ++i) {
    mesh1.add_vertex(K::Point_3(coords1(i, 0), coords1(i, 1), coords1(i, 2)));
  }
  for (int i = 0; i < connects1.rows(); ++i) {
    std::vector<Mesh::Vertex_index> face_vertices;
    for (int j = 0; j < 3; ++j) {
      face_vertices.push_back(Mesh::Vertex_index(connects1(i, j)));
    }
    mesh1.add_face(face_vertices);
  }

  // Load vertices and faces into mesh2
  for (int i = 0; i < coords2.rows(); ++i) {
    mesh2.add_vertex(K::Point_3(coords2(i, 0), coords2(i, 1), coords2(i, 2)));
  }
  for (int i = 0; i < connects2.rows(); ++i) {
    std::vector<Mesh::Vertex_index> face_vertices;
    for (int j = 0; j < 3; ++j) {
      face_vertices.push_back(Mesh::Vertex_index(connects2(i, j)));
    }
    mesh2.add_face(face_vertices);
  }

  // Create a property on edges to indicate whether they are constrained
  Mesh::Property_map<edge_descriptor,bool> is_constrained_map =
    mesh1.add_property_map<edge_descriptor,bool>("e:is_constrained", false).first;
  
  // Compute mesh difference
  bool valid_difference = PMP::corefine_and_compute_difference(
    mesh1, mesh2, mesh1, params::all_default(), params::all_default(),
    params::edge_is_constrained_map(is_constrained_map));

  if (!valid_difference) {
    throw std::runtime_error("Difference could not be computed");
  }

  if (remesh){
    // Collect selected faces
    std::vector<face_descriptor> selected_faces;
    std::vector<bool> is_selected(num_faces(mesh1), false);
    for(edge_descriptor e : edges(mesh1))
    {
      if (is_constrained_map[e])
      {
        for(halfedge_descriptor h : halfedges_around_target(halfedge(e, mesh1), mesh1))
        {
          if (!is_border(h, mesh1))
          {
            face_descriptor f = face(h, mesh1);
            if (!is_selected[f])
            {
              selected_faces.push_back(f);
              is_selected[f] = true;
            }
          }
        }
      }
    }

    // Expand face selection
    CGAL::expand_face_selection(selected_faces, mesh1, 1, 
      Vector_pmap_wrapper(is_selected), std::back_inserter(selected_faces));

    PMP::isotropic_remeshing(selected_faces, target_edge_length, mesh1,
                            params::edge_is_constrained_map(is_constrained_map));
  }
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
PYBIND11_MODULE(mesh_diff, m) {
    m.def("mesh_diff", &mesh_diff, 
          "Compute mesh1 - mesh2 and output vertices and connectivities to binary files",
          py::arg("coords1"),py::arg("connects1"),py::arg("coords2"),py::arg("connects2"),
          py::arg("remesh")=false,py::arg("target_edge_length")=1.0);
}
