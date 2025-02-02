#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <vector>
#include <set>
#include <cmath>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3>                 Mesh;
typedef Mesh::Vertex_index                                  VertexIndex;
typedef Mesh::Edge_index                                    EdgeIndex;

namespace py = pybind11;
namespace PMP = CGAL::Polygon_mesh_processing;


// Build a mesh using the given vertex coordinates and connectivity
void build_mesh(const Eigen::MatrixXd& coord, 
                const Eigen::MatrixXi& connect, 
                Mesh& mesh)
{
    // Add vertices
    std::vector<VertexIndex> vertices;
    for (int i = 0; i < coord.rows(); ++i) {
        vertices.push_back(mesh.add_vertex(Kernel::Point_3(coord(i, 0), coord(i, 1), coord(i, 2))));
    }

    // Add faces
    for (int i = 0; i < connect.rows(); ++i) {
        mesh.add_face(vertices[connect(i, 0)], vertices[connect(i, 1)], vertices[connect(i, 2)]);
    }
}

// Collect vertices on sharp edges
void collect_vertices_on_sharp_edges(const Mesh& mesh, 
                                     std::set<VertexIndex>& sharp_vertices,
                                     const double angle_threshold_rad)
{
    for (EdgeIndex e : mesh.edges()) {
        if (!mesh.is_border(e)) { // Skip border edges
            auto h = mesh.halfedge(e, 0);
            auto face1 = mesh.face(h);
            auto face2 = mesh.face(mesh.opposite(h));

            if (face1 != Mesh::null_face() && face2 != Mesh::null_face()) {
                // Compute normal vectors
                Kernel::Vector_3 normal1 = CGAL::Polygon_mesh_processing::compute_face_normal(face1, mesh);
                Kernel::Vector_3 normal2 = CGAL::Polygon_mesh_processing::compute_face_normal(face2, mesh);

                // Calculate the angle
                double cos_angle = normal1 * normal2; // Inner product
                if (cos_angle < std::cos(angle_threshold_rad)) {
                    // Collect vertices on sharp edges
                    sharp_vertices.insert(mesh.target(h));
                    sharp_vertices.insert(mesh.target(mesh.opposite(h)));
                }
            }
        }
    }
}

// Fairing of a mesh
Eigen::MatrixXd fair_mesh(const Eigen::MatrixXd& coord, 
                          const Eigen::MatrixXi& connect,
                          const double angle_threshold)
{
  // Threshold angle in radians
  const double angle_threshold_rad = angle_threshold * M_PI / 180.0;
  // Build mesh
  Mesh mesh;
  build_mesh(coord, connect, mesh);
  // Collect vertices on sharp edges
  std::set<VertexIndex> sharp_vertices;
  collect_vertices_on_sharp_edges(mesh, sharp_vertices, angle_threshold_rad);
  // Convert sharp_vertices to a vector
  std::vector<VertexIndex> region(sharp_vertices.begin(), sharp_vertices.end());
  // Fairing
  PMP::fair(mesh, region);
  // Convert mesh to Eigen matrix
  Eigen::MatrixXd new_coord(coord.rows(), 3);
  for (VertexIndex v : mesh.vertices()) {
    auto p = mesh.point(v);
    new_coord(v.idx(), 0) = p.x();
    new_coord(v.idx(), 1) = p.y();
    new_coord(v.idx(), 2) = p.z();
  }
  return new_coord;
}

PYBIND11_MODULE(fair, m) {
    m.doc() = "Mesh fairing module using CGAL and pybind11";

    m.def("fair_mesh", &fair_mesh, 
          py::arg("coord"), py::arg("connect"), py::arg("angle_threshold"),
          "Apply fairing to a mesh defined by vertex coordinates and connectivity");
}
