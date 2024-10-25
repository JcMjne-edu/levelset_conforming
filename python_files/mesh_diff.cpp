#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
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
typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
typedef CGAL::Surface_mesh<K::Point_3>                        Mesh;
typedef boost::graph_traits<Mesh>::halfedge_descriptor        halfedge_descriptor;
typedef boost::graph_traits<Mesh>::edge_descriptor            edge_descriptor;
typedef boost::graph_traits<Mesh>::face_descriptor            face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;
namespace params = PMP::parameters;
namespace py = pybind11;

//int main(int argc, char* argv[])
py::tuple mesh_diff(py::array_t<double> coords1_arr,py::array_t<int> connects1_arr,
                   py::array_t<double> coords2_arr,py::array_t<int> connects2_arr) {
  
  // Extract input data from NumPy arrays
  py::buffer_info coords_info1 = coords1_arr.request();
  py::buffer_info conn_info1 = connects1_arr.request();
  py::buffer_info coords_info2 = coords2_arr.request();
  py::buffer_info conn_info2 = connects2_arr.request();

  double* coords1 = static_cast<double*>(coords_info1.ptr);
  int* connects1 = static_cast<int*>(conn_info1.ptr);
  double* coords2 = static_cast<double*>(coords_info2.ptr);
  int* connects2 = static_cast<int*>(conn_info2.ptr);

  int num_vertices1 = coords_info1.shape[0];
  int num_faces1 = conn_info1.shape[0];
  int num_vertices2 = coords_info2.shape[0];
  int num_faces2 = conn_info2.shape[0];
  
  //const std::string filename1 = (argc > 1) ? argv[1] : CGAL::data_file_path("meshes/blobby.off");
  //const std::string filename2 = (argc > 2) ? argv[2] : CGAL::data_file_path("meshes/eight.off");
  Mesh mesh1, mesh2;
  //if(!PMP::IO::read_polygon_mesh(filename1, mesh1) || !PMP::IO::read_polygon_mesh(filename2, mesh2))
  //{
  //  std::cerr << "Invalid input." << std::endl;
  //  return 1;
  //}
  for (int i = 0; i < num_vertices1; ++i) {
    mesh1.add_vertex(K::Point_3(coords1[3*i], coords1[3*i+1], coords1[3*i+2]));
  }
  for (int i = 0; i < num_faces1; ++i) {
    std::vector<Mesh::Vertex_index> face_vertices1;
    for (int j = 0; j < 3; ++j) {
      face_vertices1.push_back(Mesh::Vertex_index(connects1[3*i+j]));
    }
    mesh1.add_face(face_vertices1);
  }

  for (int i = 0; i < num_vertices2; ++i) {
    mesh2.add_vertex(K::Point_3(coords2[3*i], coords2[3*i+1], coords2[3*i+2]));
  }
  for (int i = 0; i < num_faces2; ++i) {
    std::vector<Mesh::Vertex_index> face_vertices2;
    for (int j = 0; j < 3; ++j) {
      face_vertices2.push_back(Mesh::Vertex_index(connects2[3*i+j]));
    }
    mesh2.add_face(face_vertices2);
  }

  //create a property on edges to indicate whether they are constrained
  Mesh::Property_map<edge_descriptor,bool> is_constrained_map =
    mesh1.add_property_map<edge_descriptor,bool>("e:is_constrained", false).first;
  // update mesh1 to contain the mesh bounding the difference
  // of the two input volumes.
  bool valid_difference =
    PMP::corefine_and_compute_difference(mesh1,
                                         mesh2,
                                         mesh1,
                                         params::all_default(), // default parameters for mesh1
                                         params::all_default(), // default parameters for mesh2
                                         params::edge_is_constrained_map(is_constrained_map));
  //if (valid_difference)
  //{
  //  //std::cout << "Difference was successfully computed\n";
  //  //CGAL::IO::write_polygon_mesh("difference.off", mesh1, CGAL::parameters::stream_precision(17));
  //  // --- Output vertices to coordinates.bin ---
  //  std::ofstream output("coordinates.bin", std::ios::binary);
  //  for (auto v : vertices(mesh1)) {
  //    K::Point_3 p = mesh1.point(v);
  //    output.write((char*)&p, sizeof(K::Point_3));
  //  }
  //  output.close();
//
  //  // --- Output index of vertices in each face to connectvities.bin ---
  //  std::ofstream output_faces("connectivities.bin", std::ios::binary);
  //  
  //  for (auto f : faces(mesh1)) {
  //    auto h = halfedge(f, mesh1);
  //    std::vector<int> vertex_indices;
  //    do {
  //            int vertex_index = static_cast<int>(target(h, mesh1));  // ハーフエッジの終点から頂点インデックスを取得
  //            vertex_indices.push_back(vertex_index);
  //            h = next(h, mesh1);  // move to the next halfedge
  //        } while (h != halfedge(f, mesh1));
  //    output_faces.write((char*)vertex_indices.data(), vertex_indices.size() * sizeof(int));
  //  }
  //  output_faces.close();
  //}
  //else
  //{
  //  std::cout << "Difference could not be computed\n";
  //  return 1;
  //}

  // Collect the vertices into a vector
  std::vector<double> verts;
  for (auto v : vertices(mesh1)) {
      K::Point_3 p = mesh1.point(v);
      verts.push_back(p.x());
      verts.push_back(p.y());
      verts.push_back(p.z());
  }

  // Collect the faces into a vector
  std::vector<int> connectivities;
  for (auto f : faces(mesh1)) {
      auto h = halfedge(f, mesh1);
      do {
          connectivities.push_back(static_cast<int>(target(h, mesh1)));
          h = next(h, mesh1);
      } while (h != halfedge(f, mesh1));
  }

  // Create NumPy arrays for output
  py::array_t<double> coords({(int)verts.size() / 3, 3}, verts.data());
  py::array_t<int> connects({(int)connectivities.size() / 3, 3}, connectivities.data());

  // Return as a tuple
  return py::make_tuple(coords, connects);
}

// Pybind11 module
PYBIND11_MODULE(mesh_diff, m) {
    m.def("mesh_diff", &mesh_diff, 
          "Compute mesh1 - mesh2 and output vertices and connectivities to binary files",
          py::arg("coords1"), py::arg("connects1"), py::arg("coords2"), py::arg("connects2"));
}
