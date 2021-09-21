#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "evaluation_kernels.h"

using namespace dealii;

template <int dim, int degree>
void
test(unsigned int                                          size_approx,
     const internal::MatrixFreeFunctions::ConstraintKinds &mask,
     const unsigned int                                    mask_rep)
{
  const unsigned int n_components = 1;
  using Number                    = double;
  using VectorizedArrayType       = VectorizedArray<Number>;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);

  const MappingQ1<dim> mapping;
  const FE_Q<dim>      fe(degree);
  const QGauss<dim>    quadrature(degree + 1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number>                    constraints;
  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quadrature);

  FEEvaluation<dim, -1, -1, n_components, Number, VectorizedArrayType> fe_eval(
    matrix_free);

  const unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();

  AlignedVector<VectorizedArrayType> global_values(
    (size_approx / n_dofs_per_cell) * n_dofs_per_cell);
  AlignedVector<VectorizedArrayType> values(n_dofs_per_cell);

  std::array<internal::MatrixFreeFunctions::ConstraintKinds,
             VectorizedArrayType::size()>
    masks;

  for (unsigned int i = 0; i < mask_rep; ++i)
    masks[i] = mask;

  double min_time = 1e10;

  for (unsigned r = 0; r < 10; ++r)
    {
      std::chrono::time_point<std::chrono::system_clock> temp =
        std::chrono::system_clock::now();

      for (unsigned int i = 0; i < global_values.size(); i += n_dofs_per_cell)
        {
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
            values[j] = global_values[i + j];

          internal::FEEvaluationImplHangingNodes<
            dim,
            VectorizedArrayType,
            false>::template run<degree, degree + 1>(1,
                                                     fe_eval,
                                                     false,
                                                     masks,
                                                     values.data());

          for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
            global_values[i + j] = values[j];
        }

      min_time =
        std::min<double>(min_time,
                         std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now() - temp)
                             .count() /
                           1e9);
    }

  std::cout << min_time << " ";
}

int
main(int argc, char **argv)
{
  const unsigned int degree      = argc > 1 ? atoi(argv[1]) : 1;
  const unsigned int size_approx = argc > 2 ? atoi(argv[2]) : 10000;

  using namespace dealii::internal::MatrixFreeFunctions;

  const auto unconstrained = ConstraintKinds::unconstrained;
  const auto type_x        = ConstraintKinds::type_x;
  const auto type_y        = ConstraintKinds::type_y;
  const auto type_z        = ConstraintKinds::type_z;
  const auto face_x        = ConstraintKinds::face_x;
  const auto face_y        = ConstraintKinds::face_y;
  const auto face_z        = ConstraintKinds::face_z;
  const auto edge_xy       = ConstraintKinds::edge_xy;
  const auto edge_yz       = ConstraintKinds::edge_yz;
  const auto edge_zx       = ConstraintKinds::edge_zx;

  const std::vector<internal::MatrixFreeFunctions::ConstraintKinds> masks{
    unconstrained,                      // unconstrained
    edge_yz | type_y | type_z,          // edge 2
    edge_yz | type_y | type_z | type_x, //
    edge_yz | type_z,                   // edge 3
    edge_yz | type_z | type_x,          //
    edge_yz | type_y,                   // edge 6
    edge_yz | type_y | type_x,          //
    edge_yz,                            // edge 7
    edge_yz | type_x,                   //
    edge_zx | type_x | type_z,          // edge 0
    edge_zx | type_x | type_z | type_y, //
    edge_zx | type_z,                   // edge 1
    edge_zx | type_z | type_y,          //
    edge_zx | type_x,                   // edge 4
    edge_zx | type_x | type_y,          //
    edge_zx,                            // edge 5
    edge_zx | type_y,                   //
    edge_xy | type_x | type_y,          // edge 8
    edge_xy | type_x | type_y | type_z, //
    edge_xy | type_y,                   // edge 9
    edge_xy | type_y | type_z,          //
    edge_xy | type_x,                   // edge 10
    edge_xy | type_x | type_z,          //
    edge_xy,                            // edge 11
    edge_xy | type_z,                   //
    face_x | type_x,                    // face 0
    face_x,                             // face 1
    face_y | type_y,                    // face 2
    face_y,                             // face 3
    face_z | type_z,                    // face 4
    face_z                              // face 5
  };

  const unsigned precomp_degree = 1;

  AssertDimension(precomp_degree, degree);

  for (const auto mask : masks)
    {
      for (unsigned int i = 1; i <= VectorizedArray<double>::size(); ++i)
        test<3, precomp_degree>(size_approx, mask, i);
      std::cout << std::endl;
    }
}