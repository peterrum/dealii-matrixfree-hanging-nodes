#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "evaluation_kernels.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

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

  for (unsigned int i = mask_rep; i < VectorizedArray<Number>::size(); ++i)
    masks[i] = internal::MatrixFreeFunctions::ConstraintKinds::unconstrained;

  double min_time = 1e10;

#ifdef LIKWID_PERFMON
  std::string label = "data_" + std::to_string(mask_rep);
  LIKWID_MARKER_START(label.c_str());
#endif

  for (unsigned r = 0; r < 100; ++r)
    {
      MPI_Barrier(MPI_COMM_WORLD);

      std::chrono::time_point<std::chrono::system_clock> temp =
        std::chrono::system_clock::now();


      for (unsigned int i = 0; i < global_values.size(); i += n_dofs_per_cell)
        {
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
            values[j] = global_values[i + j];

          internal::MyFEEvaluationImplHangingNodes<
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

      MPI_Barrier(MPI_COMM_WORLD);

      min_time =
        std::min<double>(min_time,
                         std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now() - temp)
                             .count() /
                           1e9);
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP(label.c_str());
#endif

  min_time = Utilities::MPI::min(min_time, MPI_COMM_WORLD);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << min_time << " ";
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

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
    unconstrained, edge_yz | type_y | type_z, face_x};

  // const std::vector<internal::MatrixFreeFunctions::ConstraintKinds> masks{
  //  edge_yz | type_y | type_z};

  const unsigned precomp_degree = 1;

  AssertDimension(precomp_degree, degree);

  for (const auto mask : masks)
    {
      for (unsigned int i = VectorizedArray<double>::size();
           i <= VectorizedArray<double>::size();
           ++i)
        test<3, precomp_degree>(size_approx, mask, i);
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << std::endl;
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
