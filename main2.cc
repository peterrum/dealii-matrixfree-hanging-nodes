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

namespace dealii::GridGenerator
{
  template <int dim>
  void
  create_annulus(Triangulation<dim> &tria, const unsigned int n_refinements)
  {
    // according to the description in A FLEXIBLE, PARALLEL, ADAPTIVE
    // GEOMETRIC MULTIGRID METHOD FOR FEM (Clevenger, Heister, Kanschat,
    // Kronbichler): https://arxiv.org/pdf/1904.03317.pdf

    hyper_cube(tria, -1.0, +1.0);

    if (n_refinements == 0)
      return;

    for (int i = 0; i < static_cast<int>(n_refinements) - 3; i++)
      tria.refine_global();

    if (n_refinements >= 1)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            if (cell->center().norm() < 0.55)
              cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }

    if (n_refinements >= 2)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            if (0.3 <= cell->center().norm() && cell->center().norm() <= 0.43)
              cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }

    if (n_refinements >= 3)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            if (0.335 <= cell->center().norm() && cell->center().norm() <= 0.39)
              cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }

    // AssertDimension(tria.n_global_levels() - 1, n_refinements);
  }
}

template <int dim, int degree>
void
test(unsigned int n_ref_global, const bool apply_constraints)
{
  const unsigned int n_components = 1;
  using Number                    = double;
  using VectorizedArrayType       = VectorizedArray<Number>;

  Triangulation<dim> tria;
  GridGenerator::create_annulus(tria, n_ref_global);
  
  unsigned int n_cells_constrained = 0;

    for (const auto &cell : tria.active_cell_iterators())
      {
        bool flag = false;

        for (const auto f : cell->face_indices())
          if (!cell->at_boundary(f) &&
               cell->level() > cell->neighbor(f)->level())
            flag = true;

        if (flag)
          n_cells_constrained++;
      }
  
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  std::cout << n_cells_constrained << std::endl;

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

  const unsigned int n_cells = matrix_free.n_cell_batches ();
  const unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_lanes = VectorizedArrayType::size();

  AlignedVector<VectorizedArrayType> global_values(n_cells * n_dofs_per_cell);
  AlignedVector<VectorizedArrayType> values(n_dofs_per_cell);

  double min_time = 1e10;
  
  const auto & dof_info = matrix_free.get_dof_info();
  
  unsigned int counter_c = 0;
  unsigned int counter_mc = 0;

  for (unsigned r = 0; r < 10; ++r)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      
      std::chrono::time_point<std::chrono::system_clock> temp =
        std::chrono::system_clock::now();
      
      auto ptr = global_values.data();

      for (unsigned int cell = 0; cell < n_cells; cell += 1, ptr+=n_dofs_per_cell)
        {
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
            values[j] = ptr[j];
          
          if(apply_constraints)
          {
          std::array<internal::MatrixFreeFunctions::ConstraintKinds, n_lanes>
            constraint_mask;
          
          const unsigned int n_vectorization_actual = 
          matrix_free.n_active_entries_per_cell_batch(cell);
          
          bool hn_available = false;

          for (unsigned int v = 0; v < n_vectorization_actual; ++v)
          {
            const auto mask =
                dof_info.hanging_node_constraint_masks[cell * n_lanes + v];
            constraint_mask[v] = mask;
            
            hn_available |=
            (mask != internal::MatrixFreeFunctions::ConstraintKinds::unconstrained);
            
          }
          
          if(hn_available)
          {
          for (unsigned int v = n_vectorization_actual; v < n_lanes; ++v)
            constraint_mask[v] =
              internal::MatrixFreeFunctions::ConstraintKinds::unconstrained;
          
          if(r == 0)
          {
              counter_mc++;
              
          for (unsigned int v = 0; v < n_lanes; ++v)
              if(constraint_mask[v] != internal::MatrixFreeFunctions::ConstraintKinds::unconstrained)
              counter_c++;
          }
          
          //for (unsigned int v = 0; v < n_lanes; ++v)
          //    constraint_mask[v] = constraint_mask[v] & static_cast<internal::MatrixFreeFunctions::ConstraintKinds>(0b111111111);
          //
          //for (unsigned int v = 0; v < n_lanes; ++v)
          //    constraint_mask[v] = static_cast<internal::MatrixFreeFunctions::ConstraintKinds>(158);
          
          internal::MyFEEvaluationImplHangingNodes<
            dim,
            VectorizedArrayType,
            false>::template run<degree, degree + 1>(1,
                                                     fe_eval,
                                                     false,
                                                     constraint_mask,
                                                     values.data());
          
          internal::MyFEEvaluationImplHangingNodes<
            dim,
            VectorizedArrayType,
            false>::template run<degree, degree + 1>(1,
                                                     fe_eval,
                                                     true,
                                                     constraint_mask,
                                                     values.data());
          }
          }

          for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
            ptr[j] = values[j];
        }
      
      MPI_Barrier(MPI_COMM_WORLD);

      min_time =
        std::min<double>(min_time,
                         std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now() - temp)
                             .count() /
                           1e9);
    }
  
  min_time = Utilities::MPI::min(min_time, MPI_COMM_WORLD);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  std::cout << counter_c << " / " << tria.n_cells() << " " 
             << counter_mc << " / " << n_cells << " " << min_time << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  
  const unsigned int degree      = argc > 1 ? atoi(argv[1]) : 1;
  const unsigned int refinements = argc > 2 ? atoi(argv[2]) : 7;
  
  (void) degree;
  
  const unsigned precomp_degree = 1;
  
  test<3, precomp_degree>(refinements, true);
  test<3, precomp_degree>(refinements, false);
  
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  std::cout << std::endl;

}
