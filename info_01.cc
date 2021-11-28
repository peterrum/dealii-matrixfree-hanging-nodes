#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/matrix_free/matrix_free.h>

#include "benchmark.h"

using namespace dealii;

template <int dim, typename Number = double>
void
run(const std::string &geometry_type,
    const unsigned int n_refinements,
    const unsigned int degree,
    ConvergenceTable & table)
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  if (geometry_type == "annulus")
    GridGenerator::create_annulus(tria, n_refinements);
  else if (geometry_type == "quadrant")
    GridGenerator::create_quadrant(tria, n_refinements);
  else
    AssertThrow(false, ExcMessage("Unknown geometry type!"));

  table.add_value("n_refinements", n_refinements);

  table.add_value("n_levels", tria.n_global_levels());

  const MappingQ1<dim> mapping;
  const FE_Q<dim>      fe(degree);
  const QGauss<dim>    quadrature(degree + 1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  if (true)
    {
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.mapping_update_flags = update_gradients;

      AffineConstraints<Number> constraints;

      MatrixFree<dim, Number> matrix_free;
      matrix_free.reinit(
        mapping, dof_handler, constraints, quadrature, additional_data);


      const auto temp = MemoryConsumption::memory_consumption(
        matrix_free.get_dof_info().hanging_node_constraint_masks);

      const auto mem =
        Utilities::MPI::sum(static_cast<double>(temp), MPI_COMM_WORLD);

      table.add_value("mem_sp", mem);

      double n_cells_n  = 0;
      double n_cells_hn = 0;

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          const unsigned int n_vectorization_actual =
            matrix_free.n_active_entries_per_cell_batch(cell);

          for (unsigned int v = 0; v < n_vectorization_actual; ++v)
            {
              const auto mask =
                matrix_free.get_dof_info()
                      .hanging_node_constraint_masks.size() == 0 ?
                  internal::MatrixFreeFunctions::ConstraintKinds::
                    unconstrained :
                  matrix_free.get_dof_info().hanging_node_constraint_masks
                    [cell * VectorizedArray<Number>::size() + v];

              if (mask ==
                  internal::MatrixFreeFunctions::ConstraintKinds::unconstrained)
                n_cells_n++;
              else
                n_cells_hn++;
            }
        }

      n_cells_n =
        Utilities::MPI::sum(static_cast<double>(n_cells_n), MPI_COMM_WORLD);
      n_cells_hn =
        Utilities::MPI::sum(static_cast<double>(n_cells_hn), MPI_COMM_WORLD);

      table.add_value("n_cells_n", n_cells_n);
      table.add_value("n_cells_hn", n_cells_hn);
    }

  if (false)
    {
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.mapping_update_flags = update_gradients;

      AffineConstraints<Number> constraints;
      IndexSet                  locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      constraints.reinit(locally_relevant_dofs);

      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      constraints.close();

      {
        const auto mem =
          Utilities::MPI::sum(static_cast<double>(
                                constraints.memory_consumption()),
                              MPI_COMM_WORLD);
        table.add_value("mem_matrix", mem);
      }

      additional_data.use_fast_hanging_node_algorithm = false;

      MatrixFree<dim, Number> matrix_free;
      matrix_free.reinit(
        mapping, dof_handler, constraints, quadrature, additional_data);

      {
        auto temp = MemoryConsumption::memory_consumption(
          matrix_free.get_dof_info().constraint_indicator);

        for (unsigned int i = 0; i < matrix_free.n_constraint_pool_entries();
             ++i)
          {
            temp += sizeof(Number) * (matrix_free.constraint_pool_end(i) -
                                      matrix_free.constraint_pool_begin(i));
          }

        const auto mem =
          Utilities::MPI::sum(static_cast<double>(temp), MPI_COMM_WORLD);

        table.add_value("mem_gp", mem);
        table.add_value("groups_gp", matrix_free.n_constraint_pool_entries());
      }
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  constexpr int dim = 3;

  {
    ConvergenceTable table;
    for (unsigned int i = 5; i <= 9; ++i)
      run<dim>("annulus", i, 4, table);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        table.write_text(std::cout);
        std::cout << std::endl;
      }
  }

  {
    ConvergenceTable table;
    for (unsigned int i = 5; i <= 9; ++i)
      run<dim>("quadrant", i, 4, table);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        table.write_text(std::cout);
        std::cout << std::endl;
      }
  }
}
