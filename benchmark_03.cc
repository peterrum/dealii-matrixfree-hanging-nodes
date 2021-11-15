#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

namespace dealii::GridGenerator
{
  template <int dim>
  void
  create_quadrant(Triangulation<dim> &tria, const unsigned int n_refinements)
  {
    // according to the description in A FLEXIBLE, PARALLEL, ADAPTIVE
    // GEOMETRIC MULTIGRID METHOD FOR FEM (Clevenger, Heister, Kanschat,
    // Kronbichler): https://arxiv.org/pdf/1904.03317.pdf

    hyper_cube(tria, -1.0, +1.0);

    if (n_refinements == 0)
      return;

    tria.refine_global(1);

    for (unsigned int i = 1; i < n_refinements; ++i)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              bool flag = true;
              for (int d = 0; d < dim; d++)
                if (cell->center()[d] > 0.0)
                  flag = false;
              if (flag)
                cell->set_refine_flag();
            }
        tria.execute_coarsening_and_refinement();
      }

    AssertDimension(tria.n_global_levels() - 1, n_refinements);
  }



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

    for (int i = 0; i < static_cast<int>(n_refinements) - 3; ++i)
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
} // namespace dealii::GridGenerator



template <int dim, int fe_degree, typename Number, typename MemorySpace>
class LaplaceOperator;

template <int dim, int fe_degree, typename Number>
class LaplaceOperator<dim, fe_degree, Number, MemorySpace::Host>
{
public:
  using VectorType =
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>;


  LaplaceOperator(const Mapping<dim> &             mapping,
                  const DoFHandler<dim> &          dof_handler,
                  const AffineConstraints<Number> &constraints,
                  const Quadrature<1> &            quadrature)
  {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_gradients;

    matrix_free.reinit(
      mapping, dof_handler, constraints, quadrature, additional_data);
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.cell_loop(&LaplaceOperator::local_apply, this, dst, src);
  }

private:
  void
  local_apply(const MatrixFree<dim, Number> &              data,
              VectorType &                                 dst,
              const VectorType &                           src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(false, true);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(false, true);
        phi.distribute_local_to_global(dst);
      }
  }

  MatrixFree<dim, Number> matrix_free;
};



template <unsigned int dim, const int degree, typename MemorySpace>
void
run(const std::string geometry_type, const bool print_details = true)
{
  ConvergenceTable table;

  const MPI_Comm comm = MPI_COMM_WORLD;

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace>;

  const unsigned n_repetitions = 100;

  for (unsigned int n_refinements = 4; n_refinements <= 6; ++n_refinements)
    {
      parallel::distributed::Triangulation<dim> tria(comm);

      if (geometry_type == "annulus")
        GridGenerator::create_annulus(tria, n_refinements);
      else if (geometry_type == "quadrant")
        GridGenerator::create_quadrant(tria, n_refinements);
      else
        AssertThrow(false, ExcMessage("Unknown geometry type!"));

      table.add_value("n_levels", tria.n_global_levels());
      table.add_value("degree", degree);

      const MappingQ1<dim> mapping;
      const FE_Q<dim>      fe(degree);
      const QGauss<1>      quadrature(degree + 1);

      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      AffineConstraints<Number> constraints;

      LaplaceOperator<dim, degree, Number, MemorySpace> laplace_operator(
        mapping, dof_handler, constraints, quadrature);

      VectorType src, dst;

      laplace_operator.initialize_dof_vector(src);
      laplace_operator.initialize_dof_vector(dst);

      src = 1.0;

      double min_time = 1e10;

      for (unsigned int i = 0; i < n_repetitions; ++i)
        {
          MPI_Barrier(MPI_COMM_WORLD);

          std::chrono::time_point<std::chrono::system_clock> temp =
            std::chrono::system_clock::now();

          laplace_operator.vmult(dst, src);

          MPI_Barrier(MPI_COMM_WORLD);

          min_time = std::min<double>(
            min_time,
            std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now() - temp)
                .count() /
              1e9);
        }

      min_time = Utilities::MPI::min(min_time, MPI_COMM_WORLD);

      if (print_details)
        {
          table.add_value("n_dofs", src.size());
        }

      table.add_value("time", min_time);
      table.set_scientific("time", true);

      if (print_details && Utilities::MPI::this_mpi_process(comm) == 0)
        {
          table.write_text(std::cout);
          std::cout << std::endl;
        }
    }

  if (print_details && Utilities::MPI::this_mpi_process(comm) == 0)
    {
      table.write_text(std::cout);
      std::cout << std::endl;
    }
}

/**
 * mpirun -np 40 ./benchmark_02 quadrant
 */
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = 3;
  const int          fe_degree = 4;

  const std::string geometry_type =
    argc > 1 ? std::string(argv[1]) : "quadrant";

  const std::string memory_type = argc > 2 ? std::string(argv[2]) : "host";

  if (memory_type == "host")
    run<dim, fe_degree, MemorySpace::Host>(geometry_type);
#ifdef DEAL_II_COMPILER_CUDA_AWARE
  else
    run<dim, fe_degree, MemorySpace::CUDA>(geometry_type);
#endif
  else AssertThrow(false, ExcNotImplemented());
}
