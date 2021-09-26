#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "benchmark.h"

template <int dim, int spacedim = dim>
class Helper
{
public:
  Helper(const Triangulation<dim, spacedim> &triangulation)
  {
    if (dim == 3)
      {
        const unsigned int n_raw_lines = triangulation.n_raw_lines();
        this->line_to_cells.resize(n_raw_lines);

        // In 3D, we can have DoFs on only an edge being constrained (e.g. in a
        // cartesian 2x2x2 grid, where only the upper left 2 cells are refined).
        // This sets up a helper data structure in the form of a mapping from
        // edges (i.e. lines) to neighboring cells.

        // Mapping from an edge to which children that share that edge.
        const unsigned int line_to_children[12][2] = {{0, 2},
                                                      {1, 3},
                                                      {0, 1},
                                                      {2, 3},
                                                      {4, 6},
                                                      {5, 7},
                                                      {4, 5},
                                                      {6, 7},
                                                      {0, 4},
                                                      {1, 5},
                                                      {2, 6},
                                                      {3, 7}};

        std::vector<std::vector<
          std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                    unsigned int>>>
          line_to_inactive_cells(n_raw_lines);

        // First add active and inactive cells to their lines:
        for (const auto &cell : triangulation.cell_iterators())
          {
            for (unsigned int line = 0; line < GeometryInfo<3>::lines_per_cell;
                 ++line)
              {
                const unsigned int line_idx = cell->line(line)->index();
                if (cell->is_active())
                  line_to_cells[line_idx].push_back(std::make_pair(cell, line));
                else
                  line_to_inactive_cells[line_idx].push_back(
                    std::make_pair(cell, line));
              }
          }

        // Now, we can access edge-neighboring active cells on same level to
        // also access of an edge to the edges "children". These are found from
        // looking at the corresponding edge of children of inactive edge
        // neighbors.
        for (unsigned int line_idx = 0; line_idx < n_raw_lines; ++line_idx)
          {
            if ((line_to_cells[line_idx].size() > 0) &&
                line_to_inactive_cells[line_idx].size() > 0)
              {
                // We now have cells to add (active ones) and edges to which
                // they should be added (inactive cells).
                const auto &inactive_cell =
                  line_to_inactive_cells[line_idx][0].first;
                const unsigned int neighbor_line =
                  line_to_inactive_cells[line_idx][0].second;

                for (unsigned int c = 0; c < 2; ++c)
                  {
                    const auto &child =
                      inactive_cell->child(line_to_children[neighbor_line][c]);
                    const unsigned int child_line_idx =
                      child->line(neighbor_line)->index();

                    // Now add all active cells
                    for (const auto &cl : line_to_cells[line_idx])
                      line_to_cells[child_line_idx].push_back(cl);
                  }
              }
          }
      }
  }


  bool
  is_constrained(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    return is_face_constrained(cell) || is_edge_constrained(cell);
  }

  bool
  is_face_constrained(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    if (cell->is_locally_owned())
      for (unsigned int f : cell->face_indices())
        if (!cell->at_boundary(f) &&
            (cell->level() > cell->neighbor(f)->level()))
          return true;

    return false;
  }

  bool
  is_edge_constrained(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    if (dim == 3)
      if (cell->is_locally_owned())
        for (const auto line : cell->line_indices())
          for (const auto &other_cell :
               line_to_cells[cell->line(line)->index()])
            if (cell->level() > other_cell.first->level())
              return true;

    return false;
  }

private:
  std::vector<
    std::vector<std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                          unsigned int>>>
    line_to_cells;
};

namespace dealii::parallel
{
  template <int dim, int spacedim>
  std::function<
    unsigned int(const typename Triangulation<dim, spacedim>::cell_iterator &,
                 const typename Triangulation<dim, spacedim>::CellStatus)>
  hanging_nodes_weighting(const Helper<dim, spacedim> &helper,
                          const double                 weight)
  {
    return [&helper, weight](const auto &cell, const auto &) -> unsigned int {
      if (cell->is_locally_owned() == false)
        return 10000;

      if (helper.is_constrained(cell))
        return 10000 * weight;
      else
        return 10000;
    };
  }


} // namespace dealii::parallel

template <unsigned int dim, unsigned int max_degree, unsigned int degree_ = 1>
void
run(const std::string  geometry_type,
    const unsigned int n_refinements,
    const unsigned int degree,
    const bool         print_details)
{
  if (degree != degree_)
    {
      run<dim, max_degree, std::min(max_degree, degree_ + 1)>(geometry_type,
                                                              n_refinements,
                                                              degree,
                                                              print_details);
      return;
    }

  ConvergenceTable table;

  const MPI_Comm comm = MPI_COMM_WORLD;

  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;

  const unsigned n_repetitions = 100;

  for (unsigned int weight = 100; weight <= 1000; weight += 10)
    {
      parallel::distributed::Triangulation<dim> tria_pdt(comm);

      if (geometry_type == "annulus")
        GridGenerator::create_annulus(tria_pdt, n_refinements);
      else if (geometry_type == "quadrant")
        GridGenerator::create_quadrant(tria_pdt, n_refinements);
      else if (geometry_type == "quadrant_flexible")
        GridGenerator::create_quadrant_flexible(tria_pdt, n_refinements);
      else
        AssertThrow(false, ExcMessage("Unknown geometry type!"));

      table.add_value("n_levels", tria_pdt.n_global_levels());
      table.add_value("degree", degree);
      table.add_value("weight", weight / 100.);

      const Helper<dim> helper(tria_pdt);

      const auto weight_function =
        parallel::hanging_nodes_weighting(helper, weight / 100.);

      dealii::RepartitioningPolicyTools::CellWeightPolicy<dim> policy(
        weight_function);

      parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
      tria_pft.create_triangulation(
        TriangulationDescription::Utilities::
          create_description_from_triangulation(tria_pdt,
                                                policy.partition(tria_pdt)));

      tria_pdt.signals.cell_weight.connect(weight_function);

      tria_pdt.repartition();

      const auto runner = [degree, &table](const auto &       tria,
                                           const std::string &label,
                                           const bool         print_details) {
        const MappingQ1<dim> mapping;
        const FE_Q<dim>      fe(degree);
        const QGauss<dim>    quadrature(degree + 1);

        DoFHandler<dim> dof_handler(tria);
        dof_handler.distribute_dofs(fe);

        AffineConstraints<Number> constraints;

        typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
          additional_data;
        additional_data.mapping_update_flags = update_gradients;

        MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
        matrix_free.reinit(
          mapping, dof_handler, constraints, quadrature, additional_data);

        VectorType src, dst;

        matrix_free.initialize_dof_vector(src);
        matrix_free.initialize_dof_vector(dst);

        src = 1.0;

        double min_time = 1e10;

        for (unsigned int i = 0; i < n_repetitions; ++i)
          {
            MPI_Barrier(MPI_COMM_WORLD);

            std::chrono::time_point<std::chrono::system_clock> temp =
              std::chrono::system_clock::now();

            matrix_free.template cell_loop<VectorType, VectorType>(
              [](const auto &matrix_free,
                 auto &      dst,
                 const auto &src,
                 auto        range) {
                FEEvaluation<dim, -1, 0, 1, Number> phi(matrix_free, range);

                for (unsigned cell = range.first; cell < range.second; ++cell)
                  {
                    phi.reinit(cell);

                    phi.gather_evaluate(src, EvaluationFlags::gradients);

                    for (unsigned int q = 0; q < phi.n_q_points; ++q)
                      phi.submit_gradient(phi.get_gradient(q), q);

                    phi.integrate_scatter(EvaluationFlags::gradients, dst);
                  }
              },
              dst,
              src);

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

        table.add_value(label + "_t", min_time);
        table.set_scientific(label + "_t", true);
      };

      runner(tria_pdt, "pdt", true);
      runner(tria_pft, "pft", false);

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
 * mpirun -np 40 ./benchmark_02 quadrant 7 4
 * mpirun -np 40 ./benchmark_02 annulus 8 4
 */
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim        = 3;
  const unsigned int max_degree = 4;

  const std::string geometry_type =
    argc > 1 ? std::string(argv[1]) : "quadrant";
  const unsigned int n_refinements = argc > 2 ? atoi(argv[2]) : 6;
  const unsigned int degree        = argc > 3 ? atoi(argv[3]) : 1;
  const bool         print_details = true;

  AssertThrow(degree <= max_degree, ExcNotImplemented());

  run<dim, max_degree>(geometry_type, n_refinements, degree, print_details);
}
