#include "benchmark_01.h"

template <unsigned int dim, unsigned int max_degree, unsigned int degree_ = 1>
void
run(const std::string  geometry_type,
    const unsigned int min_n_refinements,
    const unsigned int max_n_refinements,
    const unsigned int degree,
    const bool         print_details)
{
  if (degree != degree_)
    {
      run<dim, max_degree, std::min(max_degree, degree_ + 1)>(geometry_type,
                                                              min_n_refinements,
                                                              max_n_refinements,
                                                              degree,
                                                              print_details);
      return;
    }

  ConvergenceTable table;

  for (unsigned int n_refinements = min_n_refinements;
       n_refinements <= max_n_refinements;
       ++n_refinements)
    {
      Test<dim, max_degree> test(geometry_type, n_refinements);

      const auto info = test.get_info(print_details);

      table.add_value("n_levels", info.n_levels);
      table.add_value("degree", degree);
      table.add_value("n_cells", info.n_cells);
      table.add_value("n_cells_n", info.n_cells_n);
      table.add_value("n_cells_hn", info.n_cells_hn);
      table.add_value("n_macro_cells", info.n_macro_cells);
      table.add_value("n_macro_cells_n", info.n_macro_cells_n);
      table.add_value("n_macro_cells_hn", info.n_macro_cells_hn);

      const auto compute_cost = [&](const auto t_n, const auto t_hn) {
        if (info.n_cells_hn == 0)
          return 1.0;

        return std::max((t_hn / (t_n / (info.n_cells_n + info.n_cells_hn)) -
                         info.n_cells_n) /
                          info.n_cells_hn,
                        1.0);
      };

      // DG (C)
      const auto t0 = test.run(false, false, false);
      const auto t1 = test.run(false, true, false);

      table.add_value("t0", t0);
      table.set_scientific("t0", true);
      table.add_value("t1", t1);
      table.set_scientific("t1", true);
      table.add_value("eta1", compute_cost(t0, t1));
      table.set_scientific("eta1", true);

      // DG (SC)
      const auto t2 = test.run(false, false, true);
      const auto t3 = test.run(false, true, true);

      table.add_value("t2", t2);
      table.set_scientific("t2", true);
      table.add_value("t3", t3);
      table.set_scientific("t3", true);
      table.add_value("eta3", compute_cost(t2, t3));
      table.set_scientific("eta3", true);

      // CG (SC)
      const auto t4 = test.run(true, false, true);
      const auto t5 = test.run(true, true, true);

      table.add_value("t4", t4);
      table.set_scientific("t4", true);
      table.add_value("t5", t5);
      table.set_scientific("t5", true);
      table.add_value("eta5", compute_cost(t4, t5));
      table.set_scientific("eta5", true);

      if (print_details &&
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          table.write_text(std::cout);
          std::cout << std::endl;
        }
    }

  if (print_details && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      table.write_text(std::cout);
      std::cout << std::endl;
    }
}

/**
 * mpirun -np 1 ./benchmark_01 annulus 8 8 1
 * mpirun -np 1 ./benchmark_01 quadrant 4 8 1
 * mpirun -np 1 ./benchmark_01 quadrant_flexible 4 8 1
 *
 *
 * mpirun -np 40 ./benchmark_01 annulus 5 8 4
 * mpirun -np 40 ./benchmark_01 quadrant 2 8 4
 * mpirun -np 40 ./benchmark_01 quadrant_flexible 2 8 4
 */
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim        = 3;
  const unsigned int max_degree = 4;

  const std::string geometry_type =
    argc > 1 ? std::string(argv[1]) : "quadrant";
  const unsigned int min_n_refinements = argc > 2 ? atoi(argv[2]) : 6;
  const unsigned int max_n_refinements = argc > 3 ? atoi(argv[3]) : 6;
  const unsigned int degree            = argc > 4 ? atoi(argv[4]) : 1;
  const bool         print_details     = true;

  AssertThrow(degree <= max_degree, ExcNotImplemented());

  run<dim, max_degree>(
    geometry_type, min_n_refinements, max_n_refinements, degree, print_details);
}
