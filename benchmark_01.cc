#include "benchmark_01.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim                = 3;
  const unsigned int degree_precompiled = 1;

  const std::string geometry_type =
    argc > 1 ? std::string(argv[1]) : "quadrant";
  const unsigned int min_n_refinements = argc > 2 ? atoi(argv[2]) : 6;
  const unsigned int max_n_refinements = argc > 3 ? atoi(argv[3]) : 6;
  const unsigned int degree            = argc > 4 ? atoi(argv[4]) : 1;

  AssertThrow(degree_precompiled == degree, ExcNotImplemented());

  ConvergenceTable table;

  for (unsigned int n_refinements = min_n_refinements;
       n_refinements <= max_n_refinements;
       ++n_refinements)
    {
      Test<dim, degree_precompiled> test(geometry_type, n_refinements);

      const auto info = test.get_info();

      table.add_value("n_levels", info.n_levels);
      table.add_value("degree", degree);
      table.add_value("n_cells", info.n_cells);
      table.add_value("n_cells_n", info.n_cells_n);
      table.add_value("n_cells_hn", info.n_cells_hn);

      const auto compute_cost = [&](const auto t_n, const auto t_hn) {
        return (t_hn / (t_n / (info.n_cells_n + info.n_cells_hn)) -
                info.n_cells_n) /
               info.n_cells_hn;
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

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          table.write_text(std::cout);
          std::cout << std::endl;
        }
    }
}