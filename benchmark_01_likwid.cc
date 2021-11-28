#include "benchmark_01.h"

template <unsigned int dim, int fe_degree_precomiled>
void
run(const std::string  geometry_type,
    const unsigned int n_refinements,
    const unsigned int degree,
    const bool         do_cg,
    const bool         do_apply_constraints,
    const bool         do_apply_quadrature_kernel,
    const bool         use_fast_hanging_node_algorithm,
    const bool         test_high_order_mapping,
    const bool         setup_only_fast_algorithm)
{
  Test<dim, fe_degree_precomiled> test(degree,
                                       geometry_type,
                                       n_refinements,
                                       setup_only_fast_algorithm,
                                       test_high_order_mapping);

  ConvergenceTable table;

  const auto info = test.get_info(false);
  table.add_value("n_levels", info.n_levels);
  table.add_value("degree", degree);
  table.add_value("n_dofs", info.n_dofs);
  table.add_value("n_cells", info.n_cells);
  table.add_value("n_cells_n", info.n_cells_n);
  table.add_value("n_cells_hn", info.n_cells_hn);
  table.add_value("n_macro_cells", info.n_macro_cells);
  table.add_value("n_macro_cells_n", info.n_macro_cells_n);
  table.add_value("n_macro_cells_hn", info.n_macro_cells_hn);

  const auto t0 = test.run(do_cg,
                           do_apply_constraints,
                           do_apply_quadrature_kernel,
                           use_fast_hanging_node_algorithm);

  table.add_value("t0", t0);
  table.set_scientific("t0", true);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      table.write_text(std::cout);
      std::cout << std::endl;
    }
}

/**
 * likwid-mpirun -np 1 -f -g MEM -m -O ./benchmark_01_likwid annulus 7 1 0 0 0
 */
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  const unsigned int dim                  = 3;
  const int          fe_degree_precomiled = -1;

  const std::string geometry_type =
    argc > 1 ? std::string(argv[1]) : "quadrant";
  const unsigned int n_refinements              = argc > 2 ? atoi(argv[2]) : 6;
  const unsigned int degree                     = argc > 3 ? atoi(argv[3]) : 1;
  const bool         do_cg                      = argc > 4 ? atoi(argv[4]) : 0;
  const bool         do_apply_constraints       = argc > 5 ? atoi(argv[5]) : 1;
  const bool         do_apply_quadrature_kernel = argc > 6 ? atoi(argv[6]) : 0;
  const bool use_fast_hanging_node_algorithm    = argc > 7 ? atoi(argv[7]) : 1;
  const bool test_high_order_mapping            = argc > 8 ? atoi(argv[8]) : 0;
  const bool setup_only_fast_algorithm          = false;

  run<dim, fe_degree_precomiled>(geometry_type,
                                 n_refinements,
                                 degree,
                                 do_cg,
                                 do_apply_constraints,
                                 do_apply_quadrature_kernel,
                                 use_fast_hanging_node_algorithm,
                                 test_high_order_mapping,
                                 setup_only_fast_algorithm);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
