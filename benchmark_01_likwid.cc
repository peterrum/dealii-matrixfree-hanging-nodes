#include "benchmark_01.h"

template <unsigned int dim, int fe_degree_precomiled>
void
run(const std::string  geometry_type,
    const unsigned int n_refinements,
    const unsigned int degree,
    const bool         do_cg,
    const bool         do_apply_constraints,
    const bool         do_apply_quadrature_kernel,
    const bool         setup_only_fast_algorithm)
{
  Test<dim, fe_degree_precomiled> test(degree,
                                       geometry_type,
                                       n_refinements,
                                       setup_only_fast_algorithm);
  test.run(do_cg, do_apply_constraints, do_apply_quadrature_kernel);
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
  const bool         do_apply_constraints       = argc > 4 ? atoi(argv[4]) : 1;
  const bool         do_apply_quadrature_kernel = argc > 5 ? atoi(argv[5]) : 0;
  const bool         setup_only_fast_algorithm  = true;

  run<dim, fe_degree_precomiled>(geometry_type,
                                 n_refinements,
                                 degree,
                                 do_cg,
                                 do_apply_constraints,
                                 do_apply_quadrature_kernel,
                                 setup_only_fast_algorithm);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
