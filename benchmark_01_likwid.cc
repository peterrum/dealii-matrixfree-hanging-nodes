#include "benchmark_01.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  const unsigned int dim                = 3;
  const unsigned int degree_precompiled = 1;

  const std::string geometry_type =
    argc > 1 ? std::string(argv[1]) : "quadrant";
  const unsigned int n_refinements              = argc > 2 ? atoi(argv[2]) : 6;
  const unsigned int degree                     = argc > 3 ? atoi(argv[3]) : 1;
  const bool         do_cg                      = argc > 4 ? atoi(argv[4]) : 0;
  const bool         do_apply_constraints       = argc > 4 ? atoi(argv[4]) : 1;
  const bool         do_apply_quadrature_kernel = argc > 5 ? atoi(argv[5]) : 0;

  AssertThrow(degree_precompiled == degree, ExcNotImplemented());

  Test<dim, degree_precompiled> test(geometry_type, n_refinements);
  test.run(do_cg, do_apply_constraints, do_apply_quadrature_kernel);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
