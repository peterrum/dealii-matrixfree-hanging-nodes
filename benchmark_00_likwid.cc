#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;

template <unsigned int dim, int fe_degree_precomiled>
void
run(const unsigned int degree)
{
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;

  const FE_Q<dim> fe(degree);
  const QGauss<1> quadrature(degree + 1);

  FEEvaluation<dim,
               fe_degree_precomiled,
               fe_degree_precomiled + 1,
               1,
               Number,
               VectorizedArrayType>
    fe_eval(fe, quadrature, update_default);

  std::array<internal::MatrixFreeFunctions::compressed_constraint_kind,
             VectorizedArrayType::size()>
    mask;

  std::fill(
    mask.begin(),
    mask.end(),
    internal::MatrixFreeFunctions::unconstrained_compressed_constraint_kind);

  const std::uint16_t quadrant        = 1;
  const std::uint16_t face_constraint = 7;
  const std::uint16_t edge_constraint = 0;

  mask[0] = internal::MatrixFreeFunctions::compress(
    static_cast<internal::MatrixFreeFunctions::ConstraintKinds>(
      quadrant + (face_constraint << 3) + (edge_constraint << 6)),
    dim);

  AlignedVector<VectorizedArrayType> data(fe.n_dofs_per_cell());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("kernel");
#endif

  for (unsigned int i = 0; i < 100; ++i)
    internal::
      FEEvaluationHangingNodesFactory<dim, Number, VectorizedArrayType>::apply(
        1, degree, fe_eval.get_shape_info(), false, mask, data.begin());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("kernel");
#endif
}

/**
 * likwid-mpirun -np 1 -f -g MEM -m -O ./benchmark_00_likwid annulus 4
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

  const unsigned int degree = argc > 1 ? atoi(argv[1]) : 1;

  std::cout << degree << " " << argv[1] << std::endl;

  run<dim, fe_degree_precomiled>(degree);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
