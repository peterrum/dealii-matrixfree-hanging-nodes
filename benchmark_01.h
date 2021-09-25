#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

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

    for (unsigned int i = 1; i < n_refinements; i++)
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
  create_quadrant_flexible(Triangulation<dim> &tria,
                           const unsigned int  n_ref_global,
                           const unsigned int  n_ref_local = 1)
  {
    GridGenerator::hyper_cube(tria, -1.0, +1.0);
    tria.refine_global(n_ref_global);

    for (unsigned int i = 0; i < n_ref_local; ++i)
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
} // namespace dealii::GridGenerator

template <class T, std::size_t N>
std::array<std::pair<unsigned int, T>, N>
sort_and_count(const std::array<T, N> &input)
{
  std::array<std::pair<unsigned int, T>, N> result;

  for (unsigned int i = 0; i < N; ++i)
    {
      result[i].first  = i;
      result[i].second = input[i];
    }

  std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
    return a.second > b.second;
  });

  return result;
}


template <int dim, int degree>
class Test
{
public:
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType0         = Vector<Number>;
  using VectorType1         = AlignedVector<VectorizedArrayType>;
  using FEEval =
    FEEvaluation<dim, degree, degree + 1, 1, Number, VectorizedArrayType>;

  struct Info
  {
    unsigned int n_levels         = 0;
    unsigned int n_cells          = 0;
    unsigned int n_cells_n        = 0;
    unsigned int n_cells_hn       = 0;
    unsigned int n_macro_cells    = 0;
    unsigned int n_macro_cells_n  = 0;
    unsigned int n_macro_cells_hn = 0;
    unsigned int n_dofs_dg        = 0;
    unsigned int n_dofs_cg        = 0;
  };

private:
  const unsigned n_repetitions = 10;
  bool           do_cg;
  bool           do_apply_constraints;
  bool           do_apply_quadrature_kernel;

  Triangulation<dim>                           tria;
  DoFHandler<dim>                              dof_handler;
  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

public:
  Test(const std::string geometry_type, const unsigned int n_refinements)
    : do_cg(false)
    , do_apply_constraints(false)
    , do_apply_quadrature_kernel(false)
    , dof_handler(tria)
  {
    if (geometry_type == "annulus")
      GridGenerator::create_annulus(tria, n_refinements);
    else if (geometry_type == "quadrant")
      GridGenerator::create_quadrant(tria, n_refinements);
    else if (geometry_type == "quadrant_flexible")
      GridGenerator::create_quadrant_flexible(tria, n_refinements);
    else
      AssertThrow(false, ExcMessage("Unknown geometry type!"));

    const MappingQ1<dim> mapping;
    const FE_Q<dim>      fe(degree);
    const QGauss<dim>    quadrature(degree + 1);

    dof_handler.distribute_dofs(fe);

    AffineConstraints<Number> constraints;
    matrix_free.reinit(mapping, dof_handler, constraints, quadrature);
  }

  Info
  get_info(const bool do_print)
  {
    Info info;

    info.n_cells       = tria.n_active_cells();
    info.n_macro_cells = matrix_free.n_cell_batches();

    constexpr unsigned int n_lanes = VectorizedArrayType::size();

    std::array<unsigned int, VectorizedArrayType::size()> n_lanes_with_hn;
    n_lanes_with_hn.fill(0);

    std::array<unsigned int, 512> hn_types;
    hn_types.fill(0);

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        std::array<internal::MatrixFreeFunctions::ConstraintKinds, n_lanes>
          constraint_mask;

        const unsigned int n_vectorization_actual =
          matrix_free.n_active_entries_per_cell_batch(cell);

        bool hn_available = false;

        for (unsigned int v = 0; v < n_vectorization_actual; ++v)
          {
            const auto mask =
              matrix_free.get_dof_info()
                .hanging_node_constraint_masks[cell * n_lanes + v];
            constraint_mask[v] = mask;

            hn_available |=
              (mask !=
               internal::MatrixFreeFunctions::ConstraintKinds::unconstrained);
          }

        if (hn_available)
          {
            for (unsigned int v = n_vectorization_actual; v < n_lanes; ++v)
              constraint_mask[v] =
                internal::MatrixFreeFunctions::ConstraintKinds::unconstrained;
          }

        if (hn_available)
          {
            info.n_macro_cells_hn++;

            unsigned int n_lanes_with_hn_counter = 0;

            for (unsigned int v = 0; v < n_vectorization_actual; ++v)
              if (constraint_mask[v] !=
                  internal::MatrixFreeFunctions::ConstraintKinds::unconstrained)
                {
                  n_lanes_with_hn_counter++;
                  hn_types[static_cast<std::uint16_t>(constraint_mask[v])]++;
                }

            info.n_cells_hn += n_lanes_with_hn_counter;
            info.n_cells_n +=
              (n_vectorization_actual - n_lanes_with_hn_counter);

            n_lanes_with_hn[n_lanes_with_hn_counter]++;
          }
        else
          {
            info.n_macro_cells_n++;
            info.n_cells_n += n_vectorization_actual;
          }
      }

    AssertThrow((info.n_cells_n + info.n_cells_hn == info.n_cells),
                ExcMessage("Number of cells do not match."));
    AssertThrow((info.n_macro_cells_n + info.n_macro_cells_hn ==
                 info.n_macro_cells),
                ExcMessage("Number of macro cells do not match."));

    if (do_print && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "Number of lanes with hn constraints:" << std::endl;
        for (const auto i : sort_and_count(n_lanes_with_hn))
          std::cout << "  " << i.first << " : " << i.second << std::endl;
        std::cout << std::endl;

        const auto to_string = [](std::uint16_t in) {
          std::string result;

          for (unsigned int i = 0; i < 9; ++i)
            {
              if ((in >> (8 - i)) & 1)
                result += "1";
              else
                result += "0";

              if ((i + 1) % 3 == 0)
                result += " ";
            }

          return result;
        };

        std::cout << "Number of occurrences of ConstraintKinds:" << std::endl;
        for (const auto i : sort_and_count(hn_types))
          if (i.second > 0)
            std::cout << "  " << to_string(i.first) << " : " << i.second
                      << std::endl;
        std::cout << std::endl;
      }

    info.n_levels = tria.n_global_levels();

    return info;
  }

  double
  run(const bool do_cg,
      const bool do_apply_constraints,
      const bool do_apply_quadrature_kernel)
  {
    this->do_cg                      = do_cg;
    this->do_apply_constraints       = do_apply_constraints;
    this->do_apply_quadrature_kernel = do_apply_quadrature_kernel;

    VectorType0 src0, dst0;
    VectorType1 src1, dst1;

    if (this->do_cg)
      {
        matrix_free.initialize_dof_vector(src0);
        matrix_free.initialize_dof_vector(dst0);

        for (auto &i : src0)
          i = 1.0;
      }
    else
      {
        unsigned int size =
          matrix_free.get_dof_handler().get_fe().n_dofs_per_cell() *
          matrix_free.n_cell_batches();
        src1.resize(size);
        dst1.resize(size);

        for (auto &i : src1)
          for (auto &j : i)
            j = 1.0;
      }

    double min_time = 1e10;

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("kernel");
#endif

    for (unsigned int i = 0; i < n_repetitions; ++i)
      {
        MPI_Barrier(MPI_COMM_WORLD);

        std::chrono::time_point<std::chrono::system_clock> temp =
          std::chrono::system_clock::now();


        if (this->do_cg)
          vmult(dst0, src0);
        else
          vmult(dst1, src1);


        min_time =
          std::min<double>(min_time,
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::system_clock::now() - temp)
                               .count() /
                             1e9);
      }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("kernel");
#endif

    min_time = Utilities::MPI::min(min_time, MPI_COMM_WORLD);

    return min_time;
  }

private:
  template <typename VectorType>
  void
  vmult(VectorType &dst, const VectorType &src)
  {
    FEEval phi(matrix_free);

    // loop over all cells
    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);

        // gather values
        if (do_apply_constraints)
          gather(phi, src); // ... and apply constraints
        else
          gather_plain(phi, src);

        // perform operation on quadrature points
        if (do_apply_quadrature_kernel)
          {
            phi.evaluate(EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);

            phi.integrate(EvaluationFlags::gradients);
          }

        // scatter values
        if (do_apply_constraints)
          scatter(phi, dst); // ... and apply constraints
        else
          scatter_plain(phi, dst);
      }
  }

  void
  gather(FEEval &phi, const VectorType0 &src)
  {
    phi.read_dof_values(src);
  }

  void
  gather(FEEval &phi, const VectorType1 &src)
  {
    this->gather_plain(phi, src);

    phi.template apply_hanging_node_constraints<false>();
  }

  void
  gather_plain(FEEval &phi, const VectorType0 &src)
  {
    phi.read_dof_values_plain(src);
  }

  void
  gather_plain(FEEval &phi, const VectorType1 &src)
  {
    for (unsigned int
           i = 0,
           k = FEEval::static_dofs_per_component * phi.get_current_cell_index();
         i < FEEval::static_dofs_per_component;
         ++i, ++k)
      phi.begin_dof_values()[i] = src[k];
  }

  void
  scatter(FEEval &phi, VectorType0 &dst)
  {
    phi.distribute_local_to_global(dst);
  }

  void
  scatter(FEEval &phi, VectorType1 &dst)
  {
    phi.template apply_hanging_node_constraints<true>();

    scatter_plain(phi, dst);
  }

  void
  scatter_plain(FEEval &phi, VectorType0 &dst)
  {
    phi.distribute_local_to_global_plain(dst);
  }

  void
  scatter_plain(FEEval &phi, VectorType1 &dst)
  {
    for (unsigned int
           i = 0,
           k = FEEval::static_dofs_per_component * phi.get_current_cell_index();
         i < FEEval::static_dofs_per_component;
         ++i, ++k)
      dst[k] += phi.begin_dof_values()[i];
  }
};
