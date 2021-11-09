#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

#include "benchmark.h"

using namespace dealii;

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
    if (a.second > b.second)
      return true;

    if (a.second < b.second)
      return false;

    return a.first < b.first;
  });

  return result;
}



template <int dim,
          int fe_degree,
          int n_q_points_1d,
          int n_components_,
          typename Number,
          typename VectorizedArrayType>
class FEEvaluationOwn : public FEEvaluation<dim,
                                            fe_degree,
                                            n_q_points_1d,
                                            n_components_,
                                            Number,
                                            VectorizedArrayType>
{
public:
  FEEvaluationOwn(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const unsigned int                                  dof_no  = 0,
    const unsigned int                                  quad_no = 0,
    const unsigned int first_selected_component                 = 0,
    const unsigned int active_fe_index   = numbers::invalid_unsigned_int,
    const unsigned int active_quad_index = numbers::invalid_unsigned_int)
    : FEEvaluation<dim,
                   fe_degree,
                   n_q_points_1d,
                   n_components_,
                   Number,
                   VectorizedArrayType>(matrix_free,
                                        dof_no,
                                        quad_no,
                                        first_selected_component,
                                        active_fe_index,
                                        active_quad_index)
  {}

  template <typename VectorType>
  inline void
  distribute_local_to_global_plain(
    VectorType &                                    dst,
    const unsigned int                              first_index = 0,
    const std::bitset<VectorizedArrayType::size()> &mask =
      std::bitset<VectorizedArrayType::size()>().flip()) const
  {
#ifdef DEBUG
    Assert(dof_values_initialized == true,
           internal::ExcAccessToUninitializedField());
#endif

    const auto dst_data = internal::get_vector_data<n_components_>(
      dst,
      first_index,
      this->dof_access_index ==
        internal::MatrixFreeFunctions::DoFInfo::dof_access_cell,
      this->active_fe_index,
      this->dof_info);

    internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType>
      distributor;
    this->read_write_operation(distributor,
                               dst_data.first,
                               dst_data.second,
                               mask);
  }

  template <bool transpose>
  void
  apply_hanging_node_constraints()
  {
    FEEvaluationBase<dim, n_components_, Number, false, VectorizedArrayType>::
      template apply_hanging_node_constraints<transpose>();
  }
};



template <int dim, int fe_degree_precomiled>
class Test
{
public:
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType0         = Vector<Number>;
  using VectorType1         = AlignedVector<VectorizedArrayType>;
  using FEEval              = FEEvaluationOwn<dim,
                                 fe_degree_precomiled,
                                 fe_degree_precomiled + 1,
                                 1,
                                 Number,
                                 VectorizedArrayType>;

  struct Info
  {
    unsigned int n_levels         = 0;
    unsigned int n_cells          = 0;
    unsigned int n_dofs           = 0;
    unsigned int n_cells_n        = 0;
    unsigned int n_cells_hn       = 0;
    unsigned int n_macro_cells    = 0;
    unsigned int n_macro_cells_n  = 0;
    unsigned int n_macro_cells_hn = 0;
    unsigned int n_dofs_dg        = 0;
    unsigned int n_dofs_cg        = 0;
  };

private:
  const unsigned int n_repetitions = 10;
  const unsigned int degree;
  const bool         setup_only_fast_algorithm;
  const bool         test_high_order_mapping;
  bool               do_cg;
  bool               do_apply_constraints;
  bool               do_apply_quadrature_kernel;
  bool               use_fast_hanging_node_algorithm;

  Triangulation<dim>                           tria;
  DoFHandler<dim>                              dof_handler;
  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  MatrixFree<dim, Number, VectorizedArrayType> matrix_free_slow;

public:
  Test(const unsigned int degree,
       const std::string  geometry_type,
       const unsigned int n_refinements,
       const bool         setup_only_fast_algorithm,
       const bool         test_high_order_mapping = false)
    : degree(degree)
    , setup_only_fast_algorithm(setup_only_fast_algorithm)
    , test_high_order_mapping(test_high_order_mapping)
    , do_cg(false)
    , do_apply_constraints(false)
    , do_apply_quadrature_kernel(false)
    , use_fast_hanging_node_algorithm(true)
    , dof_handler(tria)
  {
    AssertThrow(fe_degree_precomiled == -1 ||
                  static_cast<unsigned int>(fe_degree_precomiled) == degree,
                ExcMessage("Degrees do not match!"));

    if (geometry_type == "annulus")
      GridGenerator::create_annulus(tria, n_refinements);
    else if (geometry_type == "quadrant")
      GridGenerator::create_quadrant(tria, n_refinements);
    else if (geometry_type == "step")
      GridGenerator::create_step(tria, n_refinements);
    else if (geometry_type == "quadrant_flexible")
      GridGenerator::create_quadrant_flexible(tria, n_refinements);
    else
      AssertThrow(false, ExcMessage("Unknown geometry type!"));

    std::unique_ptr<Mapping<dim>> mapping;

    if (test_high_order_mapping == false)
      {
        mapping = std::make_unique<MappingQ1<dim>>();
      }
    else
      {
        auto mapping_temp = new MappingQCache<dim>(2);
        mapping_temp->initialize(
          MappingQ1<dim>(),
          tria,
          [](const auto &, const auto &point) {
            Point<dim> result;

            for (int d = 0; d < dim; ++d)
              result[d] = std::sin(point[d]) * 0.000001;

            return result;
          },
          true);

        mapping = std::unique_ptr<Mapping<dim>>(mapping_temp);
      }

    const FE_Q<dim>   fe(degree);
    const QGauss<dim> quadrature(degree + 1);

    dof_handler.distribute_dofs(fe);

    AffineConstraints<Number> constraints;

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_gradients;

    matrix_free.reinit(
      *mapping, dof_handler, constraints, quadrature, additional_data);

    if (true /*TODO*/)
      {
        additional_data.cell_vectorization_category.assign(
          tria.n_active_cells(), 0);
        additional_data.cell_vectorization_categories_strict = true;

        for (unsigned int i = 0; i < matrix_free.n_cell_batches(); ++i)
          for (unsigned int v = 0;
               v < matrix_free.n_active_entries_per_cell_batch(i);
               ++v)
            additional_data.cell_vectorization_category
              [matrix_free.get_cell_iterator(i, v)->active_cell_index()] =
              static_cast<unsigned int>(
                matrix_free.get_dof_info().hanging_node_constraint_masks
                  [i * VectorizedArrayType::size() + v]);

        auto temp = additional_data.cell_vectorization_category;

        std::sort(temp.begin(), temp.end());
        temp.erase(std::unique(temp.begin(), temp.end()), temp.end());

        if (false)
          std::cout << temp.size() << std::endl;

        matrix_free.reinit(
          *mapping, dof_handler, constraints, quadrature, additional_data);
      }

    if (setup_only_fast_algorithm == false)
      {
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        additional_data.use_fast_hanging_node_algorithm = false;
        matrix_free_slow.reinit(
          *mapping, dof_handler, constraints, quadrature, additional_data);
      }
  }

  Info
  get_info(const bool do_print)
  {
    Info info;

    info.n_cells       = tria.n_active_cells();
    info.n_dofs        = dof_handler.n_dofs();
    info.n_macro_cells = matrix_free.n_cell_batches();

    constexpr unsigned int n_lanes = VectorizedArrayType::size();

    std::array<unsigned int, VectorizedArrayType::size()> n_lanes_with_hn;
    n_lanes_with_hn.fill(0);

    std::array<unsigned int, 512> hn_types;
    hn_types.fill(0);

    std::array<unsigned int, VectorizedArrayType::size()> n_lanes_with_hn_same;
    n_lanes_with_hn_same.fill(0);

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
              matrix_free.get_dof_info().hanging_node_constraint_masks.size() ==
                  0 ?
                internal::MatrixFreeFunctions::ConstraintKinds::unconstrained :
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

            std::map<std::uint16_t, unsigned int> n_lanes_with_hn_same_local;

            for (unsigned int v = 0; v < n_vectorization_actual; ++v)
              if (constraint_mask[v] !=
                  internal::MatrixFreeFunctions::ConstraintKinds::unconstrained)
                n_lanes_with_hn_same_local[static_cast<std::uint16_t>(
                  constraint_mask[v])] = 0;

            for (unsigned int v = 0; v < n_vectorization_actual; ++v)
              if (constraint_mask[v] !=
                  internal::MatrixFreeFunctions::ConstraintKinds::unconstrained)
                {
                  n_lanes_with_hn_counter++;
                  hn_types[static_cast<std::uint16_t>(constraint_mask[v])]++;
                  n_lanes_with_hn_same_local[static_cast<std::uint16_t>(
                    constraint_mask[v])]++;
                }

            if (true /*todo*/)
              {
                for (const auto i : n_lanes_with_hn_same_local)
                  if (i.second != 0)
                    n_lanes_with_hn_same[i.second - 1]++;
              }
            else
              {
                n_lanes_with_hn_same[std::max_element(
                                       n_lanes_with_hn_same_local.begin(),
                                       n_lanes_with_hn_same_local.end(),
                                       [](const auto &a, const auto &b) {
                                         return a.second < b.second;
                                       })
                                       ->second -
                                     1]++;
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
          std::cout << "  " << (i.first + 1) << " : " << i.second << std::endl;
        std::cout << std::endl;

        std::cout << "Number of lanes with max same hn constraints:"
                  << std::endl;
        for (const auto i : sort_and_count(n_lanes_with_hn_same))
          std::cout << "  " << (i.first + 1) << " : " << i.second << std::endl;
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

        std::map<std::pair<unsigned int, unsigned int>, unsigned int>
          hn_types_reduced;

        const auto count_edge_bits = [](const auto m) -> unsigned int {
          return ((m >> 6) & 1) + ((m >> 7) & 1) + ((m >> 8) & 1);
        };

        const auto count_face_bits = [](const auto m) -> unsigned int {
          return ((m >> 3) & 1) + ((m >> 4) & 1) + ((m >> 5) & 1);
        };

        for (const auto i : sort_and_count(hn_types))
          if (i.second > 0)
            hn_types_reduced[{count_edge_bits(i.first),
                              count_face_bits(i.first)}] += i.second;

        std::cout << "Number of occurrences of ConstraintKinds (reduced):"
                  << std::endl;
        std::cout << "  1e0f  : " << hn_types_reduced[{1, 0}] << std::endl;
        std::cout << "  2e0f  : " << hn_types_reduced[{2, 0}] << std::endl;
        std::cout << "  3e0f  : " << hn_types_reduced[{3, 0}] << std::endl;
        std::cout << "  0e1f  : " << hn_types_reduced[{0, 1}] << std::endl;
        std::cout << "  1e1f  : " << hn_types_reduced[{1, 1}] << std::endl;
        std::cout << "  0e2f  : " << hn_types_reduced[{0, 2}] << std::endl;
        std::cout << "  0e3f  : " << hn_types_reduced[{0, 3}] << std::endl;
        std::cout << std::endl;
      }

    info.n_levels = tria.n_global_levels();

    return info;
  }

  double
  run(const bool do_cg,
      const bool do_apply_constraints,
      const bool do_apply_quadrature_kernel,
      const bool use_fast_hanging_node_algorithm = true)
  {
    this->do_cg                      = do_cg;
    this->do_apply_constraints       = do_apply_constraints;
    this->do_apply_quadrature_kernel = do_apply_quadrature_kernel;

    AssertThrow(use_fast_hanging_node_algorithm || !setup_only_fast_algorithm,
                ExcMessage("Only fast algorithm has been set up!"));

    this->use_fast_hanging_node_algorithm = use_fast_hanging_node_algorithm;

    VectorType0 src0, dst0;
    VectorType1 src1, dst1;

    const auto &matrix_free = use_fast_hanging_node_algorithm ?
                                this->matrix_free :
                                this->matrix_free_slow;

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
    const auto &matrix_free = use_fast_hanging_node_algorithm ?
                                this->matrix_free :
                                this->matrix_free_slow;

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
    for (unsigned int i = 0,
                      k = phi.dofs_per_cell * phi.get_current_cell_index();
         i < phi.dofs_per_cell;
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
    for (unsigned int i = 0,
                      k = phi.dofs_per_cell * phi.get_current_cell_index();
         i < phi.dofs_per_cell;
         ++i, ++k)
      dst[k] += phi.begin_dof_values()[i];
  }
};
