#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include "constraint_helper.h"

using namespace dealii;

namespace dealii
{
  namespace GridGenerator
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
              if (0.335 <= cell->center().norm() &&
                  cell->center().norm() <= 0.39)
                cell->set_refine_flag();
          tria.execute_coarsening_and_refinement();
        }
    }
  } // namespace GridGenerator
} // namespace dealii



template <int dim,
          int fe_degree,
          int n_q_points_1d,
          int n_components_,
          typename Number,
          typename VectorizedArrayType = VectorizedArray<Number>>
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
  read_dof_values_plain(const VectorType & src,
                        const unsigned int first_index = 0,
                        const std::bitset<VectorizedArrayType::size()> &mask =
                          std::bitset<VectorizedArrayType::size()>().flip())
  {
    const auto src_data = internal::get_vector_data<n_components_>(
      src,
      first_index,
      this->dof_access_index ==
        internal::MatrixFreeFunctions::DoFInfo::dof_access_cell,
      this->active_fe_index,
      this->dof_info);

    internal::VectorReader<Number, VectorizedArrayType> reader;
    this->read_write_operation(reader, src_data.first, src_data.second, mask);

#ifdef DEBUG
    dof_values_initialized = true;
#endif
  }

  template <typename VectorType>
  inline void
  distribute_local_to_global_plain(
    VectorType &                                    dst,
    const unsigned int                              first_index = 0,
    const std::bitset<VectorizedArrayType::size()> &mask =
      std::bitset<VectorizedArrayType::size()>().flip()) const
  {
#ifdef DEBUG
    Assert(this->dof_values_initialized == true,
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
                  const Quadrature<1> &            quadrature,
                  const bool                       apply_constraints)
    : apply_constraints(apply_constraints)
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
    FEEvaluationOwn<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);

        if (apply_constraints)
          phi.read_dof_values(src);
        else
          phi.read_dof_values_plain(src);

        phi.evaluate(false, true);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(false, true);

        if (apply_constraints)
          phi.distribute_local_to_global(dst);
        else
          phi.distribute_local_to_global_plain(dst);
      }
  }

  const bool apply_constraints;

  MatrixFree<dim, Number> matrix_free;
};



#ifdef DEAL_II_COMPILER_CUDA_AWARE
template <int dim, int fe_degree, typename Number>
class LaplaceOperatorQuad
{
public:
  __device__ void
  operator()(
    CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>
      *fe_eval) const
  {
    fe_eval->submit_gradient(fe_eval->get_gradient());
  }
};

template <int dim, int fe_degree, typename Number>
class LaplaceOperatorLocal
{
public:
  __device__ void
  operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, Number>::Data *gpu_data,
    CUDAWrappers::SharedData<dim, Number> *                     shared_data,
    const Number *                                              src,
    Number *                                                    dst) const
  {
    CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>
      fe_eval(cell, gpu_data, shared_data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(false, true);
    fe_eval.apply_for_each_quad_point(
      LaplaceOperatorQuad<dim, fe_degree, Number>());
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }
  static const unsigned int n_dofs_1d    = fe_degree + 1;
  static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
  static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);
};

template <int dim, int fe_degree, typename Number>
class LaplaceOperator<dim, fe_degree, Number, MemorySpace::CUDA>
{
public:
  using VectorType =
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;

  LaplaceOperator(const Mapping<dim> &             mapping,
                  const DoFHandler<dim> &          dof_handler,
                  const AffineConstraints<Number> &constraints,
                  const Quadrature<1> &            quadrature,
                  const bool                       apply_constraints)
  {
    AssertThrow(apply_constraints, ExcNotImplemented());

    typename CUDAWrappers::MatrixFree<dim, Number>::AdditionalData
      additional_data;
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
    LaplaceOperatorLocal<dim, fe_degree, Number> local_operator;
    matrix_free.cell_loop(local_operator, src, dst);
  }

private:
  CUDAWrappers::MatrixFree<dim, Number> matrix_free;
};
#endif



template <int dim, typename T>
class AnalyticalFunction : public Function<dim, T>
{
public:
  virtual T
  value(const Point<dim, T> &p, const unsigned int component = 0) const
  {
    (void)component;

    double temp = 0.0;

    for (unsigned int d = 0; d < dim; ++d)
      temp += std::sin(p[d]);

    return temp;
  }
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

  for (unsigned int n_refinements = 4; n_refinements <= 12; ++n_refinements)
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
      table.add_value("geometry_type", geometry_type);

      table.add_value("n_cells", tria.n_global_active_cells());

      types::global_cell_index n_cells_w_hn  = 0;
      types::global_cell_index n_cells_wo_hn = 0;

      Helper<dim> helper(tria);

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (helper.is_constrained(cell))
              n_cells_w_hn++;
            else
              n_cells_wo_hn++;
          }

      n_cells_w_hn  = Utilities::MPI::sum(n_cells_w_hn, MPI_COMM_WORLD);
      n_cells_wo_hn = Utilities::MPI::sum(n_cells_wo_hn, MPI_COMM_WORLD);

      AssertThrow(tria.n_global_active_cells() == n_cells_w_hn + n_cells_wo_hn,
                  ExcInternalError());

      table.add_value("n_cells_hn", n_cells_w_hn);

      const MappingQ1<dim> mapping;
      const FE_Q<dim>      fe(degree);
      const QGauss<1>      quadrature(degree + 1);

      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      table.add_value("n_dofs", dof_handler.n_dofs());

      AffineConstraints<Number> constraints;

      const auto run =
        [&](const bool apply_constraints) -> std::array<double, 3> {
        LaplaceOperator<dim, degree, Number, MemorySpace> laplace_operator(
          mapping, dof_handler, constraints, quadrature, apply_constraints);

        VectorType src, dst;

        laplace_operator.initialize_dof_vector(src);
        laplace_operator.initialize_dof_vector(dst);

        {
          LinearAlgebra::distributed::Vector<Number> src_host(
            src.get_partitioner());

          VectorTools::interpolate(dof_handler,
                                   AnalyticalFunction<dim, Number>(),
                                   src_host);

          LinearAlgebra::ReadWriteVector<Number> rw_vector(
            src.get_partitioner()->locally_owned_range());
          rw_vector.import(src_host, VectorOperation::insert);
          src.import(rw_vector, VectorOperation::insert);

          dst = 0.0;
        }

        double min_time = 1e10;
        double max_time = 0;
        double avg_time = 0;

        for (unsigned int i = 0; i < n_repetitions; ++i)
          {
            MPI_Barrier(MPI_COMM_WORLD);

            std::chrono::time_point<std::chrono::system_clock> temp =
              std::chrono::system_clock::now();

            laplace_operator.vmult(dst, src);

#ifdef DEAL_II_COMPILER_CUDA_AWARE
            cudaDeviceSynchronize();
#endif

            MPI_Barrier(MPI_COMM_WORLD);

            const double dt =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now() - temp)
                .count() /
              1e9;

            min_time = std::min(min_time, dt);
            max_time = std::max(max_time, dt);
            avg_time += dt;
          }

        min_time = Utilities::MPI::min(min_time, MPI_COMM_WORLD);
        max_time = Utilities::MPI::max(max_time, MPI_COMM_WORLD);
        avg_time = Utilities::MPI::sum(avg_time, MPI_COMM_WORLD) /
                   Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) /
                   n_repetitions;

        return {{min_time, max_time, avg_time}};
      };

      if (true)
        {
          const auto time = run(true);

          table.add_value("time_min", time[0]);
          table.set_scientific("time_min", true);
          table.add_value("time_max", time[1]);
          table.set_scientific("time_max", true);
          table.add_value("time_avg", time[2]);
          table.set_scientific("time_avg", true);
        }

      if (std::is_same<MemorySpace, dealii::MemorySpace::Host>::value)
        {
          const auto time = run(false);

          table.add_value("time_no_min", time[0]);
          table.set_scientific("time_no_min", true);
          table.add_value("time_no_max", time[1]);
          table.set_scientific("time_no_max", true);
          table.add_value("time_no_avg", time[2]);
          table.set_scientific("time_no_avg", true);
        }

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
 * mpirun -np 40 ./benchmark_02 host quadrant 4
 */
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim = 3;

  const std::string memory_type = argc > 1 ? std::string(argv[1]) : "host";

  const std::string geometry_type =
    argc > 2 ? std::string(argv[2]) : "quadrant";

  const unsigned int fe_degree = argc > 3 ? std::atoi(argv[3]) : 4;

  if (memory_type == "host")
    {
      switch (fe_degree)
        {
          case 1:
            run<dim, 1, MemorySpace::Host>(geometry_type);
            break;
          case 2:
            run<dim, 2, MemorySpace::Host>(geometry_type);
            break;
          case 3:
            run<dim, 3, MemorySpace::Host>(geometry_type);
            break;
          case 4:
            run<dim, 4, MemorySpace::Host>(geometry_type);
            break;
          case 5:
            run<dim, 5, MemorySpace::Host>(geometry_type);
            break;
          case 6:
            run<dim, 6, MemorySpace::Host>(geometry_type);
            break;
        }
    }
#ifdef DEAL_II_COMPILER_CUDA_AWARE
  else if (memory_type == "cuda")
    {
      switch (fe_degree)
        {
          case 1:
            run<dim, 1, MemorySpace::CUDA>(geometry_type);
            break;
          case 2:
            run<dim, 2, MemorySpace::CUDA>(geometry_type);
            break;
          case 3:
            run<dim, 3, MemorySpace::CUDA>(geometry_type);
            break;
          case 4:
            run<dim, 4, MemorySpace::CUDA>(geometry_type);
            break;
          case 5:
            run<dim, 5, MemorySpace::CUDA>(geometry_type);
            break;
          case 6:
            run<dim, 6, MemorySpace::CUDA>(geometry_type);
            break;
        }
    }
#endif
  else
    AssertThrow(false, ExcNotImplemented());
}
