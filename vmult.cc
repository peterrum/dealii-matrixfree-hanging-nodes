#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/cell_weights.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/cell_weights.h"
#include "include/operator_performance.h"
#include "include/scoped_timer.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;

namespace dealii::GridGenerator
{
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
}

template <int dim, int n_components, typename Number = double>
void
run(MPI_Comm comm, const unsigned int n_ref_global,
    const unsigned int n_ref_local,
    const unsigned int fe_degree_fine,
    const double       weight,
    const unsigned int verbosity,
    ConvergenceTable & table)
{
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  (void)table;

  parallel::distributed::Triangulation<dim> tria(
    comm,
    Triangulation<dim>::MeshSmoothing::none,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);

  if(false)
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
  else
  {
      GridGenerator::create_annulus(tria, n_ref_global);
  }

  const auto stat = [&]() {
    unsigned int n_cells_normal  = 0;
    unsigned int n_cells_with_hn = 0;

    for (const auto &cell : tria.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        bool flag = false;

        for (const auto f : cell->face_indices())
          if (!cell->at_boundary(f) &&
              (cell->neighbor(f)->has_children() /*||
               cell->level() != cell->neighbor(f)->level()*/))
            flag = true;

        if (flag)
          n_cells_with_hn++;
        else
          n_cells_normal++;
      }

    std::pair<unsigned int, unsigned int> result = {n_cells_with_hn,
                                                    n_cells_normal};

    const auto results = Utilities::MPI::gather(comm, result);

    if (verbosity)
      if (Utilities::MPI::this_mpi_process(comm) == 0)
        {
          for (const auto &result : results)
            printf("%5d %5d\n", result.first, result.second);
          printf(
            "###########################################################\n");
        }

    // std::cout << n_cells_with_hn << " " << n_cells_normal << " " <<
    // (((double) n_cells_with_hn) / ((double) n_cells_normal +
    // n_cells_with_hn))
    // << std::endl; MPI_Barrier(MPI_COMM_WORLD);
  };

  stat();

  const auto weight_function = parallel::hanging_nodes_weighting<dim>(weight);

  dealii::RepartitioningPolicyTools::CellWeightPolicy<dim> policy(
    weight_function);

  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  tria_pft.create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria, policy.partition(tria)));

  {
    unsigned int counter = 0;

    for (const auto &cell : tria_pft.active_cell_iterators())
      if (cell->is_locally_owned())
        counter++;


    const auto results = Utilities::MPI::gather(comm, counter);

    if (verbosity)
      if (Utilities::MPI::this_mpi_process(comm) == 0)
        {
          for (const auto &result : results)
            printf("%5d\n", result);
          printf(
            "###########################################################\n");
        }
  }


  tria.signals.cell_weight.connect(weight_function);

  tria.repartition();

  stat();

  DoFHandler<dim> dof_handler(tria);

  MappingQ1<dim> mapping;

  const FESystem<dim> fe(FE_Q<dim>{fe_degree_fine}, n_components);
  const QGauss<dim>   quad(fe_degree_fine + 1);

  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  const auto run_vmult = [&](const auto &op, const std::string &label) {
      (void) label;
      
    VectorType solution, rhs;
    op.initialize_dof_vector(solution);
    op.initialize_dof_vector(rhs);
    op.rhs(rhs);
    
    std::cout << rhs.size() << std::endl;

#ifdef LIKWID_PERFMON
    if (label == "gc_pdt")
      LIKWID_MARKER_START(label.c_str());
#endif

    double time = 0.0;
    {
      for (unsigned int i = 0; i < 100; ++i)
        {
          ScopedTimer timer(time);
          op.vmult(solution, rhs);
        }
    }

#ifdef LIKWID_PERFMON
    if (label == "gc_pdt")
      LIKWID_MARKER_STOP(label.c_str());
#endif

    return std::pair<double, unsigned int>(time, solution.size());
  };

  const auto result_global_pft = [&]() {
    DoFHandler<dim> dof_handler(tria_pft);
    dof_handler.distribute_dofs(fe);

    AffineConstraints<Number> constraint;
    IndexSet                  locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraint.reinit(locally_relevant_dofs);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(
                                               n_components),
                                             constraint);
    //DoFTools::make_hanging_node_constraints(dof_handler, constraint);
    constraint.close();

    Operator<dim, n_components, Number> op;
    op.reinit(mapping, dof_handler, quad, constraint);

    return run_vmult(op, "gc_pft");
  }();

  const auto result_global = [&]() {
    AffineConstraints<Number> constraint;
    IndexSet                  locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraint.reinit(locally_relevant_dofs);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(
                                               n_components),
                                             constraint);
    DoFTools::make_hanging_node_constraints(dof_handler, constraint);
    constraint.close();

    if (verbosity)
      {
        IndexSet locally_active_dofs;
        DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);

        types::global_dof_index counter_constrained     = 0;
        types::global_dof_index counter_non_constrained = 0;

        for (const auto i : locally_active_dofs)
          if (constraint.is_constrained(i))
            ++counter_constrained;
          else
            ++counter_non_constrained;

        const auto results_c =
          Utilities::MPI::gather(comm, counter_constrained);
        const auto results_nc =
          Utilities::MPI::gather(comm, counter_non_constrained);


        if (Utilities::MPI::this_mpi_process(comm) == 0)
          {
            for (unsigned int i = 0; i < results_c.size(); ++i)
              printf("%10d %10d\n", results_c[i], results_nc[i]);
            printf(
              "###########################################################\n");
          }
      }

    Operator<dim, n_components, Number> op;
    op.reinit(mapping, dof_handler, quad, constraint);

    return run_vmult(op, "gc_pdt");
  }();

  const auto result_local = [&]() {
    const unsigned int l = tria.n_global_levels() - 1;

    MGConstrainedDoFs         mg_constrained_dofs;
    AffineConstraints<Number> constraint;

    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                       dirichlet_boundary);

    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                  l,
                                                  relevant_dofs);
    constraint.reinit(relevant_dofs);
    constraint.add_lines(mg_constrained_dofs.get_boundary_indices(l));
    constraint.close();

    Operator<dim, n_components, Number> op;
    op.reinit(mapping, dof_handler, quad, constraint, l);

    return run_vmult(op, "ls");
  }();

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    printf("%5.2f %10.6f %10d %10.6f %10d %10.6f %10d\n",
           weight,
           result_global.first,
           result_global.second,
           result_local.first,
           result_local.second,
           result_global_pft.first,
           result_global_pft.second);
}



/**
 * for i in $(seq 100 10 400); do mpirun -np 40 ./vmult  0 7 3 $i 0; done
 *
 * mpirun -np 40 ./vmult  0 7 6 100 0
 */
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  constexpr unsigned int dim = 3;

  const unsigned int n_ref_global = argc > 1 ? atoi(argv[1]) : 3;
  const unsigned int n_ref_local  = argc > 2 ? atoi(argv[2]) : 0;
  const unsigned int degree       = argc > 3 ? atoi(argv[3]) : 3;
  const unsigned int weight       = argc > 4 ? atoi(argv[4]) : 100;
  const unsigned int verbosity    = argc > 5 ? atoi(argv[5]) : 1;

  ConvergenceTable table;
  run<dim, 1, double>(MPI_COMM_SELF,
    n_ref_global, n_ref_local, degree, weight / 100.0, verbosity, table);
  if (false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
