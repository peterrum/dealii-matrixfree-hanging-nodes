using namespace dealii;

namespace
{
  template <int dim, typename number, typename VectorizedArrayType>
  void
  adjust_ghost_range_if_necessary(
    const MatrixFree<dim, number, VectorizedArrayType> &data,
    const LinearAlgebra::distributed::Vector<number> &  vec)
  {
    if (vec.get_partitioner().get() ==
        data.get_dof_info(0).vector_partitioner.get())
      return;

    LinearAlgebra::distributed::Vector<number> copy_vec(vec);
    const_cast<LinearAlgebra::distributed::Vector<number> &>(vec).reinit(
      data.get_dof_info(0).vector_partitioner);
    const_cast<LinearAlgebra::distributed::Vector<number> &>(vec)
      .copy_locally_owned_data_from(copy_vec);
  }

  template <int dim, typename number, typename VectorizedArrayType>
  IndexSet
  get_refinement_edges(
    const MatrixFree<dim, number, VectorizedArrayType> &matrix_free)
  {
    const unsigned int level = matrix_free.get_mg_level();

    std::vector<IndexSet> refinement_edge_indices;
    refinement_edge_indices.clear();
    const unsigned int nlevels =
      matrix_free.get_dof_handler().get_triangulation().n_global_levels();
    refinement_edge_indices.resize(nlevels);
    for (unsigned int l = 0; l < nlevels; l++)
      refinement_edge_indices[l] =
        IndexSet(matrix_free.get_dof_handler().n_dofs(l));

    MGTools::extract_inner_interface_dofs(matrix_free.get_dof_handler(),
                                          refinement_edge_indices);
    return refinement_edge_indices[level];
  }
} // namespace



template <int dim_, int n_components, typename Number>
class Operator : public Subscriptor
{
public:
  using value_type = Number;
  using number     = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  using size_type  = types::global_dof_index;

  static const int dim = dim_;

  using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, number>;

  void
  reinit(const Mapping<dim> &             mapping,
         const DoFHandler<dim> &          dof_handler,
         const Quadrature<dim> &          quad,
         const AffineConstraints<number> &constraints,
         const unsigned int mg_level = numbers::invalid_unsigned_int)
  {
    this->trilinos_system_matrix.clear();

    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_gradients;
    data.mg_level             = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);

    constrained_indices.clear();
    for (auto i : this->matrix_free.get_constrained_dofs())
      constrained_indices.push_back(i);
    constrained_values.resize(constrained_indices.size());

    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      {
        std::vector<types::global_dof_index> interface_indices;
        IndexSet                             refinement_edge_indices;
        refinement_edge_indices = get_refinement_edges(this->matrix_free);
        refinement_edge_indices.fill_index_vector(interface_indices);

        edge_constrained_indices.clear();
        edge_constrained_indices.reserve(interface_indices.size());
        edge_constrained_values.resize(interface_indices.size());
        const IndexSet &locally_owned =
          this->matrix_free.get_dof_handler().locally_owned_mg_dofs(
            this->matrix_free.get_mg_level());
        for (unsigned int i = 0; i < interface_indices.size(); ++i)
          if (locally_owned.is_element(interface_indices[i]))
            edge_constrained_indices.push_back(
              locally_owned.index_within_set(interface_indices[i]));
      }
  }

  virtual types::global_dof_index
  m() const
  {
    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return this->matrix_free.get_dof_handler().n_dofs(
        this->matrix_free.get_mg_level());
    else
      return this->matrix_free.get_dof_handler().n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const
  {
    Assert(false, ExcNotImplemented());
    return 0;
  }

  virtual void
  initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const
  {
#if false
    adjust_ghost_range_if_necessary(matrix_free, dst);
    adjust_ghost_range_if_necessary(matrix_free, src);

    // save values for edge constrained dofs and set them to 0 in src vector
    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        edge_constrained_values[i] = std::pair<number, number>(
          src.local_element(edge_constrained_indices[i]),
          dst.local_element(edge_constrained_indices[i]));

        const_cast<LinearAlgebra::distributed::Vector<number> &>(src)
          .local_element(edge_constrained_indices[i]) = 0.;
      }

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      {
        constrained_values[i] =
          std::pair<number, number>(src.local_element(constrained_indices[i]),
                                    dst.local_element(constrained_indices[i]));

        const_cast<LinearAlgebra::distributed::Vector<number> &>(src)
          .local_element(constrained_indices[i]) = 0.;
      }
#endif

    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_range, this, dst, src, true);

#if false
    // set constrained dofs as the sum of current dst value and src value
    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      {
        const_cast<LinearAlgebra::distributed::Vector<number> &>(src)
          .local_element(constrained_indices[i])  = constrained_values[i].first;
        dst.local_element(constrained_indices[i]) = constrained_values[i].first;
      }

    // restoring edge constrained dofs in src and dst
    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        const_cast<LinearAlgebra::distributed::Vector<number> &>(src)
          .local_element(edge_constrained_indices[i]) =
          edge_constrained_values[i].first;
        dst.local_element(edge_constrained_indices[i]) =
          edge_constrained_values[i].first;
      }
#endif
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void
  vmult_interface_down(VectorType &dst, VectorType const &src) const
  {
    adjust_ghost_range_if_necessary(matrix_free, dst);
    adjust_ghost_range_if_necessary(matrix_free, src);

    // do loop
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_range, this, dst, src, true);

    // make a copy of dst and zero out everything except edge_constraints
    VectorType dst_copy(dst);
    dst = 0.0;

    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        dst.local_element(edge_constrained_indices[i]) =
          dst_copy.local_element(edge_constrained_indices[i]);
      }
  }

  void
  vmult_interface_up(VectorType &dst, VectorType const &src) const
  {
    adjust_ghost_range_if_necessary(matrix_free, dst);
    adjust_ghost_range_if_necessary(matrix_free, src);

    dst = 0.0;

    // make a copy of src vector and set everything to 0 except edge
    // constrained dofs
    VectorType src_cpy;
    src_cpy.reinit(src, /*omit_zeroing_entries=*/false);

    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        src_cpy.local_element(edge_constrained_indices[i]) =
          src.local_element(edge_constrained_indices[i]);
      }

    // do loop with copy of src
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_range, this, dst, src_cpy, false);
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const
  {
    MatrixFreeTools::compute_diagonal(matrix_free,
                                      diagonal,
                                      &Operator::do_cell_integral_local,
                                      this);

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

  virtual const TrilinosWrappers::SparseMatrix &
  get_trilinos_system_matrix() const
  {
    if (trilinos_system_matrix.m() == 0 && trilinos_system_matrix.n() == 0)
      {
        // Set up sparsity pattern of system matrix.
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        TrilinosWrappers::SparsityPattern dsp(
          this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int ?
            dof_handler.locally_owned_mg_dofs(
              this->matrix_free.get_mg_level()) :
            dof_handler.locally_owned_dofs(),
          matrix_free.get_task_info().communicator);

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          MGTools::make_sparsity_pattern(dof_handler,
                                         dsp,
                                         this->matrix_free.get_mg_level(),
                                         this->constraints);
        else
          DoFTools::make_sparsity_pattern(dof_handler, dsp, this->constraints);

        dsp.compress();
        trilinos_system_matrix.reinit(dsp);

        MatrixFreeTools::compute_matrix(matrix_free,
                                        constraints,
                                        trilinos_system_matrix,
                                        &Operator::do_cell_integral_local,
                                        this);
      }

    return this->trilinos_system_matrix;
  }

  void
  rhs(VectorType &b) const
  {
    const int dummy = 0;

    matrix_free.template cell_loop<VectorType, int>(
      [](const auto &matrix_free, auto &dst, const auto &, const auto cells) {
        FECellIntegrator phi(matrix_free, cells);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              if constexpr (n_components == 1)
                {
                  phi.submit_value(1.0, q);
                }
              else
                {
                  Tensor<1, n_components> temp;

                  for (int i = 0; i < n_components; ++i)
                    temp[i] = 1.0;

                  phi.submit_value(temp, q);
                }

            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      b,
      dummy,
      true);
  }

private:
  void
  do_cell_integral_local(FECellIntegrator &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
  }

  void
  do_cell_integral_global(FECellIntegrator &integrator,
                          VectorType &      dst,
                          const VectorType &src) const
  {
#if true
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
#elif true
    integrator.read_dof_values(src);
    integrator.distribute_local_to_global(dst);
#else
    integrator.read_dof_values(src);
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
    integrator.distribute_local_to_global(dst);
#endif
  }

  void
  do_cell_integral_range(
    const MatrixFree<dim, number> &              matrix_free,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);

    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);

        do_cell_integral_global(integrator, dst, src);
      }
  }

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;

  mutable TrilinosWrappers::SparseMatrix trilinos_system_matrix;

  /**
   * Constrained indices.
   *
   * @note Needed in matrix-free vmults.
   */
  std::vector<unsigned int> constrained_indices;

  /**
   * Constrained values.
   *
   * @note Needed in matrix-free vmults.
   */
  mutable std::vector<std::pair<number, number>> constrained_values;

  /**
   * Edge-constrained indices.
   *
   * @note Needed in matrix-free vmults.
   */
  std::vector<unsigned int> edge_constrained_indices;

  /**
   * Edge-constrained values.
   *
   * @note Needed in matrix-free vmults.
   */
  mutable std::vector<std::pair<number, number>> edge_constrained_values;
};
