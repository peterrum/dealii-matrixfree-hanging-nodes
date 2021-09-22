// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


#ifndef dealii_matrix_free_evaluation_kernels_h_
#define dealii_matrix_free_evaluation_kernels_h_

#include <deal.II/base/config.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/matrix_free/dof_info.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/hanging_nodes_internal.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/matrix_free/type_traits.h>


DEAL_II_NAMESPACE_OPEN


// forward declaration
template <int, typename, bool, typename>
class FEEvaluationBaseData;



namespace internal
{
  template <int dim, typename Number, bool is_face>
  struct MyFEEvaluationImplHangingNodes
  {
    template <int fe_degree, int n_q_points_1d>
    static bool
    run(const unsigned int                  n_desired_components,
        const FEEvaluationBaseData<dim,
                                   typename Number::value_type,
                                   is_face,
                                   Number> &fe_eval,
        const bool                          transpose,
        const std::array<MatrixFreeFunctions::ConstraintKinds, Number::size()>
          &     c_mask,
        Number *values)
    {
      if (transpose)
        run_internal<fe_degree, true>(n_desired_components,
                                      fe_eval,
                                      c_mask,
                                      values);
      else
        run_internal<fe_degree, false>(n_desired_components,
                                       fe_eval,
                                       c_mask,
                                       values);

      return false;
    }

  private:
    template <int fe_degree, unsigned int side, bool transpose>
    static inline DEAL_II_ALWAYS_INLINE void
    interpolate_2D(const unsigned int given_degree,
                   bool               is_subface_0,
                   const unsigned int v,
                   const Number *     weight,
                   Number *           values)
    {
      if (is_subface_0)
        interpolate_2D<fe_degree, side, transpose, true>(given_degree,
                                                         v,
                                                         weight,
                                                         values);
      else
        interpolate_2D<fe_degree, side, transpose, false>(given_degree,
                                                          v,
                                                          weight,
                                                          values);
    }

    template <int          fe_degree,
              unsigned int side,
              bool         transpose,
              bool         is_subface_0>
    static inline DEAL_II_ALWAYS_INLINE void
    interpolate_2D(const unsigned int given_degree,
                   const unsigned int v,
                   const Number *     weight,
                   Number *           values)
    {
      typename Number::value_type temp[40];

      const unsigned int points =
        (fe_degree != -1 ? fe_degree : given_degree) + 1;

      AssertIndexRange(points, 40);

      const unsigned int d = side / 2; // direction
      const unsigned int s = side % 2; // left or right surface

      const unsigned int offset = dealii::Utilities::pow(points, d + 1);
      const unsigned int stride =
        (s == 0 ? 0 : (points - 1)) * dealii::Utilities::pow(points, d);

      const unsigned int r1 = dealii::Utilities::pow(points, dim - d - 1);
      const unsigned int r2 = dealii::Utilities::pow(points, d);

      // copy result back
      for (unsigned int i = 0, k = 0; i < r1; ++i)
        for (unsigned int j = 0; j < r2; ++j, ++k)
          temp[k] = values[i * offset + stride + j][v];

      // perform interpolation point by point (note: r1 * r2 == points^(dim-1))
      for (unsigned int i = 0, k = 0; i < r1; ++i)
        for (unsigned int j = 0; j < r2; ++j, ++k)
          {
            typename Number::value_type sum = 0.0;
            for (unsigned int h = 0; h < points; ++h)
              sum += weight[((transpose ? 1 : points) *
                             (is_subface_0 ? k : (points - 1 - k))) +
                            ((transpose ? points : 1) *
                             (is_subface_0 ? h : (points - 1 - h)))][v] *
                     temp[h];
            values[i * offset + stride + j][v] = sum;
          }
    }

    template <int          fe_degree,
              unsigned int direction,
              unsigned int side,
              bool         transpose>
    static inline DEAL_II_ALWAYS_INLINE void
    interpolate_3D_face_all(const unsigned int dof_offset,
                            const unsigned int given_degree,
                            bool               is_subface_0,
                            const unsigned int v,
                            const Number *     weight,
                            Number *           values)
    {
      if (is_subface_0)
        interpolate_3D_face_all<fe_degree, direction, side, transpose, true>(
          dof_offset, given_degree, v, weight, values);
      else
        interpolate_3D_face_all<fe_degree, direction, side, transpose, false>(
          dof_offset, given_degree, v, weight, values);
    }

    template <int          fe_degree,
              unsigned int direction,
              unsigned int side,
              bool         transpose,
              bool         is_subface_0>
    static inline DEAL_II_ALWAYS_INLINE void
    interpolate_3D_face_all(const unsigned int dof_offset,
                            const unsigned int given_degree,
                            const unsigned int v,
                            const Number *     weight,
                            Number *           values)
    {
      typename Number::value_type temp[40];

      const unsigned int points =
        (fe_degree != -1 ? fe_degree : given_degree) + 1;

      AssertIndexRange(points, 40);

      const unsigned int stride = Utilities::pow(points, direction);
      const unsigned int d      = side / 2;

      // direction   side0   side1   side2
      // 0             -      p^2      p
      // 1            p^2      -       1
      // 2             p       -       1
      const unsigned int stride2 =
        ((direction == 0 && d == 1) || (direction == 1 && d == 0)) ?
          (points * points) :
          (((direction == 0 && d == 2) || (direction == 2 && d == 0)) ? points :
                                                                        1);

      for (unsigned int g = 0; g < points; ++g)
        {
          // copy result back
          for (unsigned int k = 0; k < points; ++k)
            temp[k] = values[dof_offset + k * stride + stride2 * g][v];

          // perform interpolation point by point
          for (unsigned int k = 0; k < points; ++k)
            {
              typename Number::value_type sum = 0.0;
              for (unsigned int h = 0; h < points; ++h)
                sum += weight[((transpose ? 1 : points) *
                               (is_subface_0 ? k : (points - 1 - k))) +
                              ((transpose ? points : 1) *
                               (is_subface_0 ? h : (points - 1 - h)))][v] *
                       temp[h];
              values[dof_offset + k * stride + stride2 * g][v] = sum;
            }
        }
    }

    template <int          fe_degree,
              unsigned int direction,
              unsigned int side,
              bool         transpose>
    static inline DEAL_II_ALWAYS_INLINE void
    interpolate_3D_face(const unsigned int dof_offset,
                        const unsigned int given_degree,
                        bool               is_subface_0,
                        const unsigned int v,
                        const Number *     weight,
                        Number *           values)
    {
      if (is_subface_0)
        interpolate_3D_face<fe_degree, direction, side, transpose, true>(
          dof_offset, given_degree, v, weight, values);
      else
        interpolate_3D_face<fe_degree, direction, side, transpose, false>(
          dof_offset, given_degree, v, weight, values);
    }

    template <int          fe_degree,
              unsigned int direction,
              unsigned int side,
              bool         transpose,
              bool         is_subface_0>
    static inline DEAL_II_ALWAYS_INLINE void
    interpolate_3D_face(const unsigned int dof_offset,
                        const unsigned int given_degree,
                        const unsigned int v,
                        const Number *     weight,
                        Number *           values)
    {
      typename Number::value_type temp[40];

      const unsigned int points =
        (fe_degree != -1 ? fe_degree : given_degree) + 1;

      AssertIndexRange(points, 40);

      const unsigned int stride = Utilities::pow(points, direction);
      const unsigned int d      = side / 2;

      // direction   side0   side1   side2
      // 0             -      p^2      p
      // 1            p^2      -       1
      // 2             p       -       1
      const unsigned int stride2 =
        ((direction == 0 && d == 1) || (direction == 1 && d == 0)) ?
          (points * points) :
          (((direction == 0 && d == 2) || (direction == 2 && d == 0)) ? points :
                                                                        1);

      for (unsigned int g = 1; g < points - 1; ++g)
        {
          // copy result back
          for (unsigned int k = 0; k < points; ++k)
            temp[k] = values[dof_offset + k * stride + stride2 * g][v];

          // perform interpolation point by point
          for (unsigned int k = 0; k < points; ++k)
            {
              typename Number::value_type sum = 0.0;
              for (unsigned int h = 0; h < points; ++h)
                sum += weight[((transpose ? 1 : points) *
                               (is_subface_0 ? k : (points - 1 - k))) +
                              ((transpose ? points : 1) *
                               (is_subface_0 ? h : (points - 1 - h)))][v] *
                       temp[h];
              values[dof_offset + k * stride + stride2 * g][v] = sum;
            }
        }
    }

    template <int fe_degree, unsigned int direction, bool transpose>
    static inline DEAL_II_ALWAYS_INLINE void
    interpolate_3D_edge(const unsigned int p,
                        const unsigned int given_degree,
                        bool               is_subface_0,
                        const unsigned int v,
                        const Number *     weight,
                        Number *           values)
    {
      if (is_subface_0)
        interpolate_3D_edge<fe_degree, direction, transpose, true>(
          p, given_degree, v, weight, values);
      else
        interpolate_3D_edge<fe_degree, direction, transpose, false>(
          p, given_degree, v, weight, values);
    }

    template <int          fe_degree,
              unsigned int direction,
              bool         transpose,
              bool         is_subface_0>
    static inline DEAL_II_ALWAYS_INLINE void
    interpolate_3D_edge(const unsigned int p,
                        const unsigned int given_degree,
                        const unsigned int v,
                        const Number *     weight,
                        Number *           values)
    {
      typename Number::value_type temp[40];

      const unsigned int points =
        (fe_degree != -1 ? fe_degree : given_degree) + 1;

      AssertIndexRange(points, 40);

      const unsigned int stride = Utilities::pow(points, direction);

      // copy result back
      for (unsigned int k = 0; k < points; ++k)
        temp[k] = values[p + k * stride][v];

      // perform interpolation point by point
      for (unsigned int k = 0; k < points; ++k)
        {
          typename Number::value_type sum = 0.0;
          for (unsigned int h = 0; h < points; ++h)
            sum += weight[((transpose ? 1 : points) *
                           (is_subface_0 ? k : (points - 1 - k))) +
                          ((transpose ? points : 1) *
                           (is_subface_0 ? h : (points - 1 - h)))][v] *
                   temp[h];
          values[p + k * stride][v] = sum;
        }
    }

    template <int fe_degree, bool transpose>
    static void
    run_internal(const unsigned int                  n_desired_components,
                 const FEEvaluationBaseData<dim,
                                            typename Number::value_type,
                                            is_face,
                                            Number> &fe_eval,
                 const std::array<MatrixFreeFunctions::ConstraintKinds,
                                  Number::size()> &  constraint_mask,
                 Number *                            values)
    {
      const Number *weights = fe_eval.get_shape_info()
                                .data.front()
                                .subface_interpolation_matrix.data();

      const unsigned int given_degree =
        fe_degree != -1 ? fe_degree :
                          fe_eval.get_shape_info().data.front().fe_degree;

      const unsigned int points = given_degree + 1;

      for (unsigned int c = 0; c < n_desired_components; ++c)
        {
          for (unsigned int v = 0; v < Number::size(); ++v)
            {
              const auto mask = constraint_mask[v];

              if (mask == MatrixFreeFunctions::ConstraintKinds::unconstrained)
                continue;

              if (dim == 2) // 2D: only faces
                {
                  const auto is_set = [](const auto a, const auto b) -> bool {
                    return (a & b) == b;
                  };

                  // direction 0:
                  if ((mask & MatrixFreeFunctions::ConstraintKinds::face_y) !=
                      MatrixFreeFunctions::ConstraintKinds::unconstrained)
                    {
                      const bool is_subface_0 =
                        (mask & MatrixFreeFunctions::ConstraintKinds::type_x) !=
                        MatrixFreeFunctions::ConstraintKinds::unconstrained;
                      if (is_set(mask,
                                 MatrixFreeFunctions::ConstraintKinds::type_y))
                        interpolate_2D<fe_degree, 2, transpose>(
                          given_degree,
                          is_subface_0,
                          v,
                          weights,
                          values); // face 2
                      else
                        interpolate_2D<fe_degree, 3, transpose>(
                          given_degree,
                          is_subface_0,
                          v,
                          weights,
                          values); // face 3
                    }

                  // direction 1:
                  if ((mask & MatrixFreeFunctions::ConstraintKinds::face_x) !=
                      MatrixFreeFunctions::ConstraintKinds::unconstrained)
                    {
                      const bool is_subface_0 =
                        (mask & MatrixFreeFunctions::ConstraintKinds::type_y) !=
                        MatrixFreeFunctions::ConstraintKinds::unconstrained;
                      if (is_set(mask,
                                 MatrixFreeFunctions::ConstraintKinds::type_x))
                        interpolate_2D<fe_degree, 0, transpose>(
                          given_degree,
                          is_subface_0,
                          v,
                          weights,
                          values); // face 0
                      else
                        interpolate_2D<fe_degree, 1, transpose>(
                          given_degree,
                          is_subface_0,
                          v,
                          weights,
                          values); // face 1
                    }
                }
              else if (dim == 3) // 3D faces and edges
                {
                  const unsigned int p0 = 0;
                  const unsigned int p1 = points - 1;
                  const unsigned int p2 = points * points - points;
                  const unsigned int p3 = points * points - 1;
                  const unsigned int p4 =
                    points * points * points - points * points;
                  const unsigned int p5 =
                    points * points * points - points * points + points - 1;
                  const unsigned int p6 = points * points * points - points;

                  const auto m = static_cast<std::uint16_t>(mask);

                  const bool type_x = (m >> 0) & 1;
                  const bool type_y = (m >> 1) & 1;
                  const bool type_z = (m >> 2) & 1;

                  static const std::array<unsigned int, 12> line_to_point = {{
                    p0, // 0
                    p1, // 1
                    p0, // 2
                    p2, // 3
                    p4, // 4
                    p5, // 5
                    p4, // 6
                    p6, // 7
                    p0, // 8
                    p1, // 9
                    p2, // 10
                    p3  // 11
                  }};

                  static const std::
                    array<std::array<std::array<unsigned int, 2>, 2>, 3>
                      line = {{{{{{7, 3}}, {{6, 2}}}},
                               {{{{5, 1}}, {{4, 0}}}},
                               {{{{11, 9}}, {{10, 8}}}}}};

                  const auto faces = (m >> 3) & 7;
                  const auto edges = (m >> 6);

                  if (edges > 0)
                    {
                      const auto process_edge_x = [&]() {
                        interpolate_3D_edge<fe_degree, 0, transpose>(
                          line_to_point[line[0][type_y][type_z]],
                          given_degree,
                          type_x,
                          v,
                          weights,
                          values);
                      };

                      const auto process_edge_y = [&]() {
                        interpolate_3D_edge<fe_degree, 1, transpose>(
                          line_to_point[line[1][type_x][type_z]],
                          given_degree,
                          type_y,
                          v,
                          weights,
                          values);
                      };

                      const auto process_edge_z = [&]() {
                        interpolate_3D_edge<fe_degree, 2, transpose>(
                          line_to_point[line[2][type_x][type_y]],
                          given_degree,
                          type_z,
                          v,
                          weights,
                          values);
                      };
                      
                      static const void *s[8] = {&&s0, &&s1, &&s2, &&s3, &&s4, &&s5, &&s6, &&s7};

                      goto *s[edges];

                        {
                          s0:
                            goto end;
                          s1:
                            process_edge_z();
                            goto end;
                          s2:
                            process_edge_x();
                            goto end;
                          s3:
                            process_edge_x();
                            process_edge_z();
                            goto end;
                          s4:
                            process_edge_y();
                            goto end;
                          s5:
                            process_edge_y();
                            process_edge_z();
                            goto end;
                          s6:
                            process_edge_x();
                            process_edge_y();
                            goto end;
                          s7:
                            process_edge_x();
                            process_edge_y();
                            process_edge_z();
                            goto end;
                        }
                      
                              end:
                        (void) edges;
                    }
/*
                  if (faces > 0)
                    {
                      switch (faces)
                        {
                          case 0:
                            break;
                          case 1:
                            interpolate_3D_face_all<fe_degree, 1, 0, transpose>(
                              p0,
                              given_degree,
                              type_y,
                              v,
                              weights,
                              values); // face 0

                            interpolate_3D_face_all<fe_degree, 2, 0, transpose>(
                              p0,
                              given_degree,
                              type_z,
                              v,
                              weights,
                              values); // face 0

                            break;
                          case 2:
                            interpolate_3D_face_all<fe_degree, 0, 2, transpose>(
                              p0,
                              given_degree,
                              type_x,
                              v,
                              weights,
                              values); // face 2

                            interpolate_3D_face_all<fe_degree, 2, 2, transpose>(
                              p0,
                              given_degree,
                              type_z,
                              v,
                              weights,
                              values); // face 2

                            break;
                          case 3:
                            break;
                          case 4:
                            interpolate_3D_face_all<fe_degree, 0, 4, transpose>(
                              p0,
                              given_degree,
                              type_x,
                              v,
                              weights,
                              values); // face 4

                            interpolate_3D_face_all<fe_degree, 1, 4, transpose>(
                              p0,
                              given_degree,
                              type_y,
                              v,
                              weights,
                              values); // face 4

                            break;
                          case 5:
                            break;
                          case 6:
                            break;
                          case 7:
                            break;
                        }
                    }
 */
                }
              else
                {
                  Assert(false, ExcNotImplemented());
                }
            }

          values += fe_eval.get_shape_info().dofs_per_component_on_cell;
        }
    }
  };


} // end of namespace internal


DEAL_II_NAMESPACE_CLOSE

#endif
