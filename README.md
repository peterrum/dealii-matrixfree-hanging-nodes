# Efficient application of hanging-node constraints for matrix-free high-order FEM computations on CPU and GPU

This project contains benchmarks for testing the application of hanging-node constraints for matrix-free high-order FEM computations within deal.II. It is the basis of the publication:

```
@article{munch2022hn,
  title  = {Efficient application of hanging-node constraints for matrix-free 
            high-order FEM computations on CPU and GPU},
  author = {Munch, Peter and Ljungkvist, Karl and Kronbichler, Martin},
  year   = {2022, accepted for publication},
}
```

### Compilation

We use the the following deal.II branch: https://github.com/peterrum/dealii/tree/compressed_constraint_kind_use. To disable the automatic choice of the vectorization type by the library, we have added the following lines:

```diff
diff --git a/include/deal.II/matrix_free/evaluation_kernels_hanging_nodes.h b/include/deal.II/matrix_free/evaluation_kernels_hanging_nodes.h
index 12a97f2bf9..7cab6f47ba 100644
--- a/include/deal.II/matrix_free/evaluation_kernels_hanging_nodes.h
+++ b/include/deal.II/matrix_free/evaluation_kernels_hanging_nodes.h
@@ -34,6 +34,8 @@ DEAL_II_NAMESPACE_OPEN
 #  define DEAL_II_ALWAYS_INLINE_RELEASE DEAL_II_ALWAYS_INLINE
 #endif
 
+#define HN_TYPE 0 // 0: (scalar, index), 1: (scalar, sorted), 2: (vectorized)
+
 
 
 namespace internal
@@ -1469,7 +1471,11 @@ namespace internal
   {
   public:
     static const VectorizationTypes VectorizationType =
+#if HN_TYPE == 0
       VectorizationTypes::index;
+#else
+      VectorizationTypes::sorted;
+#endif
 
   private:
     template <unsigned int side, bool transpose>
@@ -1737,9 +1743,15 @@ namespace internal
     static constexpr FEEvaluationImplHangingNodesRunnerTypes
     used_runner_type()
     {
+#if HN_TYPE <= 1
+  return FEEvaluationImplHangingNodesRunnerTypes::scalar;
+#elif HN_TYPE == 2
+  return FEEvaluationImplHangingNodesRunnerTypes::vectorized;
+#else
       return ((Number::size() > 2) && (fe_degree == -1 || fe_degree > 2)) ?
                FEEvaluationImplHangingNodesRunnerTypes::vectorized :
                FEEvaluationImplHangingNodesRunnerTypes::scalar;
+#endif
     }
   };
```
