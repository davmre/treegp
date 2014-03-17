#include "cover_tree.hpp"
#include <limits>
#include <pthread.h>
#include <stdint.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>


#include <google/dense_hash_map>




template<class T>
double compare(int *i1, int *i2, const node<T> * children)
{
  return children[*i1].distance_to_query - children[*i2].distance_to_query;
}

static void SWAP(int *a, int *b) {
  int tmp;
  tmp = *a;
  *a = *b;
  *b = tmp;
}

template <class T>
void halfsort (int * permutation, int num_children, node<T> * children)
{
  if (num_children <= 1)
    return;
  register int *base_ptr =  permutation;

  int *hi = &base_ptr[num_children - 1];
  int *right_ptr = hi;
  int *left_ptr;
  int swaps;

  while (right_ptr > base_ptr)
    {
      int *mid = base_ptr + ((hi - base_ptr) >> 1);

      if (compare ( mid,  base_ptr, children) < 0.){
	SWAP (mid, base_ptr);
	swaps++;
      }
      if (compare ( hi,  mid, children) < 0.) {
	SWAP (mid, hi);
	swaps++;
      }
      else
	goto jump_over;
      if (compare ( mid,  base_ptr, children) < 0.) {
	SWAP (mid, base_ptr);
	swaps++;
      }
    jump_over:;

      left_ptr  = base_ptr + 1;
      right_ptr = hi - 1;

      do
	{
	  while (compare (left_ptr, mid, children) < 0.)
	    left_ptr ++;

	  while (compare (mid, right_ptr, children) < 0.)
	    right_ptr --;

	  if (left_ptr < right_ptr)
	    {
	      SWAP (left_ptr, right_ptr);
	      swaps++;
	      if (mid == left_ptr)
		mid = right_ptr;
	      else if (mid == right_ptr)
		mid = left_ptr;
	      left_ptr ++;
	      right_ptr --;
	    }
	  else if (left_ptr == right_ptr)
	    {
	      left_ptr ++;
	      right_ptr --;
	      break;
	    }
	}
      while (left_ptr <= right_ptr);

      hi = right_ptr;
    }
}



class VectorTree {
  node<point> root;
  distfn<point>::Type dfn;
  dfn_deriv ddfn_dx;
  dfn_deriv ddfn_dtheta;
  wfn_deriv dwfn_dr;
  void * dfn_extra;
  wfn w;
  double * wp;
  double *dist_params;
  void set_dist_params(const pyublas::numpy_vector<double> &dist_params);


public:
  unsigned int n;
  int fcalls;
  VectorTree (const pyublas::numpy_matrix<double> &pts, const unsigned int narms,
	      const std::string &distfn_str, const pyublas::numpy_vector<double> &dist_params,
	      const std::string wfn_str,
	      const pyublas::numpy_vector<double> &weight_params);
  void set_v(int v_select, const pyublas::numpy_vector<double> &v);
  pyublas::numpy_vector<double> get_v(int v_select);

  double weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, double eps);


  pyublas::numpy_matrix<double> kernel_matrix(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, bool distance_only);
  pyublas::numpy_matrix<double> sparse_training_kernel_matrix(const pyublas::numpy_matrix<double> &pts, double max_distance, bool distance_only);
  pyublas::numpy_matrix<double> kernel_deriv_wrt_xi(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, int i, int k);
  pyublas::numpy_matrix<double> kernel_deriv_wrt_i(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, int param_i,  bool symmetric, const pyublas::numpy_matrix<double> distances);
  pyublas::numpy_vector<double> sparse_kernel_deriv_wrt_i(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, const pyublas::numpy_vector<int> &nzr, const pyublas::numpy_vector<int> &nzc, int param_i, const pyublas::numpy_vector<double> distance_entries);

  void dump_tree(const std::string &fname);


  ~VectorTree();
};



struct pair_dfn_extra {
  partial_dfn dfn;
  partial_dfn dfn_orig;
  partial_dfn dfn_sq;
  google::dense_hash_map<long, double> *build_cache;
  google::dense_hash_map<int, double> *query1_cache;
  google::dense_hash_map<int, double> *query2_cache;
  void * dfn_extra;
  int NPTS;
  int hits;
  int misses;
};

class MatrixTree {
  node<pairpoint> root_offdiag;
  node<pairpoint> root_diag;
  distfn<pairpoint>::Type raw_pair_dfn;
  distfn<pairpoint>::Type factored_build_dist;
  distfn<pairpoint>::Type factored_query_dist;
  pair_dfn_extra * dfn_extra;
  double * wp_pair;
  double * wp_point;
  wfn w_point;
  wfn w_upper;
  wfn w_lower;
  double max_weight;

  void set_dist_params(const pyublas::numpy_vector<double> &dist_params);
  double *dist_params;


public:
  unsigned int n;
  unsigned int nzero;
  unsigned int use_offdiag;
  int fcalls;
  int dfn_evals;
  MatrixTree (const pyublas::numpy_matrix<double> &pts,
	      const pyublas::numpy_strided_vector<int> &nonzero_rows,
	      const pyublas::numpy_strided_vector<int> &nonzero_cols,
	      const std::string &distfn_str, const pyublas::numpy_vector<double> &dist_params,
	      std::string wfn_str, const pyublas::numpy_vector<double> &weight_params);
  void set_m(const pyublas::numpy_matrix<double> &m);
  void set_m_sparse(const pyublas::numpy_strided_vector<int> &nonzero_rows,
		    const pyublas::numpy_strided_vector<int> &nonzero_cols,
		    const pyublas::numpy_strided_vector<double> &nonzero_vals);

  pyublas::numpy_matrix<double> get_m();
  void collapse_leaf_bins(unsigned int leaf_bin_size);

  double quadratic_form(const pyublas::numpy_matrix<double> &query_pt1,
			const pyublas::numpy_matrix<double> &query_pt2,
			double eps_rel, double eps_abs, int cutoff_rule);

  void print_hierarchy(const pyublas::numpy_matrix<double> &query_pt1, const pyublas::numpy_matrix<double> &query_pt2);

  void test_bounds(double max_d, int n_d);

  ~MatrixTree();
};
