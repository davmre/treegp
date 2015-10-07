#include "cover_tree.hpp"
#include "vector_mult.hpp"


#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <cmath>
#include <memory>
#include <vector>
#include <limits>

#include <sys/time.h>



#include <google/dense_hash_map>
using google::dense_hash_map;


using namespace std;
namespace bp = boost::python;

double weighted_sum_node(node<point> &n, int v_select,
			 const point &query_pt, double eps_abs,
			 int &terms_sofar,
			 double &abserr_sofar,
			 double &ws,
			 int max_terms,
			 int &nodes_touched, int &terms, int &dfn_evals, int &wfn_evals,
			 wfn w,
			 distfn<point>::Type dist,
			 const double * dist_params,
			 void * dist_extra,
			 const double* weight_params) {

  double d = n.distance_to_query; // avoid duplicate distance
				    // calculations by assuming this
				    // distance has already been
				    // computed by the parent, in the
				    // recursive expansion below. Note
				    // this calculation must be done
				    // explicitly at the root before
				    // this function is called.
  bool cutoff = false;

  nodes_touched += 1;

  // if (fabs(n.p.p[1]) < .0000001) {
    //}

  if (n.num_children == 0) {
    // if we're at a leaf, just do the multiplication

    double weight = w(d, weight_params);
    wfn_evals += 1;

    ws += weight * n.unweighted_sums[v_select];
    //printf("at leaf: ws = %lf*%lf = %lf\n", weight, n.unweighted_sums[v_select], ws);
    cutoff = true;

    terms += 1;

  } else {
    bool query_in_bounds = (d <= n.max_dist);
    if (!query_in_bounds) {
      double min_weight = w(d + n.max_dist, weight_params);
      double max_weight = w(max(0.0, d - n.max_dist), weight_params);
      wfn_evals += 2;


      double frac_remaining_terms = n.num_leaves / (double)(max_terms - terms_sofar);
      double threshold = frac_remaining_terms * (eps_abs - abserr_sofar);
      double abserr_n = .5 * (max_weight - min_weight) * n.unweighted_sums_abs[v_select];

      cutoff = abserr_n < threshold;
      if (cutoff) {
	// if we're cutting off, just compute an estimate of the sum
	// in this region
	ws += .5 * (max_weight + min_weight) * n.unweighted_sums[v_select];
	terms += 1;


	//printf("cutting off: ws = %lf*%lf = %lf\n", .5 * (max_weight + min_weight), n.unweighted_sums[v_select], ws);
	//printf("cutting off: %d leaves (representing %.1f%% of %d-%d remaining), error bound %.8f, error budget %.4f - %.4f = %.4f, so we would have been allowed %.6f\n", n.num_leaves, frac_remaining_terms*100, max_terms, terms_sofar, abserr_n, eps_abs, abserr_sofar, eps_abs - abserr_sofar, threshold);

	 terms_sofar += n.num_leaves;
	 abserr_sofar += abserr_n;
      } else{

	//printf("NOT cutting off: %d leaves (representing %.1f%% of %d-%d remaining), error bound %.8f, error budget %.4f - %.4f = %.4f, so we would have been allowed %.6f\n", n.num_leaves, frac_remaining_terms*100, max_terms, terms_sofar, abserr_n, eps_abs, abserr_sofar, eps_abs - abserr_sofar, threshold);
      }



    }
    if (!cutoff) {
      // if not cutting off, we expand the sum recursively at the
      // children of this node, from nearest to furthest.

      int small_perm[10];
      int * permutation = (int *)&small_perm;
      if(n.num_children > 10) {
	permutation = (int *)malloc(n.num_children * sizeof(int));
      }

      for(int i=0; i < n.num_children; ++i) {
	n.children[i].distance_to_query = dist(query_pt, n.children[i].p, std::numeric_limits< double >::max(), dist_params, dist_extra);
	dfn_evals += 1;
	permutation[i] = i;
      }
      halfsort(permutation, n.num_children, n.children);
      for(int i=0; i < n.num_children; ++i) {
	weighted_sum_node(n.children[permutation[i]], v_select,
			  query_pt, eps_abs, terms_sofar, abserr_sofar, ws, max_terms,
			  nodes_touched, terms, dfn_evals, wfn_evals,
			  w, dist, dist_params, dist_extra, weight_params);
      }
      if (permutation != (int *)&small_perm) {
	free(permutation);
      }

    }
  }
  return ws;
}


void set_v_node (node<point> &n, int v_select, const std::vector<double> &v) {
  if (n.num_children == 0) {
    n.unweighted_sums[v_select] = v[n.p.idx];
    n.unweighted_sums_abs[v_select] = fabs(v[n.p.idx]);
  } else {
    n.unweighted_sums[v_select] = 0;
    n.unweighted_sums_abs[v_select] = 0;
    for(int i=0; i < n.num_children; ++i) {
      set_v_node(n.children[i], v_select, v);
      n.unweighted_sums[v_select] += n.children[i].unweighted_sums[v_select];
      n.unweighted_sums_abs[v_select] += n.children[i].unweighted_sums_abs[v_select];
    }
  }
}

void get_v_node(node<point> &n, int v_select, std::vector<double> &v) {
  if (n.num_children == 0) {
    v[n.p.idx] = n.unweighted_sums[v_select];
  } else {
    for(int i=0; i < n.num_children; ++i) {
      get_v_node(n.children[i], v_select, v);
    }
  }
}


double VectorTree::weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, double eps) {
  point qp = {&query_pt(0,0), 0};
  double weight_sofar = 0;

  this->nodes_touched = 0;
  this->terms = 0;
  this->dfn_evals = 1;
  this->wfn_evals = 0;

  if (this->n == 0) {
    return 0;
  }

  double ws = 0;
  int terms_sofar = 0;
  double abserr_sofar = 0;
  int max_terms = this->root.num_leaves;

  this->root.distance_to_query = this->dfn(qp, this->root.p, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
  weighted_sum_node(this->root, v_select,
		    qp, eps, terms_sofar,
		    abserr_sofar, ws,
		    max_terms,
		    this->nodes_touched, this->terms, this->dfn_evals, this->wfn_evals, this->w,
		    this->dfn, this->dist_params,
		    this->dfn_extra, this->wp);

  return ws;
}

void dump_clusters_node (node<point> &n, int depth, int cluster_size, FILE * fp) {
  if (n.num_leaves < cluster_size) {
    fprintf(fp, "%f %f %f %d\n", n.p.p[0], n.p.p[1], n.p.p[2], n.num_leaves);
  } else {
    for(int i=0; i < n.num_children; ++i) {
      dump_clusters_node(n.children[i], depth+1, cluster_size, fp);
    }
  }
}

void VectorTree::dump_clusters(const string &fname, int cluster_size) {
  FILE * fp = fopen(fname.c_str(), "w");
  dump_clusters_node(this->root, 0, cluster_size, fp);
  fclose(fp);
}

void dump_tree_node (node<point> &n, int depth, FILE * fp) {
  fprintf(fp, "%f %f %f %d\n", n.p.p[0], n.p.p[1], n.max_dist, depth);
  for(int i=0; i < n.num_children; ++i) {
    dump_tree_node(n.children[i], depth+1, fp);
  }
}

void VectorTree::dump_tree(const string &fname) {
  FILE * fp = fopen(fname.c_str(), "w");
  dump_tree_node(this->root, 0, fp);
  fclose(fp);
}

void VectorTree::set_v(int v_select, const pyublas::numpy_vector<double> &v) {
  if (v.ndim() != 1) {
    printf("error: tree can only hold 1D arrays! (array passed has %lu dimensions)\n", v.ndim());
    exit(1);
  }
  npy_intp item_stride = v.strides()[0] / v.itemsize();
  vector<double> new_v(v.size());
  for (pyublas::numpy_vector<double>::const_iterator a = v.begin(); a < v.end(); a += item_stride) {
    new_v[ (a - v.begin())/item_stride] = *a;
  }
  set_v_node(this->root, v_select, new_v);
}

pyublas::numpy_vector<double> VectorTree::get_v(int v_select) {
  vector<double> v(this->n);
  get_v_node(this->root, v_select, v);

  pyublas::numpy_vector<double> pv(this->n);
  for (unsigned int i = 0; i < this->n; ++i) {
    pv(i) = v[i];
  }

  return pv;
}

VectorTree::VectorTree (const pyublas::numpy_matrix<double> &pts,
			const unsigned int narms,
			const string &distfn_str,
			const pyublas::numpy_vector<double> &dist_params,
			const string wfn_str,
			const pyublas::numpy_vector<double> &weight_params) {
  vector< point > points(pts.size1());
  for (unsigned i = 0; i < pts.size1 (); ++ i) {
    point p = {&pts (i, 0), i};
    points[i] = p;
  }
  this->n = pts.size1();
  this->ddfn_dtheta = NULL;
  this->ddfn_dx = NULL;
  if (distfn_str.compare("lld") == 0) {
    this->dfn = dist_3d_km;
    this->dfn_extra = NULL;
    this->ddfn_dtheta = dist3d_deriv_wrt_theta;
    this->ddfn_dx = dist3d_deriv_wrt_xi;
  } else if (distfn_str.compare("euclidean") == 0) {
    this->dfn = dist_euclidean;
    this->dfn_extra = malloc(2*sizeof(int));
    this->ddfn_dx = dist_euclidean_deriv_wrt_xi;
    this->ddfn_dtheta = dist_euclidean_deriv_wrt_theta;
    *((int *) this->dfn_extra) = pts.size2();

    if (dist_params.size() != pts.size2()) {
      printf("ERROR: computing Euclidean distances in %d dimensions, but only %d lengthscales specified!\n", pts.size2(), dist_params.size());
      exit(-1);
    }

  } else if (distfn_str.compare("lldlld") == 0) {
    this->dfn = dist_6d_km;
    this->dfn_extra = NULL;
    this->ddfn_dtheta = dist6d_deriv_wrt_theta;
  } else {
    printf("error: unrecognized distance function %s\n", distfn_str.c_str());
    exit(1);
  }

  this->dist_params = NULL;
  this->set_dist_params(dist_params);

  this->dwfn_dr = NULL;
  if (wfn_str.compare("se") == 0) {
    this->w = w_se;
    this->dwfn_dr = deriv_se_wrt_r;
  } else if (wfn_str.compare("matern32") == 0) {
    this->w = w_matern32;
    this->dwfn_dr = deriv_matern32_wrt_r;
  } else if (wfn_str.compare("compact0") == 0) {
    this->w = w_compact_q0;
  } else if (wfn_str.compare("compact2") == 0) {
    this->w = w_compact_q2;
    this->dwfn_dr = deriv_compact_q2_wrt_r;
  } else {
    printf("error: unrecognized weight function %s\n", wfn_str.c_str());
    exit(1);
  }

  this->wp = NULL;
  int n_wp = 0;
  n_wp += weight_params.size();
  if (wfn_str.compare(0, 7, "compact") == 0) {
    n_wp += 1;
  }
  if (n_wp > 0) {
    this->wp = new double[n_wp];
  }
  for (unsigned i = 0; i < weight_params.size(); ++i) {
    this->wp[i] = weight_params(i);
  }
  if (wfn_str.compare(0, 7, "compact") == 0) {
    int D = pts.size2();
    int q = atoi(wfn_str.c_str()+7);
    double j = floor(D/2) + q+ 1.0;
    //printf("compact weight, D=%d, q=%d, j=%f\n", D, q, j);
    this->wp[n_wp-1] = j;
  }

  if (this->n > 0) {
    this->root = batch_create(points, this->dfn, this->dist_params, this->dfn_extra);
    node<point> * a = NULL;
    set_leaves(this->root, a);
    this->root.alloc_arms(narms);
  }

}

void VectorTree::set_dist_params(const pyublas::numpy_vector<double> &dist_params) {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
  this->dist_params = new double[dist_params.size()];
  for (unsigned i = 0; i < dist_params.size(); ++i) {
    this->dist_params[i] = dist_params(i);
  }
}


pyublas::numpy_matrix<double> VectorTree::kernel_matrix(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, bool distance_only) {

  pyublas::numpy_matrix<double> K(pts1.size1(), pts2.size1());
  for (unsigned i = 0; i < pts1.size1 (); ++ i) {
    point p1 = {&pts1(i, 0), 0};
    for (unsigned j = 0; j < pts2.size1 (); ++ j) {
      point p2 = {&pts2(j, 0), 0};
      double d = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
      K(i,j) = distance_only ? d : this->w(d, this->wp);
    }
  }

  return K;
}


pyublas::numpy_matrix<double> VectorTree::sparse_training_kernel_matrix(const pyublas::numpy_matrix<double> &pts, double max_distance, bool distance_only) {

  pyublas::numpy_matrix<double> K(pts.size1()*2, 3);

  unsigned long nzero = 0;
  //printf("saw %d points\n", pts.size1 ());
  for (unsigned i = 0; i < pts.size1 (); ++ i) {
    v_array<v_array<point> > res;
    point p1 = {&pts(i, 0), 0};

    node<point> np1;
    np1.p = p1;
    np1.max_dist = 0.;
    np1.parent_dist = 0.;
    np1.children = NULL;
    np1.num_children = 0;
    np1.scale = 100;

    epsilon_nearest_neighbor(this->root,np1,res,max_distance, this->dfn, this->dist_params, this->dfn_extra);


    //printf("nn got %d results for %f, %f\n", res[0].index, p1.p[0], p1.p[1]);
    //printf("dfn params %f %f\n", this->dist_params[0], this->dist_params[1]);
    //printf("root is %p and has %d children\n", &this->root, this->root.num_children);

    int min_idx = res[0].index > 1 ? 1 : 0;
    for(int jj = min_idx; jj < res[0].index; ++jj) {
      point p2 = res[0][jj];
      int j = p2.idx;

      double d = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);

      //printf("point %f, %f is at distance %f\n", p2.p[0], p2.p[1], d);

      if (nzero == K.size1()) {
	K.resize(K.size1()*2, 3);
      }
      K(nzero,0) = i;
      K(nzero,1) = j;
      K(nzero,2) = distance_only ? d : this->w(d, this->wp);
      nzero++;
    }
    //printf("inserted %d neighbors for point %d\n", res[0].index-1, i);
  }
  K.resize(nzero, 3);
  return K;
}

pyublas::numpy_matrix<double> VectorTree::kernel_deriv_wrt_xi(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, int i, int k) {
  // deriv of kernel matrix with respect to k'th component of the 'ith input point.

  if (this->ddfn_dx == NULL) {
    printf("ERROR: gradient not implemented for this distance function.\n");
    exit(1);
  }
  if (this->dwfn_dr == NULL) {
    printf("ERROR: gradient not implemented for this weight function.\n");
    exit(1);
  }

  pyublas::numpy_matrix<double> K(pts1.size1(), pts2.size1());
  for(unsigned i = 0; i < pts1.size1 (); ++ i) {
    for (unsigned j = 0; j < pts2.size1 (); ++ j) {
      K(i,j) = 0;
    }
  }

  point p1 = {&pts1(i, 0), 0};
  for (unsigned j = 0; j < pts2.size1 (); ++ j) {
    point p2 = {&pts2(j, 0), 0};
    double r = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    double dr_dp1 = this->ddfn_dx(p1.p, p2.p, k, r, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);

    /*
    double eps = 1e-8;
    double pp[3];
    pp[0] = p1.p[0];
    pp[1] = p1.p[1];
    pp[2] = p1.p[2];
    pp[k] += eps;
    point ppp = {(const double*)&pp, 0};

    double r2 = this->dfn(ppp, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);

    double empirical_dd = (r2-r)/eps;

    printf("i %d r %f r2 %f dd %f empirical dd %f\n", k, r, r2, dr_dp1, empirical_dd);

    if (dr_dp1 < -9999999) {
      exit(0);
    }    */

    K(i,j) = this->dwfn_dr(r, dr_dp1, this->wp);
  }

  return K;
}

void VectorTree::kernel_deriv_wrt_xi_row(const pyublas::numpy_matrix<double> &pts1, int i, int k, pyublas::numpy_vector<double> K) {
  // deriv of kernel matrix with respect to k'th component of the 'ith input point.
  // return just the row/col corresponding to the i'th input point (everything else should be zero)

  if (this->ddfn_dx == NULL) {
    printf("ERROR: gradient not implemented for this distance function.\n");
    exit(1);
  }
  if (this->dwfn_dr == NULL) {
    printf("ERROR: gradient not implemented for this weight function.\n");
    exit(1);
  }

  // pyublas::numpy_vector<double> K(pts1.size1());
  //for (unsigned j = 0; j < pts1.size1 (); ++ j) {
  //K(j) = 0;
  //}

  point p1 = {&pts1(i, 0), 0};
  for (unsigned j = 0; j < pts1.size1 (); ++ j) {
    point p2 = {&pts1(j, 0), 0};
    double r = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    double dr_dp1 = this->ddfn_dx(p1.p, p2.p, k, r, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    K(j) = this->dwfn_dr(r, dr_dp1, this->wp);
  }

}


pyublas::numpy_vector<double> VectorTree::sparse_kernel_deriv_wrt_xi(const pyublas::numpy_matrix<double> &pts1, int k, const pyublas::numpy_vector<int> &nzr, const pyublas::numpy_vector<int> &nzc, const pyublas::numpy_vector<double> distance_entries) {

  pyublas::numpy_vector<double> entries(nzr.size());
  if (this->ddfn_dtheta == NULL) {
    printf("ERROR: gradient not implemented for this distance function.\n");
    exit(1);
  }
  if (this->dwfn_dr == NULL) {
    printf("ERROR: gradient not implemented for this weight function.\n");
    exit(1);
  }

  for (unsigned i = 0; i < nzr.size(); ++ i) {
    point p1 = {&pts1(nzr[i], 0), 0};
    point p2 = {&pts1(nzc[i], 0), 0};

    // double r = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);

    double r = distance_entries[i];
    double dr_dp1 = this->ddfn_dx(p1.p, p2.p, k, r, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    entries[i] = this->dwfn_dr(r, dr_dp1, this->wp);
  }

  return entries;
}

void VectorTree::dist_deriv_wrt_xi_row(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, int i, int k, pyublas::numpy_vector<double> D) {
  // return just the row/col corresponding to the i'th input point (everything else should be zero)

  if (this->ddfn_dx == NULL) {
    printf("ERROR: gradient not implemented for this distance function.\n");
    exit(1);
  }
  point p1 = {&pts1(i, 0), 0};
  for (unsigned j = 0; j < pts2.size1 (); ++ j) {
    point p2 = {&pts2(j, 0), 0};
    double r = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    double dr_dp1 = this->ddfn_dx(p1.p, p2.p, k, r, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    D(j) = dr_dp1;
  }

}


pyublas::numpy_matrix<double> VectorTree::kernel_deriv_wrt_i(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, int param_i, bool symmetric, const pyublas::numpy_matrix<double> distances) {

  if (this->ddfn_dtheta == NULL) {
    printf("ERROR: gradient not implemented for this distance function.\n");
    exit(1);
  }
  if (this->dwfn_dr == NULL) {
    printf("ERROR: gradient not implemented for this weight function.\n");
    exit(1);
  }

 pyublas::numpy_matrix<double> K(pts1.size1(), pts2.size1());

 if (symmetric) {
   if (pts1.size1() != pts2.size1()) {
     printf("error: kernel_deriv_wrt_i called as symmetric with unequal dimensions %d, %d\n", pts1.size1(), pts2.size1());
     exit(-1);
   }
 }

  /*
  for(unsigned i = 0; i < pts1.size1 (); ++ i) {
    for (unsigned j = 0; j < pts2.size1 (); ++ j) {
      K(i,j) = 0;
    }
    }*/


  for (unsigned i = 0; i < pts1.size1 (); ++ i) {
    point p1 = {&pts1(i, 0), 0};

    int min_j = 0;
    if (symmetric) {
      min_j = i;
    }

    for (unsigned j = min_j; j < pts2.size1 (); ++ j) {
      point p2 = {&pts2(j, 0), 0};
      // double r = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
      double r = distances(i,j);
      double dr_dtheta = this->ddfn_dtheta(p1.p, p2.p, param_i, r, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
      //K(i,j) = kernels(i,j) * -2 * distances(i,j) * dr_dtheta;
      K(i,j) = this->dwfn_dr(r, dr_dtheta, this->wp);
    }
  }

  if (symmetric) {
    for (unsigned i = 0; i < pts1.size1 (); ++ i) {
      for (unsigned j = 0; j < i; ++ j) {
	K(i,j) = K(j,i);
      }
    }
  }

  return K;
}

pyublas::numpy_vector<double> VectorTree::sparse_distances(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, const pyublas::numpy_vector<int> &nzr, const pyublas::numpy_vector<int> &nzc) {
  pyublas::numpy_vector<double> entries(nzr.size());
  for (unsigned i = 0; i < nzr.size(); ++ i) {
    point p1 = {&pts1(nzr[i], 0), 0};
    point p2 = {&pts2(nzc[i], 0), 0};

    entries[i] = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
  }
  return entries;
}


pyublas::numpy_vector<double> VectorTree::sparse_kernel_deriv_wrt_i(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, const pyublas::numpy_vector<int> &nzr, const pyublas::numpy_vector<int> &nzc, int param_i, const pyublas::numpy_vector<double> distance_entries) {

  pyublas::numpy_vector<double> entries(nzr.size());
  if (this->ddfn_dtheta == NULL) {
    printf("ERROR: gradient not implemented for this distance function.\n");
    exit(1);
  }
  if (this->dwfn_dr == NULL) {
    printf("ERROR: gradient not implemented for this weight function.\n");
    exit(1);
  }

  for (unsigned i = 0; i < nzr.size(); ++ i) {
    point p1 = {&pts1(nzr[i], 0), 0};
    point p2 = {&pts2(nzc[i], 0), 0};

    //double r = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    double r = distance_entries[i];
    double dr_dtheta = this->ddfn_dtheta(p1.p, p2.p, param_i, r, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    entries[i] = this->dwfn_dr(r, dr_dtheta, this->wp);
  }

  return entries;
}


void VectorTree::set_Kinv_for_dense_hack(const pyublas::numpy_strided_vector<int> &nonzero_rows,
					 const pyublas::numpy_strided_vector<int> &nonzero_cols,
					 const pyublas::numpy_strided_vector<double> &nonzero_vals) {

  this->Kinv_for_dense_hack = new dense_hash_map<unsigned long, double>;
  this->Kinv_for_dense_hack->set_empty_key(this->n * this->n);

  dense_hash_map<unsigned long, double> &hmap = *(this->Kinv_for_dense_hack);

  for(unsigned int i=0; i < nonzero_rows.size(); ++i) {
    unsigned long key = (unsigned long)nonzero_rows[i] * this->n + nonzero_cols[i];
    hmap[key] = nonzero_vals[i];
  }
}

double VectorTree::quadratic_form_from_dense_hack(const pyublas::numpy_matrix<double> &query_pt1, const pyublas::numpy_matrix<double> &query_pt2, double max_distance) {

  dense_hash_map<unsigned long, double> &hmap = *(this->Kinv_for_dense_hack);

  // we want to find all points which are near *both* pt1 and pt2. for the moment, let's concentrate on when pt1 and pt2 are the same.

  point pt1 = {&query_pt1(0,0), 0};
  point pt2 = {&query_pt2(0,0), 0};

  if ((pt1.p[0] != pt2.p[0]) || (pt1.p[1] != pt2.p[1])) {
    printf("ERROR: quadratic_form_from_dense_hack is not yet implemented for off-diagonal covariances.\n");
    exit(1);
  }

  // find the training points near the query point
  v_array<v_array<point> > res;
  node<point> np1;
  np1.p = pt1;
  np1.max_dist = 0.;
  np1.parent_dist = 0.;
  np1.children = NULL;
  np1.num_children = 0;
  np1.scale = 100;

  this->dense_hack_dfn_evals = 0;
  this->dense_hack_wfn_evals = 0;
  this->dense_hack_terms = 0;

  struct timeval stop, start;
  gettimeofday(&start, NULL);
  //do stuff

  if (this->dfn_extra) {
    ((int *)this->dfn_extra)[1] = 0;
  }
  epsilon_nearest_neighbor(this->root,np1,res,max_distance, this->dfn, this->dist_params, this->dfn_extra);
  if (this->dfn_extra) {
    this->dense_hack_dfn_evals += ((int *)this->dfn_extra)[1];
  }

  gettimeofday(&stop, NULL);
  this->dense_hack_tree_s = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)/1000000.0;
  gettimeofday(&start, NULL);


  // compute the test-training kernel values, and evaluate the diagonal component of the quadratic form
  v_array<double> kstar;
  alloc(kstar, res[0].index);
  double qf = 0;

  for(int ii = 1; ii < res[0].index; ++ii) {
    point train_p1 = res[0][ii];
    int i = train_p1.idx;

    double d = this->dfn(pt1, train_p1, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    push(kstar, this->w(d, this->wp));

    this->dense_hack_dfn_evals += 1;
    this->dense_hack_wfn_evals += 1;

    unsigned long train_key = (unsigned long) i * this->n + i;
    double Kinv = hmap[train_key];
    qf += kstar[ii-1] * kstar[ii-1] * Kinv;
    this->dense_hack_terms++;
  }

  // compute the off-diagonal component of the quadratic form
  for(int ii = 1; ii < res[0].index; ++ii) {
    point train_p1 = res[0][ii];
    int i = train_p1.idx;

    for(int jj = ii+1; jj < res[0].index; ++jj) {
      point train_p2 = res[0][jj];
      int j = train_p2.idx;

      long train_key = (unsigned long) i * this->n + j;
      double Kinv = hmap[train_key];

      qf += kstar[ii-1] * kstar[jj-1] * Kinv * 2;
      this->dense_hack_terms++;
    }
  }

  gettimeofday(&stop, NULL);
  this->dense_hack_math_s =(stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)/1000000.0;
  return qf;
}


VectorTree::~VectorTree() {
  if (this->dist_params != NULL) {
    delete[] this->dist_params;
    this->dist_params = NULL;
  }
  if (this->dfn_extra != NULL) {
    free(this->dfn_extra);
    this->dfn_extra = NULL;
  }
  if (this->wp != NULL) {
    delete[] this->wp;
    this->wp = NULL;
  }
}

BOOST_PYTHON_MODULE(cover_tree) {
  bp::class_<VectorTree>("VectorTree", bp::init< pyublas::numpy_matrix< double > const &, int const, string const &, pyublas::numpy_vector< double > const &, string const &, pyublas::numpy_vector< double > const &>())
    .def("dump_tree", &VectorTree::dump_tree)
    .def("dump_clusters", &VectorTree::dump_clusters)
    .def("set_v", &VectorTree::set_v)
    .def("get_v", &VectorTree::get_v)
    .def("weighted_sum", &VectorTree::weighted_sum)
    .def("kernel_matrix", &VectorTree::kernel_matrix)
    .def("sparse_training_kernel_matrix", &VectorTree::sparse_training_kernel_matrix)
    .def("kernel_deriv_wrt_xi", &VectorTree::kernel_deriv_wrt_xi)
    .def("kernel_deriv_wrt_xi_row", &VectorTree::kernel_deriv_wrt_xi_row)
    .def("dist_deriv_wrt_xi_row", &VectorTree::dist_deriv_wrt_xi_row)
    .def("kernel_deriv_wrt_i", &VectorTree::kernel_deriv_wrt_i)
    .def("sparse_kernel_deriv_wrt_i", &VectorTree::sparse_kernel_deriv_wrt_i)
    .def("sparse_kernel_deriv_wrt_xi", &VectorTree::sparse_kernel_deriv_wrt_xi)
    .def("sparse_distances", &VectorTree::sparse_distances)
    .def("quadratic_form_from_dense_hack", &VectorTree::quadratic_form_from_dense_hack)
    .def("set_Kinv_for_dense_hack", &VectorTree::set_Kinv_for_dense_hack)
    .def_readonly("nodes_touched", &VectorTree::nodes_touched)
    .def_readonly("dfn_evals", &VectorTree::dfn_evals)
    .def_readonly("wfn_evals", &VectorTree::wfn_evals)
    .def_readonly("terms", &VectorTree::terms)
    .def_readonly("dense_hack_terms", &VectorTree::dense_hack_terms)
    .def_readonly("dense_hack_dfn_evals", &VectorTree::dense_hack_dfn_evals)
    .def_readonly("dense_hack_wfn_evals", &VectorTree::dense_hack_wfn_evals)
    .def_readonly("dense_hack_tree_s", &VectorTree::dense_hack_tree_s)
    .def_readonly("dense_hack_math_s", &VectorTree::dense_hack_math_s);

  bp::class_<MatrixTree>("MatrixTree", bp::init< pyublas::numpy_matrix< double > const &, pyublas::numpy_vector< int > const &, pyublas::numpy_vector< int > const &, string const &, pyublas::numpy_vector< double > const &, string const &, pyublas::numpy_vector< double > const &>())
    .def("collapse_leaf_bins", &MatrixTree::collapse_leaf_bins)
    .def("set_m", &MatrixTree::set_m)
    .def("set_m_sparse", &MatrixTree::set_m_sparse)
    .def("get_m", &MatrixTree::get_m)
    .def("quadratic_form", &MatrixTree::quadratic_form)
    .def("print_hierarchy", &MatrixTree::print_hierarchy)
    .def("test_bounds", &MatrixTree::test_bounds)
    .def("compile", &MatrixTree::compile)
    .def_readonly("nodes_touched", &MatrixTree::nodes_touched)
    .def_readonly("dfn_evals", &MatrixTree::dfn_evals)
    .def_readonly("wfn_evals", &MatrixTree::wfn_evals)
    .def_readonly("terms", &MatrixTree::terms)
    .def_readonly("zeroterms", &MatrixTree::zeroterms)
    .def_readonly("dfn_misses", &MatrixTree::dfn_misses)
    .def_readonly("wfn_misses", &MatrixTree::wfn_misses);

}
