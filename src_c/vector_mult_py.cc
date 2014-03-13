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


using namespace std;
namespace bp = boost::python;

double weighted_sum_node(node<point> &n, int v_select,
			 const point &query_pt, double eps,
			 double &weight_sofar,
			 int &fcalls,
			 wfn w,
			 distfn<point>::Type dist,
			 const double * dist_params,
			 void * dist_extra,
			 const double* weight_params) {
  double ws = 0;
  double d = n.distance_to_query; // avoid duplicate distance
				    // calculations by assuming this
				    // distance has already been
				    // computed by the parent, in the
				    // recursive expansion below. Note
				    // this calculation must be done
				    // explicitly at the root before
				    // this function is called.
  fcalls += 1;
  bool cutoff = false;

  // if (fabs(n.p.p[1]) < .0000001) {
    //}

  if (n.num_children == 0) {
    // if we're at a leaf, just do the multiplication

    double weight = w(d, weight_params);
    ws = weight * n.unweighted_sums[v_select];
    weight_sofar += weight;

    //printf("at leaf: ws = %lf*%lf = %lf\n", weight, n.unweighted_sums[v_select], ws);
    cutoff = true;
  } else {
    bool query_in_bounds = (d <= n.max_dist);
    if (!query_in_bounds) {
      double min_weight = w(d + n.max_dist, weight_params);
      double max_weight = w(max(0.0, d - n.max_dist), weight_params);
      double cutoff_threshold = 2 * eps * (weight_sofar + n.num_leaves * min_weight);
      cutoff = n.num_leaves * (max_weight - min_weight) <= cutoff_threshold;
      if (cutoff) {
	// if we're cutting off, just compute an estimate of the sum
	// in this region
	ws = .5 * (max_weight + min_weight) * n.unweighted_sums[v_select];
	//printf("cutting off: ws = %lf*%lf = %lf\n", .5 * (max_weight + min_weight), n.unweighted_sums[v_select], ws);
	weight_sofar += min_weight * n.num_leaves;
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
	permutation[i] = i;
      }
      halfsort(permutation, n.num_children, n.children);
      for(int i=0; i < n.num_children; ++i) {
	ws +=weighted_sum_node(n.children[permutation[i]], v_select,
			       query_pt, eps, weight_sofar, fcalls,
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
  } else {
    n.unweighted_sums[v_select] = 0;
    for(int i=0; i < n.num_children; ++i) {
      set_v_node(n.children[i], v_select, v);
      n.unweighted_sums[v_select] += n.children[i].unweighted_sums[v_select];
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
  int fcalls = 0;

  if (this->n == 0) {
    return 0;
  }


  this->root.distance_to_query = this->dfn(qp, this->root.p, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
  double ws = weighted_sum_node(this->root, v_select,
				qp, eps, weight_sofar,
				fcalls, this->w,
				this->dfn, this->dist_params,
				this->dfn_extra, this->wp);

  this->fcalls = fcalls;
  return ws;
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
    this->dfn_extra = malloc(sizeof(int));
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


pyublas::numpy_matrix<double> VectorTree::sparse_training_kernel_matrix(const pyublas::numpy_matrix<double> &pts, double max_distance) {

  pyublas::numpy_matrix<double> K(pts.size1()*2, 3);

  unsigned long nzero = 0;
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

    /*
    if( (p1.p[0] < -116.8) && (p1.p[0] > -116.9) && (p1.p[1] < - 74.4) && (p1.p[1] > -74.6) ) {
      printf("nn got %d results for %f, %f\n", res[0].index, p1.p[0], p1.p[1]);
      printf("dfn params %f %f\n", this->dist_params[0], this->dist_params[1]);
      printf("root is %p and has %d children\n", &this->root, this->root.num_children);
      }*/

    for(int jj = 1; jj < res[0].index; ++jj) {
      point p2 = res[0][jj];
      int j = p2.idx;

      double d = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
      if (nzero == K.size1()) {
	K.resize(K.size1()*2, 3);
      }
      K(nzero,0) = i;
      K(nzero,1) = j;
      K(nzero,2) = this->w(d, this->wp);
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


pyublas::numpy_matrix<double> VectorTree::kernel_deriv_wrt_i(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, int param_i) {

  if (this->ddfn_dtheta == NULL) {
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


  for (unsigned i = 0; i < pts1.size1 (); ++ i) {
    point p1 = {&pts1(i, 0), 0};
    for (unsigned j = 0; j < pts2.size1 (); ++ j) {
      point p2 = {&pts2(j, 0), 0};
      double r = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
      double dr_dtheta = this->ddfn_dtheta(p1.p, p2.p, param_i, r, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
      K(i,j) = this->dwfn_dr(r, dr_dtheta, this->wp);
    }
  }

  return K;
}

pyublas::numpy_vector<double> VectorTree::sparse_kernel_deriv_wrt_i(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, const pyublas::numpy_vector<int> &nzr, const pyublas::numpy_vector<int> &nzc, int param_i) {

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

    double r = this->dfn(p1, p2, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    double dr_dtheta = this->ddfn_dtheta(p1.p, p2.p, param_i, r, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra);
    entries[i] = this->dwfn_dr(r, dr_dtheta, this->wp);
  }

  return entries;
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
    .def("set_v", &VectorTree::set_v)
    .def("get_v", &VectorTree::get_v)
    .def("weighted_sum", &VectorTree::weighted_sum)
    .def("kernel_matrix", &VectorTree::kernel_matrix)
    .def("sparse_training_kernel_matrix", &VectorTree::sparse_training_kernel_matrix)
    .def("kernel_deriv_wrt_xi", &VectorTree::kernel_deriv_wrt_xi)
    .def("kernel_deriv_wrt_i", &VectorTree::kernel_deriv_wrt_i)
    .def("sparse_kernel_deriv_wrt_i", &VectorTree::sparse_kernel_deriv_wrt_i)
    .def_readonly("fcalls", &VectorTree::fcalls);

  bp::class_<MatrixTree>("MatrixTree", bp::init< pyublas::numpy_matrix< double > const &, pyublas::numpy_vector< int > const &, pyublas::numpy_vector< int > const &, string const &, pyublas::numpy_vector< double > const &, string const &, pyublas::numpy_vector< double > const &>())
    .def("collapse_leaf_bins", &MatrixTree::collapse_leaf_bins)
    .def("set_m", &MatrixTree::set_m)
    .def("set_m_sparse", &MatrixTree::set_m_sparse)
    .def("get_m", &MatrixTree::get_m)
    .def("quadratic_form", &MatrixTree::quadratic_form)
    .def("print_hierarchy", &MatrixTree::print_hierarchy)
    .def("test_bounds", &MatrixTree::test_bounds)
    .def_readonly("fcalls", &MatrixTree::fcalls)
    .def_readonly("dfn_evals", &MatrixTree::dfn_evals);

}
