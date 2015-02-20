 #include "cover_tree.hpp"
 #include "vector_mult.hpp"
 #include <cstdlib>
 #include <cmath>
 #include <memory>
 #include <vector>
 #include <limits>
 #include <google/dense_hash_map>
 using google::dense_hash_map;

 using namespace std;
 namespace bp = boost::python;

/*
 double gt(void)
 {
   struct timespec tv;

   if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;

   return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
 }
*/
double gt(void) {
  return 1000.0;
}

 double first_half_d_query_cached(const pairpoint &p1, const pairpoint &p2, double BOUND_IGNORED, const double *params, pair_dfn_extra *p) {
   dense_hash_map<int, double> &query_cache =  *(p->query1_cache);

   double d1;
   dense_hash_map<int, double>::iterator i = query_cache.find(p2.idx1);
   if (i == query_cache.end()) {
     d1 = p->dfn(p1.pt1, p2.pt1, BOUND_IGNORED, params, p->dfn_extra);
     query_cache[p2.idx1] = d1;
     p->misses += 1;
   } else {
     d1 = query_cache[p2.idx1];
     p->hits += 1;
   }

   return d1;
  }

  double second_half_d_query_cached(const pairpoint &p1, const pairpoint &p2, double BOUND_IGNORED, const double *params, pair_dfn_extra * p) {
   dense_hash_map<int, double> &query_cache =  *(p->query2_cache);

   double d2;
   dense_hash_map<int, double>::iterator i = query_cache.find(p2.idx2);
   if (i == query_cache.end()) {
     d2 = p->dfn(p1.pt2, p2.pt2, BOUND_IGNORED, params, p->dfn_extra);
     query_cache[p2.idx2] = d2;
     p->misses += 1;
   } else {
     d2 = query_cache[p2.idx2];
     p->hits += 1;
   }

   return d2;
  }

double first_half_w_query_cached(const pairpoint &p1, const pairpoint &p2, double BOUND_IGNORED, const double *params, pair_dfn_extra *p, wfn w_point, const double * wp_point) {
   dense_hash_map<int, double> &query_cache =  *(p->query1_w_cache);

   double d1, w;
   dense_hash_map<int, double>::iterator i = query_cache.find(p2.idx1);
   if (i == query_cache.end()) {
     d1 = first_half_d_query_cached(p1, p2, BOUND_IGNORED, params, p);
     w = w_point(d1, wp_point);
     query_cache[p2.idx1] = w;

     //printf ("set first half %d to w %f for d %f\n", p2.idx1, w, d1);

     p->w_misses += 1;
   } else {

     w = query_cache[p2.idx1];

     ///printf ("cache hit on first half idx %d! retrieved %f vs true value %f for d %f\n", p2.idx1, w, w1, d1);

     p->w_hits += 1;
   }

   return w;
  }

double second_half_w_query_cached(const pairpoint &p1, const pairpoint &p2, double BOUND_IGNORED, const double *params, pair_dfn_extra * p, wfn w_point, const double * wp_point) {
   dense_hash_map<int, double> &query_cache =  *(p->query2_w_cache);

   double d2, w;
   dense_hash_map<int, double>::iterator i = query_cache.find(p2.idx2);
   if (i == query_cache.end()) {
     d2 = second_half_d_query_cached(p1, p2, BOUND_IGNORED, params, p);
     w = w_point(d2, wp_point);
     query_cache[p2.idx2] = w;
     p->w_misses += 1;
   } else {

     w = query_cache[p2.idx2];

     // printf ("cache hit on second half %d! retrieved %f vs true value %f for d %f\n", p2.idx2, w, w2, d2);


     p->w_hits += 1;
   }

   return w;
  }


 double factored_query_distance_l2(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *params, void * extra) {
   // assume the dfn returns squared distances
   pair_dfn_extra * p = (pair_dfn_extra *) extra;
   double d1 = first_half_d_query_cached(p1, p2, BOUND_IGNORED, params, p);
   double d2 = second_half_d_query_cached(p1, p2, BOUND_IGNORED, params, p);

   //printf("d1 = %.3f = d((%.3f, %.3f), (%.3f, %.3f))\n", d1, p1.pt1[0], p1.pt1[0], p2.pt1[0], p2.pt1[1]);
   //printf("d2 = %.3f = d((%.3f, %.3f), (%.3f, %.3f))\n", d2, p1.pt2[0], p1.pt2[0], p2.pt2[0], p2.pt2[1]);

   return sqrt(d1 + d2);
 }

 double factored_query_distance_l1(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *params, void * extra) {
   // assume the dfn returns actual (non-squared) distances
   pair_dfn_extra * p = (pair_dfn_extra *) extra;
   double d1 = first_half_d_query_cached(p1, p2, BOUND_IGNORED, params, p);
   double d2 = second_half_d_query_cached(p1, p2, BOUND_IGNORED, params, p);
   return d1 + d2;
 }

 double first_half_d_build_cached(const pairpoint &p1, const pairpoint &p2, double BOUND_IGNORED, const double *params, pair_dfn_extra * p) {

   dense_hash_map<long, double> &distance_cache = *(p->build_cache);
   const long NPTS = (const long) p->NPTS;

   double d1;
   long pair1_idx = p1.idx1 * NPTS + (long) p2.idx1;

   dense_hash_map<long, double>::iterator i = distance_cache.find(pair1_idx);
   if (i == distance_cache.end()) {
     d1 = p->dfn(p1.pt1, p2.pt1, BOUND_IGNORED, params, p->dfn_extra);
     distance_cache[pair1_idx] = d1;
     p->misses += 1;
   } else {
     d1 = distance_cache[pair1_idx];
     p->hits += 1;
   }

   return d1;
 }

 double second_half_d_build_cached(const pairpoint &p1, const pairpoint &p2, double BOUND_IGNORED, const double *params, pair_dfn_extra * p) {

   dense_hash_map<long, double> &distance_cache = *(p->build_cache);
   const long NPTS = (const long) p->NPTS;

   double d2;
   long pair2_idx = p1.idx2 * NPTS + (long) p2.idx2;

   dense_hash_map<long, double>::iterator i = distance_cache.find(pair2_idx);
   if (i == distance_cache.end()) {
     d2 = p->dfn(p1.pt2, p2.pt2, BOUND_IGNORED, params, p->dfn_extra);
     distance_cache[pair2_idx] = d2;
     p->misses += 1;
   } else {
     d2 = distance_cache[pair2_idx];
     p->hits += 1;
   }

   return d2;
 }

 double factored_build_distance_l2(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *params, void * extra) {
   // assume the dfn returns squared distances
   pair_dfn_extra * p = (pair_dfn_extra *) extra;

   double d1 = first_half_d_build_cached(p1, p2, BOUND_IGNORED, params, p);
   double d2 = second_half_d_build_cached(p1, p2, BOUND_IGNORED, params, p);
   return sqrt(d1 + d2);
 }
 double factored_build_distance_l1(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *params, void * extra) {
   // assume the dfn returns actaul (non-squared) distances
   pair_dfn_extra * p = (pair_dfn_extra *) extra;
   double d1 = first_half_d_build_cached(p1, p2, BOUND_IGNORED, params, p);
   double d2 = second_half_d_build_cached(p1, p2, BOUND_IGNORED, params, p);
   return d1 + d2;
 }

 void weighted_sum_node(node<pairpoint> &n, int v_select,
			const pairpoint &query_pt,
			double eps_rel,
			double eps_abs,
			int cutoff_rule, // 0 = Shen
			                 // 1 = UAI/MSTND
			                 // 2 = provable_abs
			double &weight_sofar,
			int &terms_sofar,
			double &abserr_sofar,
			double &ws,
			int max_terms,
			int &nodes_touched,
			int &terms,
			int &zeroterms,
			int &dfn_evals,
			int &wfn_evals,
			wfn w_upper,
			wfn w_lower,
			wfn w_point,
			const double* wp_pair,
			const double* wp_point,
			distfn<pairpoint>::Type dist,
			const double * dist_params,
			pair_dfn_extra * dist_extra,
			bool HACK_adj_offdiag) {
   double d = n.distance_to_query; // avoid duplicate distance
				     // calculations by assuming this
				     // distance has already been
				     // computed by the parent, in the
				     // recursive expansion below. Note
				     // this calculation must be done
				     // explicitly at the root before
				     // this function is called.
   nodes_touched += 1;
   if (n.num_children == 0 && n.n_extra_p == 0) {
     // if we're at a leaf, just do the multiplication

     double weight;
     if (w_upper == w_lower) {
       // if we can compute an exact kernel value from the product distance, do so
       weight = w_upper(d, wp_pair);
       wfn_evals += 1;
     } else {
       // otherwise, compute the product kernel explicitly.  note that
       // w_point takes an *intermediate* representation of the
       // distance, e.g. squared distance in the case of the SE
       // kernels.

       weight = first_half_w_query_cached(query_pt, n.p, std::numeric_limits< double >::max(), dist_params, dist_extra, w_point, wp_point)
	 * second_half_w_query_cached(query_pt, n.p, std::numeric_limits< double >::max(), dist_params, dist_extra, w_point, wp_point);
       if (HACK_adj_offdiag) weight *= 2;

       //       weight = w_point(first_half_d_query_cached(query_pt, n.p, std::numeric_limits< double >::max(), dist_params, dist_extra), wp_point)
       //             * w_point(second_half_d_query_cached(query_pt, n.p, std::numeric_limits< double >::max(), dist_params, dist_extra), wp_point);

     }
     ws += weight * n.unweighted_sums[v_select];

     if (weight == 0 || n.unweighted_sums[v_select] == 0) {
       zeroterms += 1;
     } else {
	   terms += 1;
     }

     switch (cutoff_rule) {
     case 0:
       weight_sofar += weight;
       break;
     case 1:
       break;
     case 2:
       terms_sofar += 1;
       break;
     }
     //printf("at leaf: ws += %lf*%lf = %lf\n", weight, n.unweighted_sums[v_select], ws);
     //printf("idx (%d, %d) pt1 (%.4f, %.4f) pt2 (%.4f, %.4f) Kinv=%.4f Kinv_abs=%.4f weight=%.4f ws=%.4f wSoFar=%.4f dist %.4f\n", n.p.idx1, n.p.idx2, n.p.pt1[0], n.p.pt1[1], n.p.pt2[0], n.p.pt2[1], n.unweighted_sums[v_select], n.unweighted_sums_abs[v_select], weight, ws, weight_sofar, d);

     return;
   }
   if (n.n_extra_p > 0) {
     double * epvals = n.extra_p_vals[v_select];
     //printf("computing exact sum of %d additional pts\n", n.n_extra_p);

     double exact_sum = 0;
     for (unsigned int i=0; i < n.n_extra_p; ++i) {
       /*       double weight = w_point(first_half_d_query_cached(query_pt, n.extra_p[i], std::numeric_limits< double >::max(), dist_params, dist_extra), wp_point)
	* w_point(second_half_d_query_cached(query_pt, n.extra_p[i], std::numeric_limits< double >::max(), dist_params, dist_extra), wp_point);*/

       double weight = first_half_w_query_cached(query_pt, n.extra_p[i], std::numeric_limits< double >::max(), dist_params, dist_extra, w_point, wp_point)
	 * second_half_w_query_cached(query_pt, n.extra_p[i], std::numeric_limits< double >::max(), dist_params, dist_extra, w_point, wp_point);

       // a little confused about what's actually going on here...
       wfn_evals += 2;
       dfn_evals += 2;

       if (HACK_adj_offdiag) weight *= 2;

       ws += weight * epvals[i];

       if (weight == 0 || epvals[i] == 0) {
	 zeroterms += 1;
       } else {
	   terms += 1;
       }


       switch (cutoff_rule) {
       case 0:
	 weight_sofar += weight;
	 break;
       case 1:
	 break;
       case 2:
	 terms_sofar += 1;
	 break;
       }

     }

     return;
   }

   bool query_in_bounds = (d <= n.max_dist);
   bool cutoff = false;
   double min_weight, max_weight, threshold;
   min_weight = -999;
   max_weight = -999;
   threshold = -999;
   if (!query_in_bounds) {
     min_weight = w_lower(d + n.max_dist, wp_pair);
     max_weight = w_upper(max(0.0, d - n.max_dist), wp_pair);
     wfn_evals += 2;
     //
     double frac_remaining_terms, abserr_n;
     switch (cutoff_rule) {
     case 0:
       threshold = 2 * eps_rel * (weight_sofar + n.num_leaves * min_weight);
       cutoff = n.num_leaves * (max_weight - min_weight) <= threshold;
       if (cutoff) {
	 ws += .5 * (max_weight + min_weight) * n.unweighted_sums[v_select];
	 weight_sofar += min_weight * n.num_leaves;
	 if ( (.5 * (max_weight + min_weight) * n.unweighted_sums[v_select]) == 0) {
	   zeroterms += 1;
	 } else {
	   terms += 1;
	 }

       }
       break;
     case 1:
       threshold = (eps_rel * fabs(ws + n.unweighted_sums[v_select] * min_weight) + eps_abs);
       cutoff = max_weight * n.unweighted_sums_abs[v_select] < threshold;
       if (cutoff) {
	 ws += .5 * (max_weight + min_weight) * n.unweighted_sums[v_select];
	 if ( (.5 * (max_weight + min_weight) * n.unweighted_sums[v_select]) == 0) {
	   zeroterms += 1;
	 } else {
	   terms += 1;
	 }

       }
       break;
     case 2:
       frac_remaining_terms = n.num_leaves / (double)(max_terms - terms_sofar);
       threshold = frac_remaining_terms * (eps_abs - abserr_sofar);
       abserr_n = .5 * (max_weight - min_weight) * n.unweighted_sums_abs[v_select];
       cutoff = abserr_n < threshold;
       if (cutoff) {
	 ws += .5 * (max_weight + min_weight) * n.unweighted_sums[v_select];
	 //printf("cutting off: %d leaves (representing %.1f%% of %d-%d remaining), error bound %.8f, error budget %.4f - %.4f = %.4f, so we would have been allowed %.6f\n", n.num_leaves, frac_remaining_terms*100, max_terms, terms_sofar, abserr_n, eps_abs, abserr_sofar, eps_abs - abserr_sofar, threshold);
	 terms_sofar += n.num_leaves;
	 abserr_sofar += abserr_n;
	 if ((.5 * (max_weight + min_weight) * n.unweighted_sums[v_select]) == 0) {
	   zeroterms +=1 ;
	 } else {
	   terms += 1;
	 }
       }

       break;
     }
       // if we're cutting off, just compute an estimate of the sum
       // in this region
       //printf("cutting off: ws = %lf*%lf = %lf\n", .5 * (max_weight + min_weight), n.unweighted_sums[v_select], ws);
       //printf("idx (%d, %d) pt1 (%.4f, %.4f) pt2 (%.4f, %.4f) Kinv=%.4f Kinv_abs=%.4f weight=%.4f ws=%.4f wSoFar=%.4f dist %.4f   (approx %d children min %.4f max %.4f max_contrib %f prod_thresh %f)\n", n.p.idx1, n.p.idx2, n.p.pt1[0], n.p.pt1[1], n.p.pt2[0], n.p.pt2[1], n.unweighted_sums[v_select], n.unweighted_sums_abs[v_select], .5 * (max_weight + min_weight), ws, weight_sofar, d, n.num_leaves, min_weight, max_weight,  max_weight * n.unweighted_sums_abs[v_select], threshold);

   }
   if (!cutoff) {
     // if not cutting off, we expand the sum recursively at the
     // children of this node, from nearest to furthest.
     //printf("NO CUTOFF AT idx (%d, %d) pt1 (%.4f, %.4f) pt2 (%.4f, %.4f) Kinv=%.4f Kinv_abs=%.4f dist %.4f (approx %d children min %.4f max %.4f cutoff %f thresh %f), recursing to \n", n.p.idx1, n.p.idx2, n.p.pt1[0], n.p.pt1[1], n.p.pt2[0], n.p.pt2[1], n.unweighted_sums[v_select], n.unweighted_sums_abs[v_select], d, n.num_leaves, min_weight, max_weight, max_weight * n.unweighted_sums_abs[v_select], threshold);

     int small_perm[10];
     int * permutation = (int *)&small_perm;
     if(n.num_children > 10) {
       permutation = (int *)malloc(n.num_children * sizeof(int));
     }

     for(int i=0; i < n.num_children; ++i) {
       n.children[i].distance_to_query = dist(query_pt, n.children[i].p, std::numeric_limits< double >::max(), dist_params, dist_extra);
       dfn_evals += 2;
       permutation[i] = i;
       //printf("%.4f ", n.children[i].distance_to_query);
     }

     halfsort(permutation, n.num_children, n.children);

     for(int i=0; i < n.num_children; ++i) {
       weighted_sum_node(n.children[permutation[i]], v_select,
			 query_pt, eps_rel, eps_abs, cutoff_rule,
			 weight_sofar, terms_sofar, abserr_sofar,
			 ws, max_terms, nodes_touched, terms, zeroterms, dfn_evals, wfn_evals,
			 w_upper, w_lower, w_point, wp_pair, wp_point,
			 dist, dist_params, dist_extra, HACK_adj_offdiag);
     }

     if (permutation != (int *)&small_perm) {
       free(permutation);
     }

   }

 }


 void set_m_node (node<pairpoint> &n, const pyublas::numpy_matrix<double> &m) {
   if (n.num_children == 0) {
     n.unweighted_sum = m(n.p.idx1, n.p.idx2);
     n.unweighted_sum_abs = fabs(m(n.p.idx1, n.p.idx2));
   } else {
     n.unweighted_sum = 0;
     n.unweighted_sum_abs = 0;
     for(int i=0; i < n.num_children; ++i) {
       set_m_node(n.children[i], m);
       n.unweighted_sum += n.children[i].unweighted_sum;
       n.unweighted_sum_abs += n.children[i].unweighted_sum_abs;
     }
   }
 }

 void set_m_node (node<pairpoint> &n, dense_hash_map<unsigned long, double> &sparse_m, unsigned long np) {
   unsigned long key = (unsigned long)n.p.idx1 * np + n.p.idx2;
   if (n.num_children == 0) {
     n.unweighted_sum = sparse_m[key];
     n.unweighted_sum_abs = fabs(sparse_m[key]);
   } else {
     n.unweighted_sum = 0;
     n.unweighted_sum_abs = 0;
     for(int i=0; i < n.num_children; ++i) {
       set_m_node(n.children[i], sparse_m, np);
       n.unweighted_sum += n.children[i].unweighted_sum;
       n.unweighted_sum_abs += n.children[i].unweighted_sum_abs;
     }
   }
 }


 void get_m_node(node<pairpoint> &n, pyublas::numpy_matrix<double> &m) {
   if (n.num_children == 0) {
     //printf("%du %du = %lf\n" n.p.idx1, n.p.idx2, n.unweighted_sum);
     m(n.p.idx1, n.p.idx2) = n.unweighted_sum;
   } else {
     for(int i=0; i < n.num_children; ++i) {
       get_m_node(n.children[i], m);
     }
   }
 }


 void MatrixTree::print_hierarchy(const pyublas::numpy_matrix<double> &query_pt1, const pyublas::numpy_matrix<double> &query_pt2) {

   pairpoint qp = {&query_pt1(0,0), &query_pt2(0,0), 0, 0};
   node<pairpoint> np;
   np.p = qp;
   np.max_dist = 0.;
   np.parent_dist = 0.;
   np.children = NULL;
   np.num_children = 0;
   np.scale = 100;

   v_array<v_array<node<pairpoint> > > res;

   k_nearest_neighbor(this->root_offdiag, np,res, 1, this->raw_pair_dfn, this->dist_params, this->dfn_extra->dfn_extra);
   node<pairpoint> *orig_n = &(res[0][1]);
   node<pairpoint> *n = orig_n;
   bool printed_root = false;
   while (n != NULL) {
     printf("pidx (%d, %d) t1 (%.4f %.4f) pt2 (%.4f %.4f) dist_to_p %.4f max_dist %.4f parent_dist %.4f num_children %d num_leaves %d scale %d\n", n->p.idx1, n->p.idx2, n->p.pt1[0], n->p.pt1[1], n->p.pt2[0], n->p.pt2[1], this->raw_pair_dfn(n->p, orig_n->p, std::numeric_limits< double >::max(), this->dist_params, this->dfn_extra->dfn_extra), n->max_dist, n->parent_dist, n->num_children, n->num_leaves, n->scale);

     n = n->debug_parent;
   }
 }


void collect_leaves(node<pairpoint> & n) {

  // if we're at a leaf, just fill in extra_p with the point/values at this node
  if (n.num_children == 0) {
    n.n_extra_p = 1;
    n.extra_p = new pairpoint[n.n_extra_p];

    n.extra_p[0] = n.p;

    n.extra_p_vals = (double **)malloc(n.narms * sizeof(double *));
    for (unsigned int a=0; a < n.narms; ++a) {
      n.extra_p_vals[a] = new double[n.n_extra_p];
      n.extra_p_vals[a][0] = n.unweighted_sums[a];
    }

  } else {
    n.n_extra_p = n.num_leaves;
    n.extra_p = new pairpoint[n.n_extra_p];
    n.extra_p_vals = (double **)malloc(n.narms * sizeof(double *));
    for (unsigned int a=0; a < n.narms; ++a) {
      n.extra_p_vals[a] = new double[n.n_extra_p];
    }

    // otherwise, recurse, then collect from immediate children
    int leaves_collected = 0;
    for (unsigned int i=0; i < n.num_children; ++i) {
      collect_leaves(n.children[i]);
      for (unsigned int j=0; j < n.children[i].n_extra_p; ++j) {
	n.extra_p[leaves_collected] = n.children[i].extra_p[j];
	for (unsigned int a = 0; a < n.narms; ++a) {
	  n.extra_p_vals[a][leaves_collected] = n.children[i].extra_p_vals[a][j];
	}
	leaves_collected++;
      }
    }
  }


  double unweighted_sum = 0;
  for (unsigned int i=0; i < n.n_extra_p;i++) {
    unweighted_sum += n.extra_p_vals[0][i];
  }
}


void cutoff_leaves(node<pairpoint> &root, double leaf_bin_width ) {

  if ((root.num_leaves == root.num_children) || (root.max_dist <= leaf_bin_width)) {
    collect_leaves(root);
    //root.free_tree_recursive();
    root.num_children = 0;
    root.children = NULL; // warning: GIANT MEMORY LEAK
  } else {
    for (unsigned int i=0; i < root.num_children; ++i) {
      cutoff_leaves(root.children[i], leaf_bin_width);
    }
  }

}


double MatrixTree::quadratic_form(const pyublas::numpy_matrix<double> &query_pt1, const pyublas::numpy_matrix<double> &query_pt2, double eps_rel, double eps_abs, int cutoff_rule) {
   pairpoint qp = {&query_pt1(0,0), &query_pt2(0,0), 0, 0};
   bool symmetric = (qp.pt1 == qp.pt2);

   pair_dfn_extra * p = (pair_dfn_extra *) this->dfn_extra;
   p->query1_cache = new dense_hash_map<int, double>((int) (10 * log(this->n)));
   p->query1_cache->set_empty_key(-1);
   if (symmetric) {
     p->query2_cache = p->query1_cache;
   } else {
     p->query2_cache = new dense_hash_map<int, double>((int) (10 * log(this->n)));
     p->query2_cache->set_empty_key(-1);
   }
   p->hits = 0;
   p->misses = 0;


   p->query1_w_cache = new dense_hash_map<int, double>((int) (10 * log(this->n)));
   p->query1_w_cache->set_empty_key(-1);
   if (symmetric) {
     p->query2_w_cache = p->query1_w_cache;
   } else {
     p->query2_w_cache = new dense_hash_map<int, double>((int) (10 * log(this->n)));
     p->query2_w_cache->set_empty_key(-1);
   }
   p->w_hits = 0;
   p->w_misses = 0;

   // assume there will be at least one point within three or so lengthscales,
   // so we can cut off any branch with really neligible weight.
   double weight_sofar = 0;
   int terms_sofar = 0;
   double abserr_sofar = 0;
   double ws = 0;

   this->nodes_touched = 0;
   this->terms = 0;
   this->zeroterms = 0;
   this->dfn_evals = 2;
   this->wfn_evals = 0;

   this->root_diag.distance_to_query = this->factored_query_dist(qp, this->root_diag.p, std::numeric_limits< double >::max(), this->dist_params, (void*)this->dfn_extra);
   if (this->use_offdiag) {
     this->root_offdiag.distance_to_query = this->factored_query_dist(qp, this->root_offdiag.p, std::numeric_limits< double >::max(), this->dist_params, (void*)this->dfn_extra);
   }

   if (symmetric) {
     int max_terms = this->root_diag.num_leaves + this->root_offdiag.num_leaves;
    weighted_sum_node(this->root_diag, 0,
		      qp, eps_rel, eps_abs, cutoff_rule,
		      weight_sofar, terms_sofar, abserr_sofar,
		      ws, max_terms,
		      this->nodes_touched, this->terms, this->zeroterms, this->dfn_evals, this->wfn_evals, this->w_upper,
			   this->w_lower, this->w_point,
			   this->wp_pair, this->wp_point,
			   this->factored_query_dist,
		      this->dist_params, this->dfn_extra, false);

    if (this->use_offdiag) {
      //this->wp_pair[0] *= 2;
      //this->wp_point[0] *= sqrt(2.0);
      weighted_sum_node(this->root_offdiag, 0,
			qp, eps_rel, eps_abs, cutoff_rule,
			weight_sofar, terms_sofar, abserr_sofar,
			ws, max_terms,
			this->nodes_touched, this->terms, this->zeroterms, this->dfn_evals, this->wfn_evals, this->w_upper,
			this->w_lower, this->w_point,
			this->wp_pair, this->wp_point,
			this->factored_query_dist,
			this->dist_params, this->dfn_extra, true);
      //this->wp_pair[0] /= 2;
      //this->wp_point[0] /= sqrt(2.0);
    }
   } else{
     int max_terms = this->nzero;
     if (this->use_offdiag) {
     weighted_sum_node(this->root_offdiag, 0,
		       qp, eps_rel, eps_abs, cutoff_rule,
		       weight_sofar, terms_sofar, abserr_sofar,
		       ws, max_terms,
		       this->nodes_touched, this->terms, this->zeroterms, this->dfn_evals, this->wfn_evals, this->w_upper,
		       this->w_lower, this->w_point,
		       this->wp_pair, this->wp_point,
		       this->factored_query_dist,
		       this->dist_params, this->dfn_extra, false);
     }

     weighted_sum_node(this->root_diag, 0,
		       qp, eps_rel, eps_abs, cutoff_rule,
		       weight_sofar, terms_sofar, abserr_sofar,
		       ws, max_terms,
		       this->nodes_touched, this->terms, this->zeroterms, this->dfn_evals, this->wfn_evals, this->w_upper,
		       this->w_lower, this->w_point,
		       this->wp_pair, this->wp_point,
		       this->factored_query_dist,
		       this->dist_params, this->dfn_extra, false);

     if (this->use_offdiag) {
     pairpoint qp2 = {&query_pt2(0,0), &query_pt1(0,0), 0, 0};
     weighted_sum_node(this->root_offdiag, 0,
		       qp2, eps_rel, eps_abs, cutoff_rule,
		       weight_sofar, terms_sofar, abserr_sofar,
		       ws, max_terms,
		       this->nodes_touched, this->terms, this->zeroterms, this->dfn_evals, this->wfn_evals, this->w_upper,
		       this->w_lower, this->w_point,
		       this->wp_pair, this->wp_point,
		       this->factored_query_dist,
		       this->dist_params, this->dfn_extra, false);
     }
   }



   this->dfn_misses = ((pair_dfn_extra *)this->dfn_extra)->misses;
   this->wfn_misses = ((pair_dfn_extra *)this->dfn_extra)->w_misses;
   //printf("quadratic form did %.0lf distance calculations for %d fcalls\n", ((double *)(this->distance_cache))[0], this->fcalls);

   delete p->query1_cache;
   delete p->query1_w_cache;
   if (!symmetric) {
     delete p->query2_cache;
     delete p->query2_w_cache;
   }

   return ws;
 }

 void MatrixTree::set_m_sparse(const pyublas::numpy_strided_vector<int> &nonzero_rows,
			       const pyublas::numpy_strided_vector<int> &nonzero_cols,
			       const pyublas::numpy_strided_vector<double> &nonzero_vals) {
   dense_hash_map<unsigned long, double> sparse_m;
   sparse_m.set_empty_key(this->n * this->n);
   for(unsigned int i=0; i < nonzero_rows.size(); ++i) {
     unsigned long key = (unsigned long)nonzero_rows[i] * this->n + nonzero_cols[i];
     sparse_m[key] = nonzero_vals[i];
   }
   set_m_node(this->root_diag, sparse_m, (unsigned long)this->n);
   if (this->use_offdiag) {
     set_m_node(this->root_offdiag, sparse_m, (unsigned long)this->n);
   }
 }

 void MatrixTree::set_m(const pyublas::numpy_matrix<double> &m) {
   if (m.size1() != m.size2()) {
     printf("error: matrixtree can only hold square matrices! (matrix passed has dimensions %lu x %lu)\n", m.size1(), m.size2());
     exit(1);
   }
   set_m_node(this->root_diag, m);

   if (this->use_offdiag) {
     set_m_node(this->root_offdiag, m);
   }
 }

void MatrixTree::collapse_leaf_bins(double leaf_bin_width) {
  cutoff_leaves(this->root_diag, leaf_bin_width);
  if (this->use_offdiag) {
    cutoff_leaves(this->root_offdiag, leaf_bin_width);
  }
}

 pyublas::numpy_matrix<double> MatrixTree::get_m() {
   vector<double> v(this->n * this->n);
   pyublas::numpy_matrix<double> pm(this->n, this->n);
   get_m_node(this->root_diag, pm);
   if (this->use_offdiag) {
     get_m_node(this->root_offdiag, pm);
   }
   return pm;
 }


 MatrixTree::MatrixTree (const pyublas::numpy_matrix<double> &pts,
			 const pyublas::numpy_strided_vector<int> &nonzero_rows,
			 const pyublas::numpy_strided_vector<int> &nonzero_cols,
			 const string &distfn_str,
			 const pyublas::numpy_vector<double> &dist_params,
			 const string wfn_str,
			 const pyublas::numpy_vector<double> &weight_params) {
   unsigned int nzero = nonzero_rows.size();
   vector< pairpoint > pairs_offdiag;
   vector< pairpoint > pairs_diag;
   unsigned int n = pts.size1();
   for(unsigned int i=0; i < nzero; ++i) {
     int r = nonzero_rows(i);
     int c = nonzero_cols(i);
     pairpoint p;
     p.pt1 = &pts(r, 0);
     p.pt2 = &pts(c, 0);
         p.idx1 = r;
    p.idx2 = c;
    if (r == c) {
      pairs_diag.push_back(p);
    } else if (r < c) {
      pairs_offdiag.push_back(p);
    }
  }
  this->n = n;
  this->nzero = nzero;

  pair_dfn_extra * p = new pair_dfn_extra;
  p->dfn_extra = NULL;
  if (distfn_str.compare("lld") == 0) {
    p->dfn_orig = dist_3d_km;
    p->dfn_sq = distsq_3d_km;
    this->raw_pair_dfn = pair_dist_3d_km;
  } else if (distfn_str.compare("euclidean") == 0) {
    p->dfn_orig = dist_euclidean;
    p->dfn_sq = sqdist_euclidean;
    p->dfn_extra = malloc(sizeof(int));
    *((int *) p->dfn_extra) = pts.size2();
    this->raw_pair_dfn = pair_dist_euclidean;
  } else if (distfn_str.compare("lldlld") == 0) {
    p->dfn_orig = dist_6d_km;
    p->dfn_sq = distsq_6d_km;
    this->raw_pair_dfn = pair_dist_6d_km;
  }else{
    printf("error: unrecognized distance function %s\n", distfn_str.c_str());
    exit(1);
  }

  p->build_cache = new dense_hash_map<long, double>(nzero);
  p->build_cache->set_empty_key(-1);
  p->hits = 0;
  p->misses = 0;
  p->NPTS = n;

  this->dfn_extra = p;
  this->dist_params = NULL;
  this->set_dist_params(dist_params);



  /* GIANT HACK: we're assuming the weight function has exactly one param,
     which functions as a leading coefficient.
   */
  this->wp_point = NULL;
  this->wp_pair = NULL;

  int n_wp = 0;
  n_wp += weight_params.size();
  if (wfn_str.compare(0, 7, "compact") == 0) {
    n_wp += 1;
  }
  if (n_wp > 0) {
    this->wp_point = new double[n_wp];
    this->wp_pair = new double[n_wp];

    this->wp_pair[0] = weight_params(0)*weight_params(0);
    this->wp_point[0] = weight_params(0);
    for (unsigned i = 1; i < weight_params.size(); ++i) {
      printf("WARNING: weight function has multiple params, violating our hack assumption. HERE BE DRAGONS.\n");
      this->wp_point[i] = weight_params[i];
      this->wp_pair[i] = weight_params[i];
    }

    if (wfn_str.compare(0, 7, "compact") == 0) {
      int D = pts.size2();
      int q = atoi(wfn_str.c_str()+7);
      double j = floor(D/2) + q+ 1.0;
      //printf("compact weight, D=%d, q=%d, j=%f\n", D, q, j);
      this->wp_point[n_wp-1] = j;
      this->wp_pair[n_wp-1] = j;
    }
  }

  if (wfn_str.compare("se") == 0) {
    this->w_point = w_se;
    this->w_upper = w_se;
    this->w_lower = w_se;

    // max weight any point can have
    this->max_weight = this->wp_pair[0];

    p->dfn = p->dfn_sq;
    this->factored_build_dist = factored_build_distance_l2;
    this->factored_query_dist = factored_query_distance_l2;
  } else  if (wfn_str.compare("matern32") == 0) {
    this->w_point = w_matern32;
    this->w_upper = w_matern32_upper;
    this->w_lower = w_matern32_lower;

    p->dfn = p->dfn_orig;
    this->factored_build_dist = factored_build_distance_l1;
    this->factored_query_dist = factored_query_distance_l1;
  } else  if (wfn_str.compare("compact0") == 0) {
    this->w_point = w_compact_q0;
    this->w_upper = w_compact_q0_upper;
    this->w_lower = w_compact_q0_lower;

    p->dfn = p->dfn_orig;
    this->factored_build_dist = factored_build_distance_l1;
    this->factored_query_dist = factored_query_distance_l1;
  } else  if (wfn_str.compare("compact2") == 0) {
    this->w_point = w_compact_q2;
    this->w_upper = w_compact_q2_upper;
    this->w_lower = w_compact_q2_lower;

    p->dfn = p->dfn_orig;
    this->factored_build_dist = factored_build_distance_l1;
    this->factored_query_dist = factored_query_distance_l1;
  } else{
     printf("error: unrecognized weight function %s\n", wfn_str.c_str());
    exit(1);
  }

  double t0 = gt();
  node<pairpoint> * a = NULL;
  if (pairs_diag.size() == 0) {
    printf("ERROR: no nonzero covariance entries on the diagonal! something is wildly wrong.\n");
    exit(1);
  } else {
    this->root_diag = batch_create(pairs_diag, this->factored_build_dist, this->dist_params, this->dfn_extra);
    set_leaves(this->root_diag, a);
    this->root_diag.alloc_arms(1);
  }
  if (pairs_offdiag.size() == 0) {
    printf("WARNING: the covariance matrix is entirely diagonal! This is probably not what you want, and will cause errors later.\n");
    //exit(1);
    this->use_offdiag = 0;
  } else {
    this->root_offdiag = batch_create(pairs_offdiag, this->factored_build_dist, this->dist_params, this->dfn_extra);
    set_leaves(this->root_offdiag, a);
    this->root_offdiag.alloc_arms(1);
    this->use_offdiag = 1;
  }

  double t1 = gt();
  // printf("built tree in %lfs: %d cache hits and %d cache misses\n", t1-t0, p->hits, p->misses);
  delete p->build_cache;
}

void MatrixTree::set_dist_params(const pyublas::numpy_vector<double> &dist_params) {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
  this->dist_params = new double[dist_params.size()+1];
  for (unsigned i = 0; i < dist_params.size(); ++i) {
    this->dist_params[i] = dist_params(i);
  }
  // HACK ALERT
  this->dist_params[dist_params.size()] = (double) this->n;
}

void MatrixTree::test_bounds(double max_d, int n_d) {

  int n = 0;
  double lgap_sum = 0;
  double ugap_sum = 0;
  double gap_percent = 0;
  double k_percent = 0;
  for (double d1 = 0; d1 <= max_d; d1 += max_d/n_d) {
    for (double d2 = 0; d2 <= max_d; d2 += max_d/n_d) {
	  double delta = d1 + d2;

	  double true_kernel = this->w_point(d1, this->wp_point) * this->w_point(d2, this->wp_point);
	  double lbound = this->w_lower(delta, this->wp_pair);
	  double ubound = this->w_upper(delta, this->wp_pair);

	  double lgap = true_kernel - lbound;
	  double ugap = ubound - true_kernel;

	  lgap_sum += lgap;
	  ugap_sum += ugap;
	  if (ubound > 0) {
	    gap_percent += (ubound-lbound)/ubound;
	    k_percent += true_kernel/ubound;
	  } else {
	    k_percent += 1.0;
	  }
	  n += 1;

	  if ((lgap < -1e-8) || ugap < -1e-8) {
	    printf("bound error! d1 %f d2 %f delta %f k %f lbound %f ubound %f lgap %f ugap %f\n", d1, d2, delta, true_kernel, lbound, ubound, lgap, ugap);
	  }
	  if (ugap > 10000) {
	    printf("massive gap! d1 %f d2 %f delta %f k %f lbound %f ubound %f lgap %f ugap %f\n", d1, d2, delta, true_kernel, lbound, ubound, lgap, ugap);
	  }
    }
  }
  printf("finished bounds test for %d distance pairs. average lgap %f, ugap %f, avg gap is %f of ubound, avg k is %f of ubound\n", n, lgap_sum/n, ugap_sum/n, gap_percent/n, k_percent/n);

}

MatrixTree::~MatrixTree() {
  if (this->dist_params != NULL) {
    delete[] this->dist_params;
    this->dist_params = NULL;
  }
  if (this->wp_pair != NULL) {
    delete[] this->wp_pair;
    this->wp_pair = NULL;
  }
  if (this->wp_point != NULL) {
    delete[] this->wp_point;
    this->wp_point = NULL;
  }
  if (this->dfn_extra->dfn_extra != NULL) {
    free(this->dfn_extra->dfn_extra);
    this->dfn_extra->dfn_extra = NULL;
  }
  delete this->dfn_extra;
}
