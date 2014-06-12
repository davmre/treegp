 #include "cover_tree.hpp"
 #include "vector_mult.hpp"
 #include <cstdlib>
 #include <cmath>
 #include <memory>
 #include <vector>
 #include <limits>

 using namespace std;
 namespace bp = boost::python;

int extra_p_counter = 0;
int fn_counter = 0;

void write_header(FILE *fp, int debug_level, int dims, int n) {
fprintf(fp, "\
\n\
#include <cmath>\n\
#include <boost/python/module.hpp>\n\
#include <boost/python/def.hpp>\n\
#include <pyublas/numpy.hpp>\n\
\n\
struct pairpoint {\n\
  double pt1[%d];\n\
  double pt2[%d];\n\
  unsigned int idx1;\n\
  unsigned int idx2;\n\
};\n\
", dims, dims);

   fprintf(fp, "\n\
void print_point(const double *p);\n\
void print_pairpoint(const pairpoint &p);\n\
double distance(const double *p1, const double *p2);\n\
double w_point(double r);\n\
double w_lower(double r);\n\
double w_upper(double r);\n\
double first_half_w_query_cached(const pairpoint &query_pt, const pairpoint &p2);\n\
double second_half_w_query_cached(const pairpoint &query_pt, const pairpoint &p2);\n\
double factored_query_distance(const pairpoint &query_pt, const pairpoint &p2);\n\
\n\
void weighted_sum_offdiag(pairpoint &query_pt, double &ws, double &abserr_sofar, int &terms_sofar, double eps_abs);\n\
void weighted_sum_diag(pairpoint &query_pt, double &ws, double &abserr_sofar, int &terms_sofar, double eps_abs);\n\
");
}

void write_weighted_sum_prototypes(FILE *fp, int debug_level) {
  for (int i=1; i <= fn_counter; ++i) {
    fprintf(fp, "void continue_weighted_sum_%d(pairpoint &query_pt, double &ws, double &abserr_sofar, int &terms_sofar, double eps_abs, double d, pairpoint *current_pt);\n", i);
  }
}

void write_boilerplate_top(FILE *fp, int debug_level, int dims, int n) {
fprintf(fp, "\
\n\
\n\
  double cached_query_distances[%d];\n\
  double cached_query_weights[%d];\n\
  unsigned int cached_query_distances_pts[%d];\n\
  unsigned int cached_query_weights_pts[%d];\n\
  unsigned int query_idx = 0;\n\
\n\
", n, n, n, n);

   fprintf(fp, "\n\
  int terms = 0;\n\
  int zeroterms = 0;\n\
  int nodes_touched = 0;\n\
  int dfn_evals = 0;\n\
\n\
  int get_terms() { return terms; }\n\
  int get_zeroterms() { return zeroterms; }\n\
  int get_nodes_touched() { return nodes_touched; }\n\
  int get_dfn_evals() { return dfn_evals; }\n\
");


   fprintf(fp, "\n\
void print_point(const double *p) {\n\
  printf(\"(\");\n\
  for (int i=0; i < %d-1; ++i) { printf(\"%%f, \", p[i]); }\n\
  printf(\"%%f)\", p[%d-1]);\n\
}\n\
void print_pairpoint(const pairpoint &p) {\n\
  printf(\"{ \");\n\
  print_point(p.pt1);\n\
  printf(\", \");\n\
  print_point(p.pt2);\n\
  printf(\", %%d, %%d }\", p.idx1, p.idx2);\n}\n", dims, dims);

}

void write_boilerplate_bottom(FILE *fp, int debug_level) {
  fprintf(fp, "\n\
using namespace boost::python;\n\
\n\
BOOST_PYTHON_MODULE(compiled_tree)\
{\n\
    def(\"quadratic_form_symmetric\", quadratic_form_symmetric);\n\
    def(\"init_distance_caches\", init_distance_caches);\n");

  if (debug_level >= 1) {
    fprintf(fp, "\
    def(\"get_nodes_touched\", get_nodes_touched);\n\
    def(\"get_terms\", get_terms);\n\
    def(\"get_zeroterms\", get_zeroterms);\n\
    def(\"get_dfn_evals\", get_dfn_evals);\n\
");
  }

  fprintf(fp, "}\n");
}

void write_euclidean_distance(FILE *fp, int debug_level, int dim, double *scales) {
  fprintf(fp, "\n\
double distance(const double *p1, const double *p2) {		\n\
  double sqdist = 0;\n\
  double diff = 0;\n\
");

  if (debug_level >= 1) {
    fprintf(fp, "dfn_evals += 1;\n");
  }

  for(int i=0; i < dim; ++i) {
    fprintf(fp, "\n\
  diff = (p1[%d] - p2[%d]) / %.12f;\n\
  sqdist += (diff * diff);\n\
", i, i, scales[i]);
  }

  if (debug_level >= 3) {
    fprintf(fp, "printf(\"called distance on \");\nprint_point(p1);\nprintf(\", \");print_point(p2);\nprintf(\", got %%f\\n\", sqrt(sqdist));\n");
  }

  fprintf(fp, "  return sqrt(sqdist);\n}\n");
}

void write_w_compact2(FILE *fp, int debug_level, double point_variance, double pair_variance, int j) {
  fprintf(fp, "\n\
double w_point(double r) {\n\
  if (r >= 1) return 0.0;\n\
\n\
  double d = 1.0 - r;\n\
\n\
  double variance = %.12f;\n\
  int j = %d;\n\
\n\
  double poly = ((j*j + 4*j + 3)*r*r + (3*j + 6)*r + 3)/3.0;\n\
  return variance * pow(d, j+2)*poly;\n\
}\
", point_variance, j);

  fprintf(fp, "\n\
double w_lower(double r) {\n\
  double d = 1.0 - r;\n\
  if (d <= 0.0) {\
    return 0.0;\n\
  }\n\
\n\
  double variance =%.12f;\n\
  int j = %d;\n\
\n\
  double poly1 = (3*j*j + 12*j + 9) * r * r / 2.0;\n\
  double poly2 = (9*j + 18) * r;\n\
  double poly = (poly1 + poly2 + 9.0)/9.0;\n\
  return variance * pow(d, j+2) * poly;\n\
}\n\
\n\
double w_upper(double r) {\n\
  double d = 1.0 - r + .25 * r * r; \n\
  if (r >= 2.0) {\n\
    return 0.0;\n\
  }\n\
\n\
  double variance = %.12f;\n\
  int j = %d;\n\
\n\
  double rsq4 = r * r / 4.0;\n\
  double jquad = (j*j + 4*j + 3);\n\
  double jlinear = 3*j + 6;\n\
\n\
  double poly1 = jquad * rsq4;\n\
  double poly1sq = poly1 * poly1;\n\
  double poly2 = jquad * jlinear * rsq4 * r;\n\
  double poly3 = poly1 * 12.0;\n\
  double poly4 = jlinear * jlinear * rsq4;\n\
  double poly5 = jlinear * r * 3.0;\n\
  double poly = (poly1sq + poly2 + poly3 + poly4 + poly5 + 9.0) / 9.0;\n\
\n\
  return variance * pow(d, j+2) * poly;\n\
}\n\
", pair_variance, j, pair_variance, j);
}


  /* lots of things wrong here. we might need to fold query_id into
     pairpoint, or we might want to eliminate pairpoint altogether and
     just pass double[] references. though we NEED indices for the
     training points, and can't hardcode the query point anyway, so
     maybe not.  also, we need to specify these functions as class
     members so they can access the cache variables. maybe. */

void write_distance_cache_boilerplate(FILE *fp, int debug_level, int n) {

  fprintf(fp, "\n\
void init_distance_caches() {\n\
 query_idx = 1;\n\
 for(int i=0; i < %d; ++i ) {\n\
   cached_query_distances_pts[i] = 0;\n\
   cached_query_weights_pts[i] = 0;\n\
 }\n\
}\n\
\n\
double first_half_d_query_cached(const pairpoint &query_pt, const pairpoint &p2) {\n\
  int idx1 = p2.idx1;\n\
  if (cached_query_distances_pts[idx1] == query_pt.idx1) {\n\
    return cached_query_distances[idx1];\n\
  } else {\n\
    double d1 = distance(query_pt.pt1, p2.pt1);\n\
    cached_query_distances[idx1] = d1;\n\
    cached_query_distances_pts[idx1] = query_pt.idx1;\n\
    return d1;\n\
  }\n\
}\n\
double first_half_w_query_cached(const pairpoint &query_pt, const pairpoint &p2) {\n\
  int idx1 = p2.idx1;\n\
  if (cached_query_weights_pts[idx1] == query_pt.idx1) {\n\
    return cached_query_weights[idx1];\n\
  } else {\n\
    double d1 = first_half_d_query_cached(query_pt, p2);\n\
    cached_query_weights[idx1] = w_point(d1);\n\
    cached_query_weights_pts[idx1] = query_pt.idx1;\n\
    return cached_query_weights[idx1];\n\
  }\n\
}\n\
double second_half_d_query_cached(const pairpoint &query_pt, const pairpoint &p2) {\n\
  int idx2 = p2.idx2;\n\
  if (cached_query_distances_pts[idx2] == query_pt.idx1) {\n\
    return cached_query_distances[idx2];\n\
  } else {\n\
    double d1 = distance(query_pt.pt1, p2.pt1);\n\
    cached_query_distances[idx2] = d1;\n\
    cached_query_distances_pts[idx2] = query_pt.idx1;\n\
    return d1;\n\
  }\n\
}\n\
double second_half_w_query_cached(const pairpoint &query_pt, const pairpoint &p2) {\n\
  int idx2 = p2.idx2;\n\
  if (cached_query_weights_pts[idx2] == query_pt.idx1) {\n\
    return cached_query_weights[idx2];\n\
  } else {\n\
    double d1 = second_half_d_query_cached(query_pt, p2);\n\
    cached_query_weights[idx2] = w_point(d1);\n\
    cached_query_weights_pts[idx2] = query_pt.idx1;\n\
    return cached_query_weights[idx2];\n\
  }\n\
}\n\
double factored_query_distance(const pairpoint &query_pt, const pairpoint &p2) {\n\
       double d1 = first_half_d_query_cached(query_pt, p2);\n\
       double d2 = second_half_d_query_cached(query_pt, p2);\n\
       return d1 + d2;\n\
}\n\
", n);
}

void write_point_literal(FILE *fp, int debug_level, const double* pt, int dims) {
  fprintf(fp, "{");
  int d;
  for(d=0; d < dims-1; ++d) {
    fprintf(fp, "%.12f, ", pt[d]);
  }
  fprintf(fp, "%.12f }", pt[d]);
}

void write_pairpoint_literal(FILE *fp, int debug_level, pairpoint p, int dims) {
  fprintf(fp, " { ");
  write_point_literal(fp, debug_level, p.pt1, dims);
  fprintf(fp, ", ");
  write_point_literal(fp, debug_level, p.pt2, dims);
  fprintf(fp, ", %d, %d }", p.idx1, p.idx2);
}

void write_weighted_sum_node(FILE *fp, int debug_level, char *dirname, node<pairpoint> n, int depth, int max_terms, int dims, int diag);

void write_weighted_sum_nonleaf(FILE *fp, int debug_level, char *dirname, node<pairpoint> n, int depth, int max_terms, int dims, int diag) {
  /* in order to have sorted children, we'll have had to have a hard-coded list of children. so I shouldn't hard-code the points again here, but I can assume that there is *already* a pairpoint object current_pt defined in the code. I can also assume that there's *already* a distance d computed.
  */

  int v_select = 0;

  fprintf(fp, "\n\
cutoff = false;\n\
if (d > %.12f ) {\n\
    double min_weight = w_lower(d + %.12f );\n\
    double max_weight = w_upper(d - %.12f );\n\
\n\
    double frac_remaining_terms = %d  / (double) (%d  - terms_sofar);\n\
    double threshold = frac_remaining_terms * (eps_abs - abserr_sofar);\n\
    double abserr_n = .5 * (max_weight - min_weight) * %.12f;\n\
    cutoff = abserr_n < threshold;\n\
    if (abserr_n < threshold) {\n\
      ws += .5 * (max_weight + min_weight) * %.12f;\n\
      terms_sofar += %d;\n\
      abserr_sofar += abserr_n;\n", n.max_dist, n.max_dist, n.max_dist, n.num_leaves, max_terms, n.unweighted_sums_abs[v_select], n.unweighted_sums[v_select], n.num_leaves);

  if (debug_level >= 1) {
    fprintf(fp, "if (((max_weight + min_weight) * %.12f) != 0) {\n\
 terms += 1;\n\
} else {\n\
 zeroterms += 1;\n\
}", n.unweighted_sums[v_select]);
  }

    if (debug_level >= 2) {
      fprintf(fp, "printf(\"cutoff, weight (%%f to %%f), added %%f for cumulative ws %%f \", min_weight, max_weight, .5*(max_weight + min_weight) * %.12f, ws);\n", n.unweighted_sums[v_select]);
    }


fprintf(fp, "\
    }\n\
  }\n\
  if (!cutoff) {\n\
    pairpoint child_points[] = {");

  for (int i=0; i < n.num_children; ++i) {
    write_pairpoint_literal(fp, debug_level, n.children[i].p, dims);
    if (i < n.num_children-1) fprintf(fp, ",\n");
  }
  fprintf(fp, "};\n");

  for (int i=0; i < n.num_children; ++i) {
    if (n.children[i].n_extra_p == 0) {
      fprintf(fp, "d = factored_query_distance(query_pt, child_points[%d]);\n", i);
      fprintf(fp, "current_pt = child_points + %d;\n", i);

      if (debug_level >= 2) {
	fprintf(fp, "printf(\"recursing to child: \");\n");
	fprintf(fp, "print_pairpoint(*current_pt);\nprintf(\"\\n\");");
      }
    }

    write_weighted_sum_node(fp, debug_level, dirname, n.children[i], depth, max_terms, dims, diag);
  }
  fprintf(fp, "\n}\n");
}

void write_weighted_sum_leaf(FILE *fp, int debug_level, node<pairpoint> n, int diag) {
  int v_select = 0;
  double unweighted_sum = n.unweighted_sums[v_select] + (1-diag) * n.unweighted_sums[v_select];
  fprintf(fp, "\n\
weight = first_half_w_query_cached(query_pt, *current_pt) * second_half_w_query_cached(query_pt, *current_pt);\n\
ws += weight * %.12f;\n\
terms_sofar += 1;\n\
", unweighted_sum);

  if (debug_level >= 1) {
    fprintf(fp, "if ((weight * %.12f) != 0) {\n\
 terms += 1;\n\
} else {\n\
 zeroterms += 1;\n\
}", unweighted_sum);
  }

    if (debug_level >= 2) {
      fprintf(fp, "printf(\"at leaf, weight %%f, added %%f for cumulative ws %%f \", weight, weight*%.12f, ws);\n", unweighted_sum);
    }



}

void write_weighted_sum_multileaf(FILE *fp, int debug_level, node<pairpoint> n, int dims, int diag) {
  int v_select = 0;
  double *epvals = n.extra_p_vals[v_select];
  for (int i=0; i < n.n_extra_p; ++i) {
    fprintf(fp, "pairpoint extra_p_%d = ", ++extra_p_counter);
    write_pairpoint_literal(fp, debug_level, n.extra_p[i], dims);
    fprintf(fp, ";\n");
    fprintf(fp, "weight = first_half_w_query_cached(query_pt, extra_p_%d) * second_half_w_query_cached(query_pt, extra_p_%d);\n", extra_p_counter, extra_p_counter);
    fprintf(fp, "ws += weight * %.12f;\n", epvals[i] + (1-diag) * epvals[i]);

    if (debug_level >= 1) {
      fprintf(fp, "if ((weight * %.12f) != 0) {\n\
 terms += 1;\n\
} else {\n\
 zeroterms += 1;\n\
}", epvals[i] + (1-diag) * epvals[i]);
    }


  }
  fprintf(fp, "terms_sofar += %d;\n", n.n_extra_p);
}

void write_treefile_preamble(FILE *fp, int debug_level) {
  fprintf(fp, "#include \"compiled_tree.h\"\n\n");
  fprintf(fp, "\
extern int terms;\n\
extern int zeroterms;\n\
extern int nodes_touched;\n\
extern int dfn_evals;\n\
");
}

void write_new_treefile(FILE *fp, int debug_level, char *dirname, node<pairpoint> n, int depth, int max_terms, int dims, int diag) {
  char new_tree_fname[512];
  snprintf(new_tree_fname,512, "%s/tree_%d.cc", dirname, ++fn_counter);

  fprintf(fp, "continue_weighted_sum_%d(query_pt, ws, abserr_sofar, terms_sofar, eps_abs, d, current_pt);\n", fn_counter);

  FILE *new_tree_fp = fopen(new_tree_fname, "w");
  write_treefile_preamble(new_tree_fp, debug_level);
  fprintf(new_tree_fp, "\n\nvoid continue_weighted_sum_%d(pairpoint &query_pt, double &ws, double &abserr_sofar, int &terms_sofar, double eps_abs, double d, pairpoint *current_pt) {\n\
double weight;\n\
bool cutoff;\n", fn_counter);
  write_weighted_sum_nonleaf(new_tree_fp, debug_level, dirname, n, depth, max_terms, dims, diag);
  fprintf(new_tree_fp, "}\n");
  fclose(new_tree_fp);
}

void write_weighted_sum_node(FILE *fp, int debug_level, char *dirname, node<pairpoint> n, int depth, int max_terms, int dims, int diag) {
  if (debug_level >= 1) {
    fprintf(fp, "nodes_touched += 1;\n");
  }

  if (n.num_children == 0) {
    if (n.n_extra_p == 0) {
      write_weighted_sum_leaf(fp, debug_level, n, diag);
    } else {
      write_weighted_sum_multileaf(fp, debug_level, n, dims, diag);
    }
  } else {

    if (n.num_children == 1) {
      printf("skipping node with only a single child!\n");
      write_weighted_sum_node(fp, debug_level, dirname, n.children[0], depth, max_terms, dims, diag);
    } else {

      if ( (depth % 5) == 0 && n.num_leaves > 20) {
	write_new_treefile(fp, debug_level, dirname, n, depth+1, max_terms, dims, diag);
      } else {
	write_weighted_sum_nonleaf(fp, debug_level, dirname, n, depth+1, max_terms, dims, diag);
      }
    }
  }
}

void write_quadratic_form_symmetric(FILE *fp, int debug_level, int dims) {

  fprintf(fp, "double quadratic_form_symmetric(const pyublas::numpy_matrix<double> &query_pt, double eps_abs) {\n\
pairpoint qp;\n\
for(int i=0; i < %d; ++i) {\n\
  qp.pt1[i] = query_pt(0,i);\n\
  qp.pt2[i] = query_pt(0,i);\n\
}\n\
qp.idx1 = query_idx++;\n\
qp.idx2 = 0;\n\
", dims);

  if (debug_level >= 1) {
    fprintf(fp, "terms = 0;\nzeroterms=0;\ndfn_evals=0;\nnodes_touched=0;\n");
  }
  if (debug_level >= 2) {
    fprintf(fp, "printf(\"called quadratic_form_symmetric with query: \");\n\
print_pairpoint(qp);\n\
printf(\"\\n\");\n");
  }

  fprintf(fp, "\n\
double ws = 0;\n\
int terms_sofar = 0;\n\
double abserr_sofar = 0;\n\
weighted_sum_diag(qp, ws, abserr_sofar, terms_sofar, eps_abs);\n\
weighted_sum_offdiag(qp, ws, abserr_sofar, terms_sofar, eps_abs);\n\
return ws;\n\
}\n\
");
}

void write_weighted_sum_init(FILE *fp, int debug_level, pairpoint &p, int dims) {
  fprintf(fp, "pairpoint root = ");
  write_pairpoint_literal(fp, debug_level, p, dims);
  fprintf(fp, ";\npairpoint *current_pt = &root;");
  fprintf(fp, ";\ndouble d = factored_query_distance(query_pt, *current_pt);\n");
  fprintf(fp, "bool cutoff;\ndouble weight;\n");
}

void write_weighted_sum_diag(FILE *fp, int debug_level, char *dirname, node<pairpoint> root_diag, int max_terms, int dims) {
  fprintf(fp, "void weighted_sum_diag(pairpoint &query_pt, double &ws, double &abserr_sofar, int &terms_sofar, double eps_abs) {\n");

  if (debug_level >= 2) {
    fprintf(fp, "printf(\"weighted_sum_diag\\n\");\n");
  }

  write_weighted_sum_init(fp, debug_level, root_diag.p, dims);
  write_weighted_sum_node(fp, debug_level, dirname, root_diag, 0, max_terms, dims, 1);
  fprintf(fp, "}\n");
}

void write_weighted_sum_offdiag(FILE *fp, int debug_level, char *dirname, node<pairpoint> root_offdiag, int max_terms, int dims) {
  fprintf(fp, "void weighted_sum_offdiag(pairpoint &query_pt, double &ws, double &abserr_sofar, int &terms_sofar, double eps_abs) {\n");

  if (debug_level >= 2) {
    fprintf(fp, "printf(\"weighted_sum_offdiag ws=%%f\\n\", ws);\n");
  }

  write_weighted_sum_init(fp, debug_level, root_offdiag.p, dims);
  write_weighted_sum_node(fp, debug_level, dirname, root_offdiag, 0, max_terms, dims, 0);
  fprintf(fp, "}\n");
}


void MatrixTree::compile(char *dirname, int debug_level) {

  // open file


  int dims = *((int *) this->dfn_extra->dfn_extra);
  printf("writing compiled code with %d dimensions\n.", dims);
  double * scales = this->dist_params;
  double point_variance = this->wp_point[0];
  double pair_variance = this->wp_pair[0];
  int j = int(this->wp_point[1]);
  int max_terms = this->root_diag.num_leaves + this->root_offdiag.num_leaves;

  // write Python module boilerplate
  char header_fname[512];
  snprintf(header_fname, 512, "%s/compiled_tree.h", dirname);
  FILE *header_fp = fopen(header_fname, "w");
  write_header(header_fp, debug_level, dims, this->n);

  // write distance and weight functions
  char common_fname[512];
  snprintf(common_fname, 512, "%s/common.cc", dirname);
  FILE *common_fp = fopen(common_fname, "w");
  fprintf(common_fp, "#include \"compiled_tree.h\"\n\n");
  write_boilerplate_top(common_fp, debug_level, dims, this->n);
  write_euclidean_distance(common_fp, debug_level, dims, scales);
  write_w_compact2(common_fp, debug_level, point_variance, pair_variance, j);
  write_distance_cache_boilerplate(common_fp, debug_level, this->n);
  write_quadratic_form_symmetric(common_fp, debug_level, dims);

  // write weight function
  char tree_fname[512];
  snprintf(tree_fname, 512, "%s/tree.cc", dirname);
  FILE *tree_fp = fopen(tree_fname, "w");
  write_treefile_preamble(tree_fp, debug_level);
  write_weighted_sum_diag(tree_fp, debug_level, dirname, this->root_diag, max_terms, dims);
  write_weighted_sum_offdiag(tree_fp, debug_level, dirname, this->root_offdiag, max_terms, dims);
  fclose(tree_fp);
  write_weighted_sum_prototypes(header_fp, debug_level);

  // write any other required boilerplate
  write_boilerplate_bottom(common_fp, debug_level);


  fclose(common_fp);
  fclose(header_fp);

}
