#ifndef COVER_TREE_H
#define COVER_TREE_H

#include<string.h>
#include<string>
#include<math.h>
#include<list>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include <vector>
#include "stack.h"

#include <limits>
#include <stdint.h>
#include <iostream>


struct point {
  const double * p;
  unsigned int idx;
};

struct pairpoint {
  const double * pt1;
  const double * pt2;
  unsigned int idx1;
  unsigned int idx2;
};

template<typename T> struct distfn {
  typedef double (*Type)(const T, const T, double, const double*, void*);
};

typedef double (*partial_dfn)(const double *, const double *, double, const double*, const void*);


typedef double (*dfn_deriv)(const double *, const double *,  int i, double, double, const double *, void *);

double dist_euclidean(const point p1, const point p2, double BOUND_IGNORED, const double *scales, void *dims);
double dist_euclidean(const double *, const double *, double BOUND_IGNORED, const double *scales, const void *dims);
double sqdist_euclidean(const double *, const double *, double BOUND_IGNORED, const double *scales, const void *dims);
double pair_dist_euclidean(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *scales, void *dims);
double dist_euclidean_deriv_wrt_xi(const double *p1, const double *p2, int i, double d, double BOUND_IGNORED, const double *scales, void *dims);
double dist_euclidean_deriv_wrt_theta(const double *p1, const double *p2, int i, double d, double BOUND_IGNORED, const double *scales, void *dims);

double dist_km(const double *p1, const double *p2);
double dist_3d_km(const point p1, const point p2, double BOUND_IGNORED, const double *scales, void * extra);
double dist_3d_km(const double * p1, const double*  p2, double BOUND_IGNORED, const double *scales, const void * extra);
double distsq_3d_km(const double * p1, const double*  p2, double BOUND_IGNORED, const double *scales, const void * extra);
double pair_dist_3d_km(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *scales, void * extra);
double dist3d_deriv_wrt_theta(const double *p1, const double *p2, int i, double d, double BOUND_IGNORED, const double *scales, void *dims);
double dist3d_deriv_wrt_xi(const double * p1, const double * p2, int i, double d, double BOUND_IGNORED, const double *scales, void * extra);

double dist_6d_km(const point p1, const point p2, double BOUND_IGNORED, const double *scales, void *extra);
double dist_6d_km(const double * p1, const double * p2, double BOUND_IGNORED, const double *scales, const void *extra);
double distsq_6d_km(const double * p1, const double * p2, double BOUND_IGNORED, const double *scales,const void *extra);
double pair_dist_6d_km(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *scales, void * extra);
double dist6d_deriv_wrt_theta(const double *p1, const double *p2, int i, double d, double BOUND_IGNORED, const double *scales, void *dims);

typedef double (*wfn)(double, const double *);
typedef double (*wfn_deriv)(double, double, const double*);

double w_se(double d, const double * variance);
double deriv_se_wrt_r(double r, double dr_dtheta, const double *variance);
double w_e(double d, const double * variance);
double w_matern32(double d, const double * variance);
double w_matern32_upper(double d, const double * variance);
double w_matern32_lower(double d, const double * variance);
double deriv_matern32_wrt_r(double r, double dr_dtheta, const double *variance);
double w_compact_q0(double d, const double * extra);
double w_compact_q0_upper(double d, const double * extra);
double w_compact_q0_lower(double d, const double * extra);
double w_compact_q2(double d, const double * extra);
double deriv_compact_q2_wrt_r(double r, double dr_dtheta, const double *extra);
double w_compact_q2_upper(double d, const double * extra);
double w_compact_q2_lower(double d, const double * extra);




template<class T>
class node {
 public:
  T p;
  double max_dist;  // The maximum distance to any grandchild.
  double parent_dist; // The distance to the parent.
  node<T>* children;
  unsigned short int num_children; // The number of children.
  unsigned int num_leaves; // The number of leaves under this node.
  short int scale; // Essentially, an upper bound on the distance to any child.

  // additional info to support vector multiplication
  double distance_to_query;

  unsigned int narms;
  double *unweighted_sums; // unweighted sums of each vector, over the points contained at this node
  double *unweighted_sums_abs; // unweighted absolute sums of each vector, over the points contained at this node
  double unweighted_sum;
  double unweighted_sum_abs;


  T *extra_p;
  unsigned int n_extra_p;
  double **extra_p_vals; // a list of n_arms elements, each containing
			 // values for each of the n_extra_p points

  node<T>* debug_parent;

  node();
  //~node();
  void alloc_arms(unsigned int narms);
  void free_tree();
};

template<class T>
node<T>::node() {
  /*
    WARNING: because I haven't defined a copy constructor, crazy
    things can happen with the default copy constructor where one node
    ends up pointing at the internals of another node. I should fix this.
   */

  this->unweighted_sums = &(this->unweighted_sum);
  this->unweighted_sums_abs = &(this->unweighted_sum_abs);
  this->narms = 1;
  this->n_extra_p = 0;
}

/*
template<class T>
node<T>::~node() {


  this->free_tree_recursive();
  if (this->narms > 1) {
    delete this->unweighted_sums;
    delete this->unweighted_sums_abs;
  }

  if(this->n_extra_p > 0) {
    delete this->extra_p;
    for (unsigned int a=0; a < this->narms; ++a) {
      delete this->extra_p_vals[a];
    }
    free( this->extra_p_vals); // mallocced in collect_leaves
    }
}*/

void epsilon_nearest_neighbor(const node<point> &top_node, const node<point> &query,
			      v_array<v_array<point> > &results, double epsilon,
			      distfn<point>::Type distance,
			      const double* dist_params, void* dist_extra);
void k_nearest_neighbor(const node<pairpoint> &top_node, const node<pairpoint> &query,
			v_array<v_array< node<pairpoint> > > &results, int k,
			distfn<pairpoint>::Type distance,
			const double* dist_params, void* dist_extra);

template<class T>
void node<T>::alloc_arms(unsigned int narms) {
  if (this->narms > 1) {
    delete this->unweighted_sums;
    this->unweighted_sums = NULL;
    delete this->unweighted_sums_abs;
    this->unweighted_sums_abs = NULL;
  }

  if ( narms > 1 ) {
    this->unweighted_sums = new double[narms];
    this->unweighted_sums_abs = new double[narms];
  } else {
    this->unweighted_sums = &(this->unweighted_sum);
    this->unweighted_sums_abs = &(this->unweighted_sum_abs);
  }
  this->narms = narms;

  // recursively apply to all children
  for(unsigned int i=0; i < this->num_children; ++i) {
    this->children[i].alloc_arms(narms);
  }
}


template<class T>
void node<T>::free_tree() {
  if (this->narms > 1) {
    delete this->unweighted_sums;
    delete this->unweighted_sums_abs;
  }

  for(unsigned int i=0; i < num_children; ++i) {
    delete (this->children + i);
  }
}


template<class T>
void set_leaves(node<T> &n, node<T> * parent) {
  n.debug_parent = parent;
  if (n.num_children == 0) {
    n.num_leaves = 1;
  } else {
    n.num_leaves = 0;
    for(int i=0; i < n.num_children; ++i) {
      set_leaves(n.children[i], &n);
      n.num_leaves += n.children[i].num_leaves;
    }
  }
}

template <class T>
struct ds_node {
  v_array<double> dist;
  T p;
};

const double base = 1.3;
const double il2 = 1. / log(base);

inline double dist_of_scale (int s)
{

  return pow(base, s);
}

inline int get_scale(double d)
{

  return (int) ceilf(il2 * log(d));
}

inline int min(int f1, int f2)
{
  if ( f1 <= f2 )
    return f1;
  else
    return f2;
}

inline double max(double f1, double f2)
{
  if ( f1 <= f2 )
    return f2;
  else
    return f1;
}

template <class T>
node<T> new_node(const ds_node<T> &p)
{
  node<T> new_node;
  new_node.p = p.p;
  return new_node;
}

template <class T>
node<T> new_leaf(const ds_node<T> &p)
{
  node<T> new_leaf;
  new_leaf.p = p.p;
  new_leaf.max_dist = 0.;
  new_leaf.parent_dist = 0.;
  new_leaf.children = NULL;
  new_leaf.num_children = 0;
  new_leaf.scale = 100;
  return new_leaf;
}

template <class T>
double max_set(v_array< ds_node<T> > &v)
{
  double max = 0.;
  for (int i = 0; i < v.index; i++)
    if ( max < v[i].dist.last())
      max = v[i].dist.last();
  return max;
}

template <class T>
void split(v_array< ds_node<T> >& point_set, v_array< ds_node<T> >& far_set, int max_scale)
{
  unsigned int new_index = 0;
  double fmax = dist_of_scale(max_scale);
  for (int i = 0; i < point_set.index; i++){
    if (point_set[i].dist.last() <= fmax) {
      point_set[new_index++] = point_set[i];
    }
    else
      push(far_set,point_set[i]);
  }
  point_set.index=new_index;
}

template <class T>
void dist_split(v_array<ds_node <T> >& point_set,
		v_array<ds_node <T> >& new_point_set,
		T new_point,
		int max_scale,
		typename distfn<T>::Type distance,
		const double* dist_params,
		void * dist_extra)
{
  /*

   */

  unsigned int new_index = 0;
  double fmax = dist_of_scale(max_scale);
  for(int i = 0; i < point_set.index; i++)
    {
      double new_d;
      new_d = distance(new_point, point_set[i].p, fmax, dist_params, dist_extra);
      if (new_d <= fmax ) {
	push(point_set[i].dist, new_d);
	push(new_point_set,point_set[i]);
      }
      else
	point_set[new_index++] = point_set[i];
    }
  point_set.index = new_index;
}

/*
   max_scale is the maximum scale of the node we might create here.
   point_set contains points which are 2*max_scale or less away.
*/

template<class T>
node<T> batch_insert(const ds_node<T>& p,
		  int max_scale,
		  int top_scale,
		  v_array< ds_node<T> >& point_set,
		  v_array< ds_node<T> >& consumed_set,
		  v_array<v_array< ds_node<T> > >& stack,
		  typename distfn<T>::Type distance,
		const double* dist_params, void* dist_extra)
{
  if (point_set.index == 0)
    return new_leaf(p);
  else {
    double max_dist = max_set(point_set); //O(|point_set|)
    int next_scale = min (max_scale - 1, get_scale(max_dist));

    /*
    if ((next_scale == -2147483648) || (top_scale - next_scale > 100) ) {
      printf("whoawhoawhoa\n");
      printf("max_set %f\n", max_dist);
      printf("next scale %d %d %d\n", max_scale-1, get_scale(max_dist), next_scale);
      //printf("p %f %f %f %f\n", p.p.pt1[0], p.p.pt1[1], p.p.pt2[0], p.p.pt2[1]);
      //printf("p %f %f\n", p.p.p[0], p.p.p[1]);
      //printf("ps %f %f %f %f\n", point_set[0].p.pt1[0], point_set[0].p.pt1[1], point_set[0].p.pt2[0], point_set[0].p.pt2[1]);
      printf("distance %f\n", distance(point_set[0].p, p.p, std::numeric_limits< double >::max(), dist_params, dist_extra));
      }*/

    if ((next_scale == -2147483648) || (top_scale - next_scale >= 100)) // We have points with distance 0.
      {
	v_array< node<T> > children;
	push(children,new_leaf(p));
	while (point_set.index > 0)
	  {
	    push(children,new_leaf(point_set.last()));
	    push(consumed_set,point_set.last());
	    point_set.decr();
	  }
	node<T> n = new_node(p);
	n.scale = 100; // A magic number meant to be larger than all scales.
	n.max_dist = 0;
	alloc(children,children.index);
	n.num_children = children.index;
	n.children = children.elements;
	return n;
      }
    else
      {
	v_array< ds_node<T> > far = pop(stack);
	split(point_set,far,max_scale); //O(|point_set|)

	node<T> child = batch_insert(p, next_scale, top_scale,
				  point_set, consumed_set, stack,
				  distance, dist_params, dist_extra);

	if (point_set.index == 0)
	  {
	    push(stack,point_set);
	    point_set=far;
	    return child;
	  }
	else {
	  node<T> n = new_node(p);
	  v_array< node<T> > children;
	  push(children, child);
	  v_array<ds_node<T> > new_point_set = pop(stack);
	  v_array<ds_node<T> > new_consumed_set = pop(stack);
	  while (point_set.index != 0) { //O(|point_set| * num_children)
	    ds_node<T> new_point_node = point_set.last();
	    T new_point = point_set.last().p;
	    double new_dist = point_set.last().dist.last();
	    push(consumed_set, point_set.last());
	    point_set.decr();

	    dist_split(point_set, new_point_set, new_point, max_scale, distance, dist_params, dist_extra); //O(|point_saet|)
	    dist_split(far,new_point_set,new_point,max_scale, distance, dist_params, dist_extra); //O(|far|)

	    node<T> new_child =
	      batch_insert(new_point_node, next_scale, top_scale,
			   new_point_set, new_consumed_set, stack,
			   distance, dist_params, dist_extra);
	    new_child.parent_dist = new_dist;

	    push(children, new_child);

	    double fmax = dist_of_scale(max_scale);
	    for(int i = 0; i< new_point_set.index; i++) //O(|new_point_set|)
	      {
		new_point_set[i].dist.decr();
		if (new_point_set[i].dist.last() <= fmax)
		  push(point_set, new_point_set[i]);
		else
		  push(far, new_point_set[i]);
	      }
	    for(int i = 0; i< new_consumed_set.index; i++) //O(|new_point_set|)
	      {
		new_consumed_set[i].dist.decr();
		push(consumed_set, new_consumed_set[i]);
	      }
	    new_point_set.index = 0;
	    new_consumed_set.index = 0;
	  }
	  push(stack,new_point_set);
	  push(stack,new_consumed_set);
	  push(stack,point_set);
	  point_set=far;
	  n.scale = top_scale - max_scale;
	  n.max_dist = max_set(consumed_set);
	  alloc(children,children.index);
	  n.num_children = children.index;
	  n.children = children.elements;
	  return n;
	}
      }
  }
}

template<class T> node<T> batch_create(const std::vector<T> &points,
					typename distfn<T>::Type distance,
					const double* dist_params, void* dist_extra)
{
  v_array<ds_node<T> > point_set;
  v_array<v_array<ds_node<T> > > stack;

  ds_node<T> initial_pt;
  initial_pt.p = points[0];
  for (std::vector<point>::size_type i = 1; i < points.size(); i++) {
    ds_node<T> temp;
    push(temp.dist, distance(points[0], points[i], std::numeric_limits< double >::max(), dist_params, dist_extra));
    temp.p = points[i];
    push(point_set,temp);
  }
  v_array< ds_node < T > > consumed_set;

  double max_dist = max_set(point_set);

  node<T> top = batch_insert(initial_pt,
			  get_scale(max_dist),
			  get_scale(max_dist),
			  point_set,
			  consumed_set,
			  stack,
			  distance, dist_params, dist_extra);
  for (int i = 0; i<consumed_set.index;i++)
    free(consumed_set[i].dist.elements);
  free(consumed_set.elements);
  for (int i = 0; i<stack.index;i++)
    free(stack[i].elements);
  free(stack.elements);
  free(point_set.elements);
  return top;
}





#endif
