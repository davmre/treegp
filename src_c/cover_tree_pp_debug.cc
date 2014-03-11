#include "cover_tree.hpp"
#include <limits>
#include <stdint.h>
#include <iostream>
#include <stdio.h>
using namespace std;

struct d_node {
  double dist;
  const node<pairpoint> *n;
};

inline double compare(const d_node *p1, const d_node* p2)
{
  return p1 -> dist - p2 -> dist;
}

#define SWAP(a, b)				\
  do						\
    {						\
      d_node tmp = * a;				\
      * a = * b;				\
      * b = tmp;				\
    } while (0)

static int distances=0;

static void halfsort (v_array<d_node > cover_set)
{
  if (cover_set.index <= 1)
    return;
  register d_node *base_ptr =  cover_set.elements;

  d_node *hi = &base_ptr[cover_set.index - 1];
  d_node *right_ptr = hi;
  d_node *left_ptr;

  while (right_ptr > base_ptr)
    {
      d_node *mid = base_ptr + ((hi - base_ptr) >> 1);

      if (compare ( mid,  base_ptr) < 0.)
	SWAP (mid, base_ptr);
      if (compare ( hi,  mid) < 0.)
	SWAP (mid, hi);
      else
	goto jump_over;
      if (compare ( mid,  base_ptr) < 0.)
	SWAP (mid, base_ptr);
    jump_over:;

      left_ptr  = base_ptr + 1;
      right_ptr = hi - 1;

      do
	{
	  while (compare (left_ptr, mid) < 0.)
	    left_ptr ++;

	  while (compare (mid, right_ptr) < 0.)
	    right_ptr --;

	  if (left_ptr < right_ptr)
	    {
	      SWAP (left_ptr, right_ptr);
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

static v_array<v_array<d_node> > get_cover_sets(v_array<v_array<v_array<d_node> > > &spare_cover_sets)
{
  v_array<v_array<d_node> > ret = pop(spare_cover_sets);
  while (ret.index < 101)
    {
      v_array<d_node> temp;
      push(ret, temp);
    }
  return ret;
}

inline bool shell(double parent_query_dist, double child_parent_dist, double upper_bound)
{
  return parent_query_dist - child_parent_dist <= upper_bound;
  //    && child_parent_dist - parent_query_dist <= upper_bound;
}

static int internal_k =1;
static void update_k(double *k_upper_bound, double upper_bound)
{
  double *end = k_upper_bound + internal_k-1;
  double *begin = k_upper_bound;
  for (;end != begin; begin++)
    {
      if (upper_bound < *(begin+1))
	*begin = *(begin+1);
      else {
	*begin = upper_bound;
	break;
      }
    }
  if (end == begin)
    *begin = upper_bound;
}
static double *alloc_k()
{
  return (double *)malloc(sizeof(double) * internal_k);
}
static void set_k(double* begin, double max)
{
  for(double *end = begin+internal_k;end != begin; begin++)
    *begin = max;
}

static double internal_epsilon =0.;
static void update_epsilon(double *upper_bound, double new_dist) {}
static double *alloc_epsilon()
{
  return (double *)malloc(sizeof(double));
}
static void set_epsilon(double* begin, double max)
{
  *begin = internal_epsilon;
}

static void update_unequal(double *upper_bound, double new_dist)
{
  if (new_dist != 0.)
    *upper_bound = new_dist;
}
static double* (*alloc_unequal)() = alloc_epsilon;
static void set_unequal(double* begin, double max)
{
  *begin = max;
}

static void (*update)(double *foo, double bar) = update_k;
static void (*setter)(double *foo, double bar) = set_k;
static double* (*alloc_upper)() = alloc_k;

inline void copy_zero_set(node<pairpoint>* query_chi, double* new_upper_bound,
			  v_array<d_node> &zero_set, v_array<d_node> &new_zero_set,
			  distfn<pairpoint>::Type distance,
			  const double* dist_params, void* dist_extra)
{
  new_zero_set.index = 0;
  d_node *end = zero_set.elements + zero_set.index;
  for (d_node *ele = zero_set.elements; ele != end ; ele++)
    {
      double upper_dist = *new_upper_bound + query_chi->max_dist;
      if (shell(ele->dist, query_chi->parent_dist, upper_dist))
	{
	  distances++;
	  double d = distance(query_chi->p, ele->n->p, upper_dist, dist_params, dist_extra);

	  if (d <= upper_dist)
	    {
	      if (d < *new_upper_bound)
		update(new_upper_bound, d);
	      d_node temp = {d, ele->n};
	      push(new_zero_set,temp);
	    }
	}
    }
}

static inline void copy_cover_sets(node<pairpoint>* query_chi, double* new_upper_bound,
			    v_array<v_array<d_node> > &cover_sets,
			    v_array<v_array<d_node> > &new_cover_sets,
			    int current_scale, int max_scale,
			    distfn<pairpoint>::Type distance,
			    const double* dist_params, void* dist_extra)
{
  for (; current_scale <= max_scale; current_scale++)
    {
      d_node* ele = cover_sets[current_scale].elements;
      d_node* end = cover_sets[current_scale].elements + cover_sets[current_scale].index;
      for (; ele != end; ele++)
	{
	  double upper_dist = *new_upper_bound + query_chi->max_dist + ele->n->max_dist;
	  if (shell(ele->dist, query_chi->parent_dist, upper_dist))
	    {
	      distances++;
	      double d = distance(query_chi->p, ele->n->p, upper_dist, dist_params, dist_extra);

	      if (d <= upper_dist)
		{
		  if (d < *new_upper_bound)
		    update(new_upper_bound,d);
		  d_node temp = {d, ele->n};
		  push(new_cover_sets[current_scale],temp);
		}
	    }
	}
    }
}



/*
  An optimization to consider:
  Make all distance evaluations occur in descend.

  Instead of passing a cover_set, pass a stack of cover sets.  The
  last element holds d_nodes with your distance.  The next lower
  element holds a d_node with the distance to your query parent,
  next = query grand parent, etc..

  Compute distances in the presence of the tighter upper bound.
 */

static inline void descend(const node<pairpoint>* query, double* upper_bound,
		      int current_scale,
		      int &max_scale, v_array<v_array<d_node> > &cover_sets,
		    v_array<d_node> &zero_set,
		    distfn<pairpoint>::Type distance,
		    const double* dist_params, void* dist_extra)
{
  d_node *end = cover_sets[current_scale].elements + cover_sets[current_scale].index;
  for (d_node *parent = cover_sets[current_scale].elements; parent != end; parent++)
    {
      const node<pairpoint> *par = parent->n;
      double upper_dist = *upper_bound + query->max_dist + query->max_dist;
      if (parent->dist <= upper_dist + par->max_dist && par->children)
	{
	  node<pairpoint> *chi = par->children;
	  if (parent->dist <= upper_dist + chi->max_dist)
	    {
	      if (chi->num_children > 0)
		{
		  if (max_scale < chi->scale)
		    max_scale = chi->scale;
		  d_node temp = {parent->dist, chi};
		  push(cover_sets[chi->scale], temp);
		}
	      else if (parent->dist <= upper_dist)
		{
		  d_node temp = {parent->dist, chi};
		  push(zero_set, temp);
		}
	    }
	  node<pairpoint> *child_end = par->children + par->num_children;
	  for (chi++; chi != child_end; chi++)
	    {
	      double upper_chi = *upper_bound + chi->max_dist + query->max_dist + query->max_dist;
	      if (shell(parent->dist, chi->parent_dist, upper_chi))
		{
		  distances++;
		  double d = distance(query->p, chi->p, upper_chi, dist_params, dist_extra);
		  if (d <= upper_chi)
		    {
		      if (d < *upper_bound)
			update(upper_bound, d);
		      if (chi->num_children > 0)
			{
			  if (max_scale < chi->scale)
			    max_scale = chi->scale;
			  d_node temp = {d, chi};
			  push(cover_sets[chi->scale],temp);
			}
		      else
			if (d <= upper_chi - chi->max_dist)
			  {
			    d_node temp = {d, chi};
			    push(zero_set, temp);
			  }
		    }
		}
	    }
	}
    }
}

static void brute_nearest(const node<pairpoint>* query,v_array<d_node> zero_set,
		   double* upper_bound,
		   v_array<v_array< node<pairpoint> > > &results,
		   v_array<v_array<d_node> > &spare_zero_sets,
		   distfn<pairpoint>::Type distance,
		   const double* dist_params, void* dist_extra)
{
  if (query->num_children > 0)
    {
      v_array<d_node> new_zero_set = pop(spare_zero_sets);
      node<pairpoint>* query_chi = query->children;
      brute_nearest(query_chi, zero_set, upper_bound, results, spare_zero_sets, distance, dist_params, dist_extra);
      double* new_upper_bound = alloc_upper();

      node<pairpoint> *child_end = query->children + query->num_children;
      for (query_chi++;query_chi != child_end; query_chi++)
	{
	  setter(new_upper_bound,*upper_bound + query_chi->parent_dist);
	  copy_zero_set(query_chi, new_upper_bound, zero_set, new_zero_set, distance, dist_params, dist_extra);
	  brute_nearest(query_chi, new_zero_set, new_upper_bound, results, spare_zero_sets, distance, dist_params, dist_extra);
	}
      free (new_upper_bound);
      new_zero_set.index = 0;
      push(spare_zero_sets, new_zero_set);
    }
  else
    {
      v_array< node<pairpoint> > temp;
      push(temp, *query);
      d_node *end = zero_set.elements + zero_set.index;
      for (d_node *ele = zero_set.elements; ele != end ; ele++)
	if (ele->dist <= *upper_bound)
	  push(temp, *(ele->n));
      push(results,temp);
    }
}

static void internal_batch_nearest_neighbor(const node<pairpoint> *query,
				     v_array<v_array<d_node> > &cover_sets,
				     v_array<d_node> &zero_set,
				     int current_scale,
				     int max_scale,
				     double* upper_bound,
				     v_array<v_array< node<pairpoint> > > &results,
				     v_array<v_array<v_array<d_node> > > &spare_cover_sets,
				     v_array<v_array<d_node> > &spare_zero_sets,
				     distfn<pairpoint>::Type distance,
				     const double* dist_params, void* dist_extra)
{
  printf("ITBNN idx (%d %d) pt1 (%.4f, %.4f) pt2 (%.4f, %.4f) query_scale %d current_scale %d max_scale %d upper_bound %.4f. %d csets[cs], %d zset, %d results, %d spare_csets, %d spare_zsets ", query->p.idx1, query->p.idx2, query->p.pt1[0], query->p.pt1[1], query->p.pt2[0], query->p.pt2[1], query->scale, current_scale, max_scale, *upper_bound, cover_sets[current_scale].index, zero_set.index, results.index, spare_cover_sets.index, spare_zero_sets.index);

  if (current_scale > max_scale) {// All remaining points are in the zero set.
    brute_nearest(query, zero_set, upper_bound, results, spare_zero_sets, distance, dist_params, dist_extra);
    printf("brute nearest.\n");
  }
  else
    if (query->scale <= current_scale && query->scale != 100)
      // Our query has too much scale.  Reduce.
      {
	printf("reducing scale.\n");
	node<pairpoint> *query_chi = query->children;
	v_array<d_node> new_zero_set = pop(spare_zero_sets);
	v_array<v_array<d_node> > new_cover_sets = get_cover_sets(spare_cover_sets);
	double* new_upper_bound = alloc_upper();

	node<pairpoint> *child_end = query->children + query->num_children;
	for (query_chi++; query_chi != child_end; query_chi++)
	  {
	    setter(new_upper_bound,*upper_bound + query_chi->parent_dist);
	    copy_zero_set(query_chi, new_upper_bound, zero_set, new_zero_set, distance, dist_params, dist_extra);
	    copy_cover_sets(query_chi, new_upper_bound, cover_sets, new_cover_sets,
			    current_scale, max_scale, distance, dist_params, dist_extra);
	    internal_batch_nearest_neighbor(query_chi, new_cover_sets, new_zero_set,
					    current_scale, max_scale, new_upper_bound,
					    results, spare_cover_sets, spare_zero_sets,
					    distance, dist_params, dist_extra);
	  }
	free (new_upper_bound);
	new_zero_set.index = 0;
	push(spare_zero_sets, new_zero_set);
	push(spare_cover_sets, new_cover_sets);
	internal_batch_nearest_neighbor(query->children, cover_sets, zero_set,
					current_scale, max_scale, upper_bound, results,
					spare_cover_sets, spare_zero_sets,
					distance, dist_params, dist_extra);
      }
    else // reduce cover set scale
      {
	printf("descending.\n");
	halfsort(cover_sets[current_scale]);
	descend(query, upper_bound, current_scale, max_scale,cover_sets, zero_set, distance, dist_params, dist_extra);
	cover_sets[current_scale++].index = 0;
	internal_batch_nearest_neighbor(query, cover_sets, zero_set,
					current_scale, max_scale, upper_bound, results,
					spare_cover_sets, spare_zero_sets, distance, dist_params, dist_extra);
      }
}

static void batch_nearest_neighbor(const node<pairpoint> &top_node, const node<pairpoint> &query,
			    v_array<v_array< node<pairpoint> > > &results,
			    distfn<pairpoint>::Type distance,
			    const double* dist_params, void* dist_extra)
{
  v_array<v_array<v_array<d_node> > > spare_cover_sets;
  v_array<v_array<d_node> > spare_zero_sets;

  v_array<v_array<d_node> > cover_sets = get_cover_sets(spare_cover_sets);
  v_array<d_node> zero_set = pop(spare_zero_sets);

  double* upper_bound = alloc_upper();
  setter(upper_bound,std::numeric_limits< double >::max());

  distances=0;
  distances++;
  double top_dist = distance(query.p, top_node.p, std::numeric_limits< double >::max(), dist_params, dist_extra);
  update(upper_bound, top_dist);
  d_node temp = {top_dist, &top_node};
  push(cover_sets[0], temp);

  internal_batch_nearest_neighbor(&query,cover_sets,zero_set,0,0,upper_bound,results,
				  spare_cover_sets,spare_zero_sets, distance, dist_params, dist_extra);

  free(upper_bound);
  push(spare_cover_sets, cover_sets);

  printf("total distances %d\n", distances);
  distances=0;

  for (int i = 0; i < spare_cover_sets.index; i++)
    {
      v_array<v_array<d_node> > cover_sets = spare_cover_sets[i];
      for (int j = 0; j < cover_sets.index; j++)
	free (cover_sets[j].elements);
      free(cover_sets.elements);
    }
  free(spare_cover_sets.elements);

  push(spare_zero_sets, zero_set);

  for (int i = 0; i < spare_zero_sets.index; i++)
    free(spare_zero_sets[i].elements);
  free(spare_zero_sets.elements);
}

void k_nearest_neighbor(const node<pairpoint> &top_node, const node<pairpoint> &query,
			v_array<v_array< node<pairpoint> > > &results, int k,
			distfn<pairpoint>::Type distance,
			const double* dist_params, void* dist_extra)
{
  internal_k = k;
  update = update_k;
  setter = set_k;
  alloc_upper = alloc_k;

  batch_nearest_neighbor(top_node, query,results, distance, dist_params, dist_extra);
}

void epsilon_nearest_neighbor(const node<pairpoint> &top_node, const node<pairpoint> &query,
			      v_array<v_array< node<pairpoint> > > &results, double epsilon,
			      distfn<pairpoint>::Type distance,
			      const double* dist_params, void* dist_extra)
{
  internal_epsilon = epsilon;
  update = update_epsilon;
  setter = set_epsilon;
  alloc_upper = alloc_epsilon;

  batch_nearest_neighbor(top_node, query,results, distance, dist_params, dist_extra);
}

void unequal_nearest_neighbor(const node<pairpoint> &top_node, const node<pairpoint> &query,
			      v_array<v_array< node<pairpoint> > > &results, distfn<pairpoint>::Type distance,
		    const double* dist_params, void* dist_extra)
{
  update = update_unequal;
  setter = set_unequal;
  alloc_upper = alloc_unequal;

  batch_nearest_neighbor(top_node, query, results, distance, dist_params, dist_extra);
}
