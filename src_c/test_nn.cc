#include "cover_tree.h"
#include <limits.h>
#include <values.h>
#include <stdint.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
using namespace std;

float diff_timeval(timeval t1, timeval t2)
{
  return (float) (t1.tv_sec - t2.tv_sec) + (t1.tv_usec - t2.tv_usec) * 1e-6;
}

float diff_clock(clock_t t1, clock_t t2)
{
  return (float) (t1 - t2) / (float) CLOCKS_PER_SEC;
}

int compare(const void* p1, const void* p2)
{
  if (p1<p2)
    return -1;
  else
    return 1;
}

int main(int argc, char *argv[])
{
  timeval start;
  //  clock_t start_clock = clock();
  gettimeofday(&start, NULL);
  //v_array<point> v = make_same(10000);
  if (argc <2 )
    {
      cout << "usage: test_nn <k> <dataset>" << endl;
      exit(1);
    }
  int k = atoi(argv[1]);
  FILE* fd = fopen(argv[2],"r");
  vector<point> v = parse_points(fd);
  //printf("point length = %i\n",v[0].index);
  //printf("first point = \n");
  //print(v.elements[0]);

  timeval parsed;
  clock_t parsed_clock = clock();
  gettimeofday(&parsed,NULL);
  printf("parse in %f seconds\n",diff_timeval(parsed,start));

  node top = batch_create(v, distance_bounded);
  timeval created;
  //  clock_t created_clock = clock();
  gettimeofday(&created, NULL);
  printf("created in %f seconds\n",diff_timeval(created,parsed));


  v_array<v_array<point> > res;
  k_nearest_neighbor(top,top,res,k, distance_bounded);


  timeval queried;
  clock_t queried_clock = clock();
  gettimeofday(&queried, NULL);
  int thresh=MAXINT;
  if (1e10 / v.size()< v.size())
    thresh = (int) 1e10 / v.size();

  if (thresh < 10)
    thresh = 10;

  v_array<v_array<point> > brute_neighbors;
  for (int i=0; i < res.index  && i < thresh; i++) {
    point this_point = res[i][0];
    float upper_dist[k];
    point min_points[k];
    for (int j=0; j<k; j++)
      {
	upper_dist[j] = MAXFLOAT;
	min_points[j] = this_point;
      }
    for (vector<point>::size_type j=0; j < v.size(); j++) {
      float dist = distance_bounded (this_point, v[j], upper_dist[0]);
      if (dist < upper_dist[0]) {
	int l=0;
	for (;l<k-1; l++)
	  {
	    if (dist < upper_dist[l+1])
	      {
		upper_dist[l] = upper_dist[l+1];
		min_points[l] = min_points[l+1];
	      }
	    else {
	      upper_dist[l] = dist;
	      min_points[l] = v[j];
	      break;
	    }
	  }
	if (l == k-1)
	  {
	    upper_dist[l] = dist;
	    min_points[l] = v[j];
	  }
      }
    }
    v_array<point> us;
    push(us,this_point);
    for (int j = 0; j<k; j++){
      push(us,min_points[j]);
    }
    push(brute_neighbors,us);
    }
  timeval bruted;
  clock_t bruted_clock = clock();
  gettimeofday(&bruted, NULL);
  float qp,bq;
  if (true) //diff_clock(queried_clock,parsed_clock) < 1)
    qp = diff_timeval(queried,parsed);
  else
    qp = diff_clock(queried_clock,parsed_clock);
  if (true) //diff_clock(bruted_clock,queried_clock) < 1)
    bq = diff_timeval(bruted,queried);
  else
    bq = diff_clock(bruted_clock,queried_clock);
  printf("%s\t",argv[2]);
  printf("%d\t",(int)v.size());
  printf("%f\t",qp);
  if (v.size() <= (std::vector<int>::size_type)thresh)
    {
      printf("%f\t",bq);
      printf("%f\n",bq / qp);
    }
  else
    {
      float mul = v.size() / thresh;
      printf("%f(*)\t",bq*mul);
      printf("%f(*)\n",bq / qp * mul);
    }
  for (int i=0; i < brute_neighbors.index; i++) {
    point this_point = brute_neighbors[i][0];
    for (int j = 1; j < brute_neighbors[i].index; j++)
      {
	int flag = 0;
	point this_neighbor = brute_neighbors[i][j];
	float brute_distance = distance_bounded (this_neighbor, this_point, MAXFLOAT);
	for (int l = 1; l < res[i].index; l++)
	  {
	    if (brute_distance == distance_bounded(res[i][l],this_point,MAXFLOAT))
	      {
		flag = 1;
		break;
	      }
	  }
	if (flag == 0)
	  {
	    printf(" distances unequal %f\n", brute_distance);
	    printf("point         = ");print(this_point);
	    printf("brute neighbor = "); print(this_neighbor);
	    printf("our_neighbors = \n");
	    for (int l = 1; l < res[i].index; l++)
	      {
		printf("%f = distance, point = ",distance_bounded(res[i][l],this_point,MAXFLOAT));
		print(res[i][l]);
	      }
	  }
      }
  }
}
