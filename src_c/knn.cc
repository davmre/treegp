#include "cover_tree.h"
using namespace std;
// Compute the k nearest neighbors

int main(int argc, char *argv[])
{
  int k = atoi(argv[1]);
  vector<point> set_of_points = parse_points(fopen(argv[2],"r"));
  vector<point> set_of_queries = parse_points(fopen(argv[3],"r"));

  node top = batch_create(set_of_points, distance_bounded);
  node top_query = batch_create(set_of_queries, distance_bounded);


  v_array<v_array<point> > res;
  k_nearest_neighbor(top,top_query,res,k, distance_bounded);

  printf("Printing results\n");
  for (int i = 0; i < res.index; i++)
    {
      for (int j = 0; j<res[i].index; j++)
	print(res[i][j]);
      printf("\n");
    }
  printf("results printed\n");
}
