#include "cover_tree.h"
#include "vector_mult.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <string.h>
using namespace std;

vector<float> read_floats(std::istream& source) {
  vector<float> parsed;
  std::string line;
  float f;
  while(getline(source, line))
    {
      f = ::atof(line.c_str());
      parsed.push_back(f);
    }
  return parsed;
}

int main(int argc, char *argv[])
{
  std::ifstream source(argv[1]);
  vector<float> v = read_floats(source);
  vector<point> set_of_points = parse_points(fopen(argv[2],"r"));
  vector<point> query_pt_arr = parse_points(fopen(argv[3],"r"));
  point query_pt = query_pt_arr[0];


  int n = (int)set_of_points.size();
  int vn = (int)v.size();
  int qn = (int)query_pt_arr.size();
  printf("n=%d\n", n);
  printf("vn=%d\n", vn);
  printf("qn=%d\n", qn);
  if (n != vn || qn != 1) {
    exit(1);
  }

  VectorTree tree = VectorTree(set_of_points, "pair");
  tree.set_v(0, v);
  float ws = tree.weighted_sum(0, query_pt, 0.0001, "se");



  printf("fcalls: %d\n", tree.fcalls);
  printf("weighted sum is %f\n", ws);
}
