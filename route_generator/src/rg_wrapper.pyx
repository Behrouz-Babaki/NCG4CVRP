from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.pair cimport pair

cdef extern from "rgen.hpp":
  vector[pair[set[int], double]] _get_route_costs(int, int, const vector[int]&, const vector[vector[double]]&)

def get_route_costs(vrp_graph):

  num_locations = vrp_graph.number_of_nodes() - 1
  distances = vrp_graph.graph['distance_matrix']
  demands = [0] + [vrp_graph.nodes[i]['demand'] for i in range(1, num_locations)]
  capacity = vrp_graph.graph['capacity']

  cost_tuples = _get_route_costs(num_locations, capacity, demands, distances)
  route_costs = {frozenset(i.first):i.second for i in cost_tuples}
  return route_costs

