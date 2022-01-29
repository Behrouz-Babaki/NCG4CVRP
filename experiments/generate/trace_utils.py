# coding: utf-8

import numpy as np
from collections import namedtuple

from networkx import DiGraph
from gurobipy import Model, GRB, LinExpr
from clarke_wright import _ClarkeWright

VrpInstance = namedtuple('VrpInstance', 
                         ['num_locations',
                          'depot_location',
                          'customer_locations',
                          'capacity', 
                          'demands', 
                          'depot_id', 
                          'distances'])


class ClarkWright(object):

    def __init__(self, vrp_digraph):

        load_capacity = vrp_digraph.graph['capacity']
        cw = _ClarkeWright(vrp_digraph,
                           load_capacity)

        cw.run()
        self.costs = []
        for route in cw.best_routes:
            c = 0
            for i in range(len(route)-1):
                c += vrp_digraph.edges[route[i], route[i+1]]['cost']
            self.costs.append(c)

        self._routes = [r[1:-1] for r in cw.best_routes]

    @property
    def routes(self):
        return list(map(frozenset, self._routes))

def instance_to_graph(vrp_instance):

    G = DiGraph()
    G.graph['capacity'] = vrp_instance.capacity
    G.graph['distance_matrix'] = vrp_instance.distances

    for i in range(1, vrp_instance.num_locations):
        G.add_edge('Source', i, cost=vrp_instance.distances[0][i])
        G.add_edge(i, 'Sink', cost=vrp_instance.distances[i][0])

    for i in range(1, vrp_instance.num_locations):
        for j in range(i, vrp_instance.num_locations):
            G.add_edge(i, j, cost=vrp_instance.distances[i][j])
            G.add_edge(j, i, cost=vrp_instance.distances[j][i])

    for i in range(1, vrp_instance.num_locations):
        G.nodes[i]['demand'] = vrp_instance.demands[i]
        G.nodes[i]['location'] = vrp_instance.customer_locations[i-1]

    G.nodes['Source']['demand'] = 0
    G.nodes['Sink']['demand'] = 0
    G.nodes['Source']['location'] = vrp_instance.depot_location
    G.nodes['Sink']['location'] = vrp_instance.depot_location

    G.graph['n_res'] = 2
    for edge in G.edges(data=True):
        i, j, data = edge
        demand_head_node = G.nodes[j]['demand']
        data['res_cost'] = np.array([1, demand_head_node])

    return G

def generate(n, depot_type, customer_positioning, demand_distribution_type, r):

    np.random.seed()

    if depot_type == 'CENTRAL':
        depot_location = (500, 500)
    elif depot_type == 'ECCENTRIC':
        depot_location = (0, 0)
    elif depot_type == 'RANDOM':
        depot_location = tuple(np.random.choice(1001, 2))

    possible_locations = [(i, j) for i in range(1001) for j in range(1001)]
    possible_locations.remove(depot_location)

    if customer_positioning == 'RANDOM':
        customer_locations_indices = np.random.choice(1002000, n, replace=False)
        customer_locations = [possible_locations[i] for i in customer_locations_indices]
    elif customer_positioning == 'CLUSTERED':
        n_clusters = 3
        seed_locations_indices = np.random.choice(len(possible_locations), n_clusters, replace=False)

        seed_locations = []
        for i in seed_locations_indices:
            l = possible_locations[i]
            seed_locations.append(l)
            possible_locations.remove(l)

        possible_locations = np.array(possible_locations)
        seed_locations = np.array(seed_locations)
        diff = possible_locations[:, np.newaxis] - seed_locations
        distances = np.linalg.norm(diff, axis=2)
        probs = np.sum(np.exp(-distances/40), axis=1)
        probs = probs / probs.sum()

        customer_locations_indices = np.random.choice(len(possible_locations), n-n_clusters, p=probs, replace=False)
        customer_locations = list(map(tuple, possible_locations[customer_locations_indices].tolist()))
        customer_locations += list(map(tuple, seed_locations))


    demand_distribution_map = {
        'small_values_large_cv': (1, 10),
        'small_values_small_cv': (5, 10),
        'large_values_large_cv': (1, 100),
        'large_values_small_cv': (50, 100)
    }


    lo, hi = demand_distribution_map[demand_distribution_type]
    demands = np.random.randint(lo, hi+1, n).tolist()


    capacity = int(np.ceil(r*sum(demands)/n))

    num_locations = n+1
    depot_id = 0
    demands = [0] + demands
    coords = np.array([depot_location] + customer_locations)
    distances = np.linalg.norm(coords - coords[:,np.newaxis], axis=2)
    distances = np.around(distances, decimals=1).tolist()    

    vrp_instance = VrpInstance(num_locations, depot_location, customer_locations,
                               capacity, demands, 0, distances)
    return instance_to_graph(vrp_instance)





class DualFinder(object):
    def __init__(self, 
                 num_customers, 
                 route_costs, 
                 current_routes, 
                 nonzero_routes, 
                 zero_customers, 
                 objective):
        
        self.num_customers = num_customers

        self.model = Model('dual')
        self.model.ModelSense = GRB.MINIMIZE
        self.model.params.OutputFlag = 0
        self.model.params.UpdateMode = 1

        self.variables = dict()

        for i in range(num_customers):
            if i + 1 in zero_customers:
                ub = 0.0
            else:
                ub = GRB.INFINITY
            self.variables[i+1] = self.model.addVar(lb=0.0, ub=ub, vtype=GRB.CONTINUOUS)

        for r in current_routes:
            if r in nonzero_routes:
                rel = GRB.EQUAL
            else:
                rel = GRB.LESS_EQUAL
            self.model.addLConstr(sum(self.variables[i] for i in r), rel, route_costs[r])

        self.model.addLConstr(sum(self.variables.values()), GRB.EQUAL, objective)
            
        self.model.update()
        
    def find(self, l):
        
        obj = 0.5 * sum(self.variables[i+1]*self.variables[i+1] for i in range(self.num_customers))
        for i in range(self.num_customers):
            obj -= self.variables[i+1] * l[i]
        self.model.setObjective(obj)
        
        self.model.update()
        self.model.optimize()
        
        solution = [self.variables[i+1].x for i in range(self.num_customers)]
        return solution
        

class RouteFinder(object):
    def __init__(self, 
                 num_customers, 
                 route_costs):
        
        self.num_customers = num_customers
        self.route_costs = route_costs
        routes = []
        costs = []
        for r, c in route_costs.items():
            row = [0] * num_customers
            for i in r:
                row[i-1] = 1
            routes.append(row)
            costs.append(c)        
            
        self.routes = np.array(routes)
        self.costs = np.array(costs)
        
    def find(self, current_routes, nonzero_routes, zero_customers, objective, l):
        dual_finder = DualFinder(self.num_customers, self.route_costs, 
                                 current_routes, nonzero_routes, 
                                 zero_customers, objective)

        duals = dual_finder.find(l)
        self.duals = np.array(duals)
        
        reduced_costs = self.costs - np.dot(self.routes, self.duals)
        min_index = np.argmin(reduced_costs)
        return reduced_costs[min_index], self.routes[min_index]

def get_optimal_duals(num_customers, route_costs):

    model = Model()
    model.ModelSense = GRB.MINIMIZE
    model.params.OutputFlag = 0
    model.params.UpdateMode = 1

    variables = dict()
    for k, v in route_costs.items():
        variables[k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, obj=v)

    constraints = dict()
    for i in range(num_customers):
        expr = LinExpr()
        for r in variables:
            if i + 1 in r:
                expr.add(variables[r])
        constraints[i+1] = model.addLConstr(expr, GRB.GREATER_EQUAL, 1)

    model.optimize()

    optimal_duals = {k: c.Pi for k, c in constraints.items()}
    return optimal_duals