#!/usr/bin/env python
# coding: utf-8

import os, sys
script_dir = os.path.dirname(__file__)
rg_path = os.path.join(script_dir, '..', '..', '..', 'route_generator')
sys.path.append(os.path.abspath(rg_path))

import pickle
import gzip
import argparse
import numpy as np
from gurobipy import Model, GRB, Column
from route_generator import get_route_costs
from trace_utils import ClarkWright, generate, RouteFinder, get_optimal_duals

def add_column(route, cost, model, constraints):
    column = Column()
    for cid in route:
        column.addTerms(1.0, constraints[cid])
    v = model.addVar(lb=0.0, ub=GRB.INFINITY,
                      obj=cost,
                      vtype=GRB.CONTINUOUS,
                      column=column)
    return v


def get_info(variables, constraints):
    
    var_info = []
    for r, v in variables.items():
        vinfo = dict()
        vinfo['route'] = r
        vinfo['cost'] = v.Obj
        vinfo['value'] = v.x
        vinfo['reduced_cost'] = v.RC
        vinfo['is_basic'] = (v.VBasis == 0)
        var_info.append(vinfo)
            
    num_customers = len(constraints)
    constraint_info = []
    for i in range(num_customers):
        c = constraints[i+1]
        cinfo = dict()
        cinfo['dual'] = c.Pi
        cinfo['is_basic'] = (c.CBasis == 0)
        cinfo['slack'] = c.Slack
        cinfo['si_low'] = c.SARHSLow
        if np.isinf(cinfo['si_low']) and cinfo['si_low'] < 0:
            cinfo['si_low'] = 0
        cinfo['si_up'] = c.SARHSUp
        if np.isinf(cinfo['si_up']) and cinfo['si_up'] > 0:
            cinfo['si_up'] = num_customers
        constraint_info.append(cinfo)

    return dict(vinfo=var_info, cinfo=constraint_info)


def cg_od(vrp, route_costs, cw_routes):

    
    num_customers = vrp.number_of_nodes() - 2

    rf = RouteFinder(num_customers, route_costs)
    optimal_duals = get_optimal_duals(num_customers, route_costs)
    optimal_duals = [optimal_duals[i+1] for i in range(num_customers)]

    master = Model('master')
    master.ModelSense = GRB.MINIMIZE
    master.params.OutputFlag = 0
    master.params.UpdateMode = 1
    constraints = dict()
    variables = dict()

    for i in range(num_customers):
        constraints[i+1] = master.addLConstr(0, GRB.GREATER_EQUAL, 1)

    
    basis_info = []
    current_routes = []
    for route in cw_routes:
        cost = route_costs[route]
        v = add_column(route, cost, master, constraints)
        variables[route] = v
        current_routes.append(route)
        
    done = False
    num_iters = 0
    while not done:
        num_iters += 1
        master.optimize()
        binfo = get_info(variables, constraints)
        binfo['duals'] = np.array([constraints[i+1].Pi for i in range(num_customers)])

        nonzero_routes = []
        for r in current_routes:
            if variables[r].x > 1e-3:
                nonzero_routes.append(r)
        
        zero_customers = []
        for i in range(num_customers):
            if constraints[i+1].slack > 1e-3:
                zero_customers.append(i+1)
        next_cost, next_route = rf.find(current_routes, nonzero_routes, zero_customers, master.ObjVal, optimal_duals)
        binfo['adjusted_duals'] = rf.duals
        next_route = frozenset((np.where(next_route==1)[0] + 1).tolist())
        if next_cost > -1e-3:
            done = True
            next_route = None
        else:
            cost = route_costs[next_route]
            v = add_column(next_route, cost, master, constraints)
            variables[next_route] = v
            current_routes.append(next_route)

        binfo['objective'] = master.ObjVal
        binfo['added_route'] = next_route
        binfo['is_last'] = done
        basis_info.append(binfo)        

    return basis_info

def main():

    demand_distribution_types = [
    'small_values_large_cv',
    'small_values_small_cv',
    'large_values_large_cv',
    'large_values_small_cv'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-customers', required=True, type=int)
    parser.add_argument('--r', required=True, type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--depot-type', choices=('central', 'eccentric', 'random'), default='central')
    parser.add_argument('--customer-positioning', choices=('clustered', 'random'), default='clustered')
    parser.add_argument('--demand-distribution', choices=range(4), default=1)
    args = parser.parse_args()

    demand_distribution_type = demand_distribution_types[args.demand_distribution]
    instance_info = dict()

    properties = {
        'num_customers': args.num_customers,
        'r': args.r,
        'depot_type': args.depot_type,
        'customer_positioninig': args.customer_positioning,
        'demand_distribution_type': args.demand_distribution
    }
    instance_info['properties'] = properties

    vrp = generate(args.num_customers, args.depot_type.upper(), args.customer_positioning.upper(), demand_distribution_type, args.r)
    CW = ClarkWright(vrp)
    cw_routes = CW.routes
    route_costs = get_route_costs(vrp)

    instance_info['graph'] = vrp
    instance_info['cw_routes'] = cw_routes
    instance_info['route_costs'] = route_costs
    instance_info['basis_info'] = cg_od(vrp, route_costs, cw_routes)
    
    with gzip.open(args.output, 'wb') as f:
        pickle.dump(instance_info, f)


if __name__ == '__main__':
    main()