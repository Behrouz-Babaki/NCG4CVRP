#!/usr/bin/env python
# coding: utf-8

import gzip, pickle, os, random, argparse
import zarr
import numpy as np


def read_instance_info(samples):
    num_customers = samples['properties']['num_customers']
    
    capacity = samples['graph'].graph['capacity']
    demands = np.array([samples['graph'].nodes[i+1]['demand'] for i in range(num_customers)])
    distances = np.array(samples['graph'].graph['distance_matrix'])
    locations = []
    depot_location = np.array(samples['graph'].nodes['Sink']['location'])
    for i in range(num_customers):
        location = np.array(samples['graph'].nodes[i+1]['location'])
        locations.append(location - depot_location)    
    
    route_costs = []
    routes = []
    for k, v in samples['route_costs'].items():
        route_costs.append(v)
        route = np.zeros(num_customers)
        route[np.array(list(k))-1] = 1
        routes.append(route)

    route_costs = np.array(route_costs)
    routes = np.array(routes)
    
    return capacity, demands, locations, distances, routes, route_costs


def read_iteration_info(samples, routes):

    num_customers = samples['properties']['num_customers']

    graphs = []
    for info in samples['basis_info']:
        duals = info['duals']
        adjusted_duals = info['adjusted_duals']
        is_last = info['is_last']

        col_keys = ['cost', 'value', 'reduced_cost', 'is_basic']
        column_features = [[float(i[k]) for k in col_keys] for i in info['vinfo']]
        column_features = np.array(column_features)

        row_keys = ['dual', 'is_basic', 'slack', 'si_low', 'si_up']
        row_features = [[float(i[k]) for k in row_keys] for i in info['cinfo']]
        row_features = np.array(row_features)

        if is_last: 
            selected_index = -1
        else:
            query = np.zeros(num_customers)
            query[np.array(list(info['added_route']))-1] = 1
            selected_index = np.where(np.all(routes==query, axis=1))[0].item()    

        edges = np.zeros((2, 0), dtype=np.int64)
        for i, v in enumerate(info['vinfo']):
            row_indices = np.array(list(v['route'])) - 1
            col_indices = np.full(len(row_indices), i)
            edges = np.hstack((edges, np.vstack((row_indices, col_indices))))

        graphs.append((edges, column_features, row_features, selected_index, duals, adjusted_duals))

    return graphs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True, nargs='+')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--num-instances', type=int)
    parser.add_argument('--num-customers', type=int, default=21)
    args = parser.parse_args()


    root = zarr.open(args.outdir, mode='w')
    instances = root.create_group('instances')
    bases = root.create_group('bases')
    
    instance_id = 0
    basis_id = 0

    fpaths = []
    for idir in args.indir:
        fpaths += list(map(lambda x: os.path.join(idir, x), filter(lambda x: x.endswith('.pkl'), os.listdir(idir))))
    random.shuffle(fpaths)

    for fpath in fpaths:

        with gzip.open(fpath, 'rb') as f:
            samples = pickle.load(f)

        num_customers = samples['properties']['num_customers']
        if num_customers != args.num_customers:
            continue

        capacity, demands, locations, distances, routes, route_costs = read_instance_info(samples)
        graphs = read_iteration_info(samples, routes)

        ins = instances.create_group('%d'%instance_id)
        ins.create_dataset('capacity', data=capacity)
        ins.create_dataset('demands', data=demands)
        ins.create_dataset('distances', data=distances)
        ins.create_dataset('locations', data=locations)
        ins.create_dataset('routes', data=routes)
        ins.create_dataset('route_costs', data=route_costs)

        for i in range(len(graphs)):
            edges, column_features, row_features, selected_index, duals, adjusted_duals = graphs[i]
            b = bases.create_group('%d'%basis_id)
            b.create_dataset('duals', data=duals)
            b.create_dataset('adjusted_duals', data=adjusted_duals)
            b.create_dataset('edges', data=edges)
            b.create_dataset('column_features', data=column_features)
            b.create_dataset('row_features', data=row_features)
            b.create_dataset('selected_index', data=selected_index)
            b.create_dataset('instance_index', data=instance_id)
            basis_id += 1
        
        instance_id += 1

        if args.num_instances is not None and instance_id == args.num_instances:
            break
