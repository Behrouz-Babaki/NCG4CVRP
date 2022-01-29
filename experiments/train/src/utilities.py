
import zarr
import numpy as np
import torch


class L2PDataset(object):
    
    def __init__(self, archive_path):
        self.data = zarr.open(archive_path, mode='r')
        
    def __len__(self):
        return len(self.data['bases'])

    def __getitem__(self, index):

        edges = self.data['bases/%d/edges'%index].__array__()
        col_features = self.data['bases/%d/column_features'%index].__array__()
        row_features = self.data['bases/%d/row_features'%index].__array__()
        selected_index = self.data['bases/%d/selected_index'%index].__array__().item()
        duals = self.data['bases/%d/duals'%index].__array__()
        adjusted_duals = self.data['bases/%d/adjusted_duals'%index].__array__()
                

        instance_index = self.data['bases/%d/instance_index'%index].__array__().item()
        capacity = self.data['instances/%d/capacity'%instance_index].__array__().item()
        demands = self.data['instances/%d/demands'%instance_index].__array__()
        locations = self.data['instances/%d/locations'%instance_index].__array__()
        routes = self.data['instances/%d/routes'%instance_index].__array__()
        route_costs = self.data['instances/%d/route_costs'%instance_index].__array__()
        in_route_costs = col_features[:,0]
        
        G_ = []
        num_customers = locations.shape[0]
        row_ind, col_ind = edges
        split_indices = np.where(np.diff(col_ind, prepend=np.nan))[0][1:] 
        column_routes = np.split(row_ind, split_indices)        
        for route in column_routes:
            r = np.zeros(num_customers)
            r[route] = 1
            G_.append(r)
        G_ = np.array(G_)
        h_ = col_features[:,0]
        G = np.vstack((G_, -np.eye(num_customers)))
        h = np.concatenate((h_, np.zeros(num_customers)))

        return (edges, duals, adjusted_duals, 
                capacity, demands, locations, 
                in_route_costs, routes, route_costs, 
                selected_index, G, h)  


def load_batch_l2p(batch):
    
    (edgess, dualss, adjusted_dualss,
     capacities, demandss, locationss,
     in_route_costss, routess, route_costss, 
     selected_indices, Gs, hs) = zip(*batch)

    num_customerss = [d.shape[0] for d in demandss]
    num_in_routess = [irc.shape[0] for irc in in_route_costss]
        
    in_route_features = np.concatenate(in_route_costss)[:,np.newaxis]
    customer_features_1 = np.concatenate(dualss)[:,np.newaxis]    
    adjusted_dualss = np.concatenate(adjusted_dualss)
    dual_sums = [d.sum() for d in dualss]
    
    demandss = np.concatenate(demandss)
    locationss = np.vstack([np.vstack(l) for l in locationss])
    capacities_repeated = np.repeat(capacities, num_customerss)
    customer_features_2 = np.column_stack((demandss, locationss, capacities_repeated))    

    
    shifts = np.cumsum([
        [0] + num_customerss[:-1],
        [0] + num_in_routess[:-1]
    ], axis=1)  
    
    edge_indices = np.concatenate([e + shifts[:,i,np.newaxis] for i, e in enumerate(edgess)], axis=1)


    num_customerss = torch.as_tensor(num_customerss, dtype=torch.int)
    num_in_routess = torch.as_tensor(num_in_routess, dtype=torch.int)    
    in_route_features = torch.as_tensor(in_route_features, dtype=torch.float)
    customer_features_1 = torch.as_tensor(customer_features_1, dtype=torch.float)
    customer_features_2 = torch.as_tensor(customer_features_2, dtype=torch.float)
    edge_indices = torch.as_tensor(edge_indices, dtype=torch.long)
    adjusted_dualss = torch.as_tensor(adjusted_dualss, dtype=torch.float)
    
    routess = [torch.as_tensor(r, dtype=torch.float) for r in routess]
    route_costss = [torch.as_tensor(c, dtype=torch.float) for c in route_costss]

    return (edge_indices, in_route_features, 
            customer_features_1, customer_features_2, 
            num_customerss, num_in_routess,
            routess, route_costss, selected_indices, 
            dual_sums, adjusted_dualss, Gs, hs)

