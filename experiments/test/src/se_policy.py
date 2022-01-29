import torch
import torch.nn as nn
import torch.nn.functional as F
from st import SetEncoder
from qpf import QPFunction

class Policy(nn.Module):
    def __init__(self, learn_quadratic=True, ks=None, 
                 se_emb_size=32,
                 A_dim=5):
        
        super(Policy, self).__init__()
        
        self.learn_quadratic = learn_quadratic
        self.ks = ks
        
        se_out_dim = A_dim+1 if learn_quadratic else 1
        self.se_module = SetEncoder(dim_input=5, 
                                    emb_size=se_emb_size,
                                    dim_output=se_out_dim)
        
    def forward(self, inputs):
        (edge_indices, in_route_features, 
         customer_features_1, customer_features_2, 
         num_customerss, num_in_routess,
         routess, route_costss, selected_indices, 
         dual_sums, adjusted_dualss, Gs, hs) = inputs
                
        customer_features = torch.cat((customer_features_1, customer_features_2), axis=1)
        customer_features_split = torch.split(customer_features, num_customerss.tolist(), dim=0)
        customer_features = torch.stack(customer_features_split)
        
        customer_embeddings = self.se_module(customer_features)
        customer_embeddings = torch.unbind(customer_embeddings) 
        
        
        batch_loss = 0
        self.kacc = None
        in_top_k = []
        
        batch_size = len(Gs)
        for i in range(batch_size):
            n_vars = Gs[i].shape[1]
            p = customer_embeddings[i][:,0]
            Q = 1e-3 * torch.eye(n_vars)
            if self.learn_quadratic:
                A = customer_embeddings[i][:,1:]
                Q += A @ A.T

            w = QPFunction()(Q, p, Gs[i], hs[i], dual_sums[i])

            reduced_costs = route_costss[i] - torch.mv(routess[i], w)

            selected_index = selected_indices[i]
            if selected_index < 0:
                selected_index = route_costss[i].shape[0]

            logits = torch.cat((-reduced_costs, torch.zeros(1)))
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([selected_index], dtype=torch.long))
            batch_loss += loss

            if self.ks is not None:
                topk_values = torch.topk(logits, max(self.ks), largest=True, sorted=True).values
                in_top_k.append([logits[selected_index] >= topk_values[i-1] for i in self.ks])                

        if self.ks is not None:
            self.kacc = torch.mean(torch.tensor(in_top_k, dtype=torch.float), axis=0)
            
        batch_loss /= batch_size        
        return batch_loss            
        
    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu'))) 