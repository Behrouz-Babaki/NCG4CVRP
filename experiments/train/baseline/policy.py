import torch
import torch.nn as nn
import torch.nn.functional as F
from qpf import QPFunction

class Policy(nn.Module):
    def __init__(self, learn_quadratic=True, ks=None, 
                 A_dim=5):
        
        super(Policy, self).__init__()
        
        self.learn_quadratic = learn_quadratic
        self.ks = ks
        
        params_dim = A_dim+1 if learn_quadratic else 1 
        
        self.params = torch.nn.parameter.Parameter(data=torch.zeros(21, params_dim, dtype=torch.float), 
                                              requires_grad=True)
        nn.init.xavier_uniform_(self.params)
        
    def forward(self, inputs):
        (edge_indices, in_route_features, 
         customer_features_1, customer_features_2, 
         num_customerss, num_in_routess,
         routess, route_costss, selected_indices, 
         dual_sums, adjusted_dualss, Gs, hs) = inputs
        
        batch_loss = 0
        self.kacc = None
        in_top_k = []
        
        batch_size = len(Gs)
        for i in range(batch_size):
            n_vars = Gs[i].shape[1]
            p = self.params[:,0]
            Q = 1e-3 * torch.eye(n_vars)
            if self.learn_quadratic:
                A = self.params[:,1:]
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