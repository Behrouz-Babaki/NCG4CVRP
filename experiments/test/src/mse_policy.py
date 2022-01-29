import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import BGCN
from st import SetEncoder

class Policy(nn.Module):
    def __init__(self, learn_quadratic=True,
                 ks=None, 
                 gcn_emb_dim=32, 
                 gcn_out_dim=4,
                 se_emb_size=32):
        
        super(Policy, self).__init__()
        
        self.ks = ks
        self.gcn_module = BGCN(emb_size=gcn_emb_dim, 
                               cons_nfeats=1, var_nfeats=1, 
                               output_dim=gcn_out_dim)
        self.gcn_module.initialize_parameters()
        
        self.se_module = SetEncoder(dim_input=gcn_out_dim+4, 
                                    emb_size=se_emb_size,
                                    dim_output=1)
        
    def forward(self, inputs):
        (edge_indices, in_route_features, 
         customer_features_1, customer_features_2, 
         num_customerss, num_in_routess,
         routess, route_costss, selected_indices, 
         dual_sums, adjusted_dualss, Gs, hs) = inputs
        
        gcn_inputs = (edge_indices, in_route_features, customer_features_1, num_in_routess, num_customerss)        
        customer_embeddings = self.gcn_module(gcn_inputs)
        customer_features = torch.cat((customer_embeddings, customer_features_2), axis=1)        
        
        customer_features_split = torch.split(customer_features, num_customerss.tolist(), dim=0)
        customer_features = torch.stack(customer_features_split)
        
        customer_embeddings = self.se_module(customer_features)
        predicted_dualss = customer_embeddings.reshape(torch.numel(customer_embeddings))
        batch_loss = F.mse_loss(adjusted_dualss, predicted_dualss)

        
        ws = torch.unbind(customer_embeddings.squeeze()) 
        
        self.batch_ce_loss = 0
        self.kacc = None
        in_top_k = []
        
        batch_size = len(Gs)
        with torch.no_grad():
            for i in range(batch_size):
                reduced_costs = route_costss[i] - torch.mv(routess[i], ws[i])

                selected_index = selected_indices[i]
                if selected_index < 0:
                    selected_index = route_costss[i].shape[0]

                logits = torch.cat((-reduced_costs, torch.zeros(1)))
                ce_loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([selected_index], dtype=torch.long))
                self.batch_ce_loss += ce_loss

                if self.ks is not None:
                    topk_values = torch.topk(logits, max(self.ks), largest=True, sorted=True).values
                    in_top_k.append([logits[selected_index] >= topk_values[i-1] for i in self.ks])                

            if self.ks is not None:
                self.kacc = torch.mean(torch.tensor(in_top_k, dtype=torch.float), axis=0)
            
        self.batch_ce_loss /= batch_size        
        return batch_loss            

    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))         
