#!/usr/bin/env python

import argparse
import torch
import numpy as np

from utilities import L2PDataset as Dataset
from utilities import load_batch_l2p as load_batch
from baseline_policy import Policy as BaselinePolicy
from mlp_policy import Policy as MlpPolicy
from gcn_policy import Policy as GcnPolicy
from se_policy import Policy as SePolicy
from gcn_se_policy import Policy as GcnSePolicy
from mse_policy import Policy as MsePolicy
from kld_policy import Policy as KldPolicy

policy_map = {
    'baseline': BaselinePolicy,
    'mlp': MlpPolicy,
    'gcn': GcnPolicy,
    'se': SePolicy,
    'gcn-se': GcnSePolicy,
    'mse': MsePolicy,
    'kld': KldPolicy
}

def get_inference_model(model_type, model_file):
    Policy = policy_map[model_type]
    learned_model = Policy(learn_quadratic=False)
    learned_model.restore_state(model_file)
    return learned_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--model-type', required=True,
                        choices=('baseline', 'mlp', 'gcn', 'se', 'gcn-se', 'mse', 'kld'))
    parser.add_argument('--model-file', required=True)
    parser.add_argument('--result-file', required=True)
    args = parser.parse_args()

    top_ks = [1, 10, 100, 1000]
    batch_size = 128
    test_data = Dataset(args.test_data)
    num_workers = 4

    dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                            shuffle = False, num_workers = num_workers, collate_fn = load_batch)

    model = get_inference_model(args.model_type, args.model_file)
    mean_loss = 0
    mean_kacc = np.zeros(len(top_ks))

    n_samples_processed = 0
    for batch in dataloader:
        with torch.no_grad():
            loss = model(batch)

        batch_size = len(batch[-1])
        mean_loss += loss.detach().item() * batch_size
        mean_kacc += model.kacc.detach().numpy() * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    result_str = args.model_type + '\n'
    result_str += f"TEST LOSS: {mean_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_ks, mean_kacc)])
    with open(args.result_file, 'w') as f:
        print(result_str)
        print(result_str, file=f)
