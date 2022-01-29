"""
File adapted from https://github.com/pg2455/Hybrid-learn2branch
"""
import os
import argparse
import datetime
import numpy as np
import torch

from policy import Policy
from utilities import L2PDataset as Dataset
from utilities import load_batch_l2p as load_batch


def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def pretrain(model, dataloader):
    model.gcn_module.pre_train_init()
    i = 0
    while True:
        for batch in dataloader:

            (edge_indices, in_route_features,
            customer_features_1, customer_features_2,
            num_customerss, num_in_routess,
            routess, route_costss, selected_indices,
            dual_sums, adjusted_dualss, Gs, hs) = batch

            state = (edge_indices, in_route_features, customer_features_1,
                     num_in_routess, num_customerss)

            if not model.gcn_module.pre_train(state):
                break

        res = model.gcn_module.pre_train_next()
        if res is None:
            break
        else:
            layer = res

        i += 1

    return i

def process(model, dataloader, optimizer=None):

    mean_loss = 0
    mean_kacc = np.zeros(len(model.ks))

    n_samples_processed = 0
    for batch in dataloader:
        if optimizer:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = model(batch)

        batch_size = len(batch[-1])
        mean_loss += model.batch_ce_loss.detach().numpy() * batch_size
        mean_kacc += model.kacc.detach().numpy() * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-path',
        help='name of the directory contanining training and validation data',
        type=str,
        default="",
    )

    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr =  3 *1e-4
    max_epochs = 500
    patience = 15
    early_stopping = 30
    top_ks = [1, 10, 100, 1000]
    num_workers = 4

    ### LOG ###
    if not os.path.isdir('models'):
        os.mkdir('models')
    logfile = 'log.txt'

    ### NUMPY / TORCH SETUP ###
    rng = np.random.RandomState(0)
    torch.manual_seed(rng.randint(np.iinfo(int).max))

    ### SET-UP DATASET ###
    dir = args.data_path
    train_data = Dataset(f'{dir}/train.zarr')
    valid_data = Dataset(f'{dir}/valid.zarr')


    pretrain_data = train_data
    pretrain_loader = torch.utils.data.DataLoader(pretrain_data, batch_size=pretrain_batch_size,
                            shuffle = False, num_workers = num_workers, collate_fn = load_batch)

    model = Policy(ks=top_ks)

    model.save_state('models/starting.pkl')

    ### TRAINING LOOP ###
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience, verbose=True)

    best_loss = np.inf
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        # TRAIN
        if epoch == 0:
            n = pretrain(model=model, dataloader=pretrain_loader)
            log(f"PRETRAINED {n} LAYERS", logfile)
            model.save_state('models/pretrained.pkl')
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                    shuffle = False, num_workers = num_workers, collate_fn = load_batch)
            train_loss, train_kacc = process(model, train_loader, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_ks, train_kacc)]), logfile)

        # TEST
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size,
                                shuffle = False, num_workers = num_workers, collate_fn = load_batch)
        valid_loss, valid_kacc = process(model, valid_loader, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_ks, valid_kacc)]), logfile)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state('models/best.pkl')
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)
        scheduler.step(valid_loss)

    model.restore_state('models/best.pkl')
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size,
                        shuffle = False, num_workers = num_workers, collate_fn = load_batch)
    valid_loss, valid_kacc = process(model, valid_loader, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_ks, valid_kacc)]), logfile)



