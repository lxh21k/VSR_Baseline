import argparse
from csv import writer
import datetime
import os

import torch
import torch.optim as optim

from tqdm import tqdm

import common.utils as utils
import dataset.data_loader as data_loader
from models.lr_scheduler import CosineAnnealingRestartLR

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', help="Directory containing params.json")

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Update args into params
    # vars(args) is a dict of all the arguments passed to the script
    print(vars(args))   
    params.update(vars(args))

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    dataloaders = data_loader.fetch_dataloader(params)

    exit()

    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)

    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingRestartLR(optimizer=optimizer,
                                         periods=params.scheduler.periods,
                                         restart_weights=params.scheduler.restart_weights,
                                         eta_min=params.scheduler.eta_min
                                        )

    manager = Manager(model=model, 
                      optimizer=optimizer, 
                      scheduler=scheduler, 
                      params=params, 
                      dataloaders=dataloaders,
                      # writer=writer
                      # logger=logger
                      )

    train_and_evaluate(model, manager)