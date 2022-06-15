import argparse
from csv import writer
import datetime
import os
from typing import OrderedDict

import torch
import torch.optim as optim

from tqdm import tqdm

import common.utils as utils
from common.manager import Manager
import dataset.data_loader as data_loader
from models.lr_scheduler import CosineAnnealingRestartLR
from models.edvr_model import EDVRModel
from models.edvr_net import EDVR

from loss.losses import compute_losses

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

def train(model, manager):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # loss status and val/test status initial
    manager.reset_loss_status()

    # set model to training mode
    torch.cuda.empty_cache()
    model.net.train()

    # Use tqdm for progress bar
    with tqdm(total=len(manager.dataloaders['train'])) as t:
        for i, data_batch in enumerate(manager.dataloaders['train']):
            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)

            # infor print
            print_str = manager.print_train_info()


            # compute model output and loss
            # output_batch = model(data_batch)
            model.feed_data(data_batch)

            # optimize parameters

            output_batch = model.net(model.lq)

            loss = compute_losses(data_batch, output_batch, manager.params)

            # update loss status and print current loss and average loss
            manager.update_loss_status(loss=loss, split="train")

            # clear previous gradients, compute gradients of all variables wrt loss
            manager.optimizer.zero_grad()
            loss['total'].backward()
            # performs updates using calculated gradients
            manager.optimizer.step()

            # manager.writer.add_scalar("Loss/train", manager.loss_status['total'].val, manager.step)
            # manager.logger.info("Loss/train: step {}: {}".format(manager.step, manager.loss_status['total'].val))
            # update step: step += 1
            manager.update_step()

            t.set_description(desc=print_str)
            t.update()

    manager.scheduler.step()
    # update epoch: epoch += 1
    manager.update_epoch()

def train_and_evaluate(model, manager):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
    """

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        manager.load_checkpoints()

    for epoch in range(manager.params.num_epochs):
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, manager)

        # Evaluate for one epoch on validation set
        # evaluate(model, manager)

        # Save best model weights accroding to the params.major_metric
        manager.check_best_save_last_checkpoints(latest_freq=5)

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

    dataloaders, datasamplers = data_loader.fetch_dataloader(params)

    # model = EDVR(num_in_ch=params.network['num_in_ch'],
    #            num_out_ch=params.network['num_out_ch'],
    #            num_feat=params.network['num_feat'],
    #            num_frame=params.network['num_frame'],
    #            deformable_groups=params.network['deformable_groups'],
    #            num_extract_block=params.network['num_extract_block'],
    #            num_reconstruct_block=params.network['num_reconstruct_block']
    #            )

    model = EDVRModel(params)

    # if params.cuda:
    #     print("Loading model to GPU...")
    #     model = model.cuda()
    #     model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if params.optim_type == "Adam":
        optimizer = optim.Adam(model.net.parameters(), lr=4e-4, betas=(0.9, 0.999))
    
    # TODO: spearate dcn params and normal params for different lr (Ref: BasicSR )

    scheduler = CosineAnnealingRestartLR(optimizer=optimizer,
                                         periods=params.scheduler["periods"],
                                         restart_weights=params.scheduler["restart_weights"],
                                         eta_min=params.scheduler["eta_min"]
                                        )


    manager = Manager(model=model, 
                      optimizer=optimizer, 
                      scheduler=scheduler, 
                      params=params, 
                      dataloaders=dataloaders
                      # writer=writer
                      # logger=logger
                      )

    train_and_evaluate(model, manager)