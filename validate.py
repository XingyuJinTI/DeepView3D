import sys
import os
import time
import pandas as pd
import torch
from options import options_train
import datasets
import models
from loggers import loggers
from util.util_print import str_error, str_stage, str_verbose, str_warning
from util import util_loadlib as loadlib


###################################################

print(str_stage, "Parsing arguments")
opt, unique_opt_params = options_train.parse()
# Get all parse done, including subparsers
print(opt)

###################################################

print(str_stage, "Setting device")
if opt.gpu == '-1':
    device = torch.device('cpu')
else:
    loadlib.set_gpu(opt.gpu)
    device = torch.device('cuda')
if opt.manual_seed is not None:
    loadlib.set_manual_seed(opt.manual_seed)

###################################################

print(str_stage, "Setting up models")
Model = models.get_model(opt.net)
model = Model(opt, logger)
model.to(device)
print(model)
print("# model parameters: {:,d}".format(model.num_parameters()))
"""
initial_epoch = 1
if opt.resume != 0:
    if opt.resume == -1:
        net_filename = os.path.join(logdir, 'checkpoint.pt')
    elif opt.resume == -2:
        net_filename = os.path.join(logdir, 'best.pt')
    else:
        net_filename = os.path.join(
            logdir, 'nets', '{epoch:04d}.pt').format(epoch=opt.resume)
    if not os.path.isfile(net_filename):
        print(str_warning, ("Network file not found for opt.resume=%d. "
                            "Starting from scratch") % opt.resume)
    else:
        additional_values = model.load_state_dict(net_filename, load_optimizer='auto')
        try:
            initial_epoch += additional_values['epoch']
        except KeyError as err:
            # Old saved model does not have epoch as additional values
            epoch_loss_csv = os.path.join(logdir, 'epoch_loss.csv')
            if opt.resume == -1:
                try:
                    initial_epoch += pd.read_csv(epoch_loss_csv)['epoch'].max()
                except pd.errors.ParserError:
                    with open(epoch_loss_csv, 'r') as f:
                        lines = f.readlines()
                    initial_epoch += max([int(l.split(',')[0]) for l in lines[1:]])
            else:
                initial_epoch += opt.resume
"""
###################################################

print(str_stage, "Setting up data loaders")
start_time = time.time()
dataset = datasets.get_dataset(opt.dataset)
dataset_vali = dataset(opt, mode='vali', model=model)
dataloader_vali = torch.utils.data.DataLoader(
    dataset_vali,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True,
    shuffle=False
)
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# val points: " + str(len(dataset_vali)))
print(str_verbose, "# val batches: " + str(len(dataloader_vali)))

###################################################

print(str_stage, "Validating")
model.validate(dataloader_vali)
