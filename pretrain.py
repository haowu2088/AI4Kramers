"""
pretrain autoencoder model on the negative and candidate datasets.   
"""

# Standard Python Libraries
import pprint    # pretty printing Python data structures in easily-read way
import time      # accessing time-related functions

# Third-Party Packages
import torch    # PyTorch, a deep learning framework that provides tensor computation (like NumPy) with strong acceleration via GPU
from torch.utils.data import DataLoader
import torch.optim as optim    # for various optimization algorithms for neural networks
from torch.optim.lr_scheduler import MultiStepLR  # provides a way to adjust the learning rate based on epochs
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np    # for large multi-dimensional array and matrix processing
import matplotlib.pyplot as plt      
from accelerate import Accelerator    # simplifies running PyTorch models on any hardware configuration
from kogger import Logger

# Defined custom modules
import config as cfg    # Likely a custom configuration module where parameters specific to your project are stored, 
                        # such as paths, model settings, and training hyperparameters.
from dataset import PretrainCIFData    # Appears to be a custom module for handling your specific dataset format, 
                                       # particularly if working with CIF files for pre-training.
from dataset_helper import collate_pool  # Likely a custom helper function that deals with how batches of data are collated or combined when loaded via a DataLoader.
from model import CrystalGraph    # Likely refers to a custom-defined model, 
                                  # possibly a graph-based neural network, tailored for working with crystallographic data.
from utils import AverageMeter    # Suggests a utility module possibly containing a class or function 
                                  # to help in measuring and tracking average values (like loss averages) over time.


def pretrain(accelerator, model, train_loader, optimizer, scheduler, config, logger):
    # switch to train mode
    model.train()  

    batch_time  = AverageMeter()
    data_time   = AverageMeter()
    losses      = AverageMeter()
    train_loss  = []

    for epoch in range(config['start_epoch'], config['start_epoch']+config['epochs']):
        end = time.time()
        for batch_idx, (inputs, target, cif_id) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # input: list, len=4
            # label: [b, ]
            atom_fea, nbr_fea, nbr_fea_idx, degree, crystal_atom_idx = inputs
            # (1) accelerate distributed train with multi-gpus
            #loss = model.module.pretrain(atom_fea, nbr_fea, nbr_fea_idx, degree, crystal_atom_idx)  # [b, features]
            # (2) accelerate with single gpu for debug
            loss = model.pretrain(atom_fea, nbr_fea, nbr_fea_idx, degree, crystal_atom_idx)  # [b, features]

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            with torch.no_grad():
                losses.update(loss.item(), target.shape[0])
                train_loss.append(losses.val)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            accelerator.wait_for_everyone()
            if accelerator.is_main_process and (epoch % config['log_epoch_freq'] == 0 or epoch == 1) and batch_idx % config['log_batch_freq'] == 0:
                logger.info('[Pretrain] Epoch [{}/{}] [{}/{}]\t BT {:.3f} ({:.3f})\t DT {:.3f} ({:.3f})\t Loss {:.4e} ({:.4e})\t [Save]'.format(epoch, config['start_epoch']+config['epochs']-1, batch_idx, len(train_loader), batch_time.val, batch_time.avg, data_time.val, data_time.avg, losses.val, losses.avg))
                output_dir = '{}-{}epoch'.format(config['pre_ckpt_path'], epoch)    # Wu: to change accordingly
                accelerator.save_state(output_dir=output_dir)    # Wu: notice this!

        scheduler.step()
        
        # Wu: saves the training loss every epoch, 
        # ensures you have a continuous record of the loss to monitor training progress, which is useful for debugging and analysis.
        if epoch % 1 == 0:    # Wu: always true
            np.save(config['restore_loss'], np.array(train_loss))
        
        # Wu: Saving Model State (in Gao's original code, the two lines below are commented)
        # "accelerator.is_main_process" ensures that the saving operation is only performed by the main process. 
        # This is important in distributed training environments where multiple processes are running in parallel, 
        # as you generally want only one process (usually the main one) to handle writing to disk to avoid conflicts or redundancy.
        if accelerator.is_main_process and epoch % config['save_epoch_freq'] == 0:
            accelerator.save_state(output_dir=config['pre_ckpt_path'])

    return np.array(train_loss)


def main():
    # load and set config
    args   = cfg.get_parser().parse_args()
    config = cfg.load_config(yaml_filename=args.filename)
    config = cfg.process_config(config)

    accelerator = Accelerator()

    logger = Logger('PID %d' % accelerator.process_index, file=config['log_file'])
    if accelerator.is_main_process:
        logger.info('Load config successfully!')
        logger.info(pprint.pformat(config))

    # load data
    if accelerator.is_main_process:
        logger.info('Load data...')
    dataset = PretrainCIFData(
        root_dir      = config['root_dir'],
        processed_dir = config['processed_dir'],
        radius        = config['radius'],
        max_num_nbr   = config['max_num_nbr'],
        dmin          = config['dmin'],
        step          = config['step'],
        logger        = logger
    )

    train_loader = DataLoader(
        dataset     = dataset,
        collate_fn  = collate_pool,
        batch_size  = config['batch_size'],
        shuffle     = config['shuffle'],
        num_workers = config['num_workers'],
        pin_memory  = True
    )

    # build model
    inputs, _, _                            = dataset[0]
    orig_atom_fea_len                       = inputs[0].shape[-1]
    nbr_fea_len                             = inputs[1].shape[-1]
    crystal_gnn_config                      = config['crystal_gnn_config']
    crystal_gnn_config['orig_atom_fea_len'] = orig_atom_fea_len
    crystal_gnn_config['nbr_fea_len']       = nbr_fea_len
    model = CrystalGraph(
        crystal_gnn_config  = crystal_gnn_config,
        head_output_dim     = config['head_output_dim'],
        drop_rate           = config['drop_rate'],
        decoder_sample_size = config['sample_size'],
        device              = accelerator.device
    )

    # pretrain
    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=config['weight_decay'])
    scheduler = MultiStepLR(optimizer, milestones=config['lr_milestones'], gamma=0.1)

    model, train_loader, optimizer, scheduler = accelerator.prepare(model, train_loader, optimizer, scheduler)

    # pretrain
    if accelerator.is_main_process:
        logger.info('Pretrain...')

    if config['continuous_pretrain']:
        accelerator.load_state(input_dir=config['pre_ckpt_path'])

    pretrain_loss = pretrain(
        accelerator  = accelerator,
        model        = model,
        train_loader = train_loader,
        optimizer    = optimizer,
        scheduler    = scheduler,
        config       = config,
        logger       = logger
    )

    if accelerator.is_main_process:
        np.save(config['restore_loss'], pretrain_loss)
        
        # plot pretrain loss
        fig = plt.figure()
        start = config['start_epoch']
        pretrain_idx = np.arange(start, config['epochs'] * len(train_loader) + start)
        plt.semilogy(pretrain_idx, pretrain_loss)
        plt.title('Pretrain Loss')
        plt.xlabel('Iteration')
        fig.savefig(config['figs_pretrain'])

        plt.show()

        logger.info('Done!')


if __name__ == '__main__':
    main()
