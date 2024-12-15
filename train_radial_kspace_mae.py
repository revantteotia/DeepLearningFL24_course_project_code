
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.schedulers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import math
from dataloaders.imagenet_radial_kspace_dataset_hdf5 import ImagenetRadialKspaceDatasetHDF5
from models.radial_kspace_mae import RadialKspaceMaskedAutoencoder 

import os

from accelerate import Accelerator
from accelerate.logging import get_logger

import logging 

os.environ["ACCELERATE_LOG_LEVEL"] = "INFO"

# configuring the logger to display log message 
# along with log level and time  
logging.basicConfig(filename="imagenet_logscaled_radial_kspace_mae_run_logs.log", 
                    format='%(asctime)s: %(levelname)s: %(message)s', 
                    level=logging.INFO) 

import argparse

logger = get_logger(__name__, "INFO") # NOTE: LOGGER NOT USED
logger.setLevel(logging.INFO)

# Define number of epochs, bs, etc...
num_epochs = 300 # can be overwritten by args
# warmup_steps = 10000
warmup_epochs = 15
# grad_clip = 0.1
max_grad_norm= 1
batch_size_per_gpu = 256 # 512
log_interval = 10
lr=1.5e-4
checkpoint_interval = 500 # total training steps are 320,000 for 4 gpu and 300 batch size
gradient_accumulation_steps = 4 # we will heve 128*4*8 effective batch size 4096
dataloader_workers= 16
# pin_memory = True
pin_memory = True
# prefetch_factor=2
prefetch_factor=None
num_spokes = 256
data_root_dir = "/gpfs/scratch/rt2741/current_training_datasets/imagenet-large-radial-kspace-256-spokes-hdf5" 


# Set the seed
seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():

    parser = argparse.ArgumentParser(description="Example argparse usage")
    parser.add_argument('--model_depth', type=int, default=12, help='num of transformer layers (default: 12)')
    parser.add_argument('--decoder_model_depth', type=int, default=8, help='num of transformer layers (default: 12)')
    parser.add_argument('--embed_dim', type=int, default=768, help='transformer emb dim')
    parser.add_argument('--decoder_embed_dim', type=int, default=512, help='transformer emb dim')
    parser.add_argument('--decoder_num_heads', type=int, default=8, help='transformer num of attention heads')
    parser.add_argument('--num_heads', type=int, default=12, help='transformer num of attention heads')
    parser.add_argument('--num_epochs', type=int, default=300, help='number_of_epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='base learning rate')
    parser.add_argument('--dropout', type=float, default=0.01, help='dropout rate (default: 0.01)')
    parser.add_argument('--mask_ratio', type=float, default=0.90, help='mask ratio')
    parser.add_argument('--attn_dropout', type=float, default=0.01, help='attn layers dropout rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
    # parser.add_argument('--norm', type=str, default="none", help='normalize global complex matrix, or mag only or imag, real separately')
    parser.add_argument('--channels', type=str, default="real_imag", help='use real-imag or mag-phase')
    parser.add_argument('--log_scale_amplitude', action='store_true')
    parser.add_argument('--log_scale_freq', action='store_true')
    parser.add_argument('--z_normalize_spokes', action='store_true')
    parser.add_argument('--min_max_normalize', action='store_true')
    parser.add_argument('--use_half_spoke', action='store_true', help="since kspace from real images is symmetric. We can just use half spoke length")

    parser.add_argument('--loss_type', type=str, default="MSE", help='use real-imag or mag-phase')
  
    parser.add_argument('--exp_name', type=str, default="mae_spokes", help='name of wandb exp')
    parser.add_argument('--do_initial_eval', action='store_true')
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    args = parser.parse_args()

    EXPERIMENT_NAME = args.exp_name + f"loss-{args.loss_type}_" + f"msk_ratio-{args.mask_ratio}_" + f"use_half_spoke-{args.use_half_spoke}_" + f"log_scale_amplitude-{args.log_scale_amplitude}_" + f"min_max_normalize-{args.min_max_normalize}_" 
    num_epochs = args.num_epochs
    lr=args.lr

    accelerator = Accelerator(log_with="wandb", 
                              gradient_accumulation_steps=gradient_accumulation_steps, 
                            #   mixed_precision="fp16",
                              )

    accelerator.print("Loading data ...")

    # dataset
    print("loading train dataset ...")
    train_dataset = ImagenetRadialKspaceDatasetHDF5(root_dir=data_root_dir, 
                                              train=True, 
                                            #   dev=True,
                                              log_scale_amplitude=args.log_scale_amplitude,
                                              log_scale_freq=args.log_scale_freq,
                                              z_normalize_spokes=args.z_normalize_spokes,
                                              min_max_normalize=args.min_max_normalize,
                                              use_half_spoke=args.use_half_spoke,
                                              )
    print("train dataset loaded!")
    print("loading val dataset ...")
    val_dataset = ImagenetRadialKspaceDatasetHDF5(root_dir=data_root_dir, 
                                              train=False, 
                                            #   dev=True,
                                              log_scale_amplitude=args.log_scale_amplitude,
                                              log_scale_freq=args.log_scale_freq,
                                              z_normalize_spokes=args.z_normalize_spokes,
                                              min_max_normalize=args.min_max_normalize,
                                              use_half_spoke=args.use_half_spoke,
                                              )

    print("val dataset loaded!")
    print("total train data size: ", len(train_dataset))
    print("total val data size: ", len(val_dataset))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size_per_gpu, 
                              shuffle=True, 
                              num_workers=dataloader_workers,
                              pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor,
                              )
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size_per_gpu, 
                            shuffle=False, 
                            num_workers=dataloader_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            )
    print("Dataloaders loaded!")

    # total training iterations
    accelerator.print("accelerator.num_processes:", accelerator.num_processes)

    accelerator.print("Loading model ...")
    # model
    model = RadialKspaceMaskedAutoencoder(
                                    num_spokes=num_spokes,
                                    spoke_length=train_dataset.spoke_length,
                                    embed_dim=args.embed_dim,
                                    depth=args.model_depth, 
                                    num_heads=args.num_heads,
                                    decoder_embed_dim=args.decoder_embed_dim,
                                    decoder_depth=args.decoder_model_depth, 
                                    decoder_num_heads=args.decoder_num_heads,
                                    loss_type=args.loss_type,
                                    ) 
    # model = model.to(device) # accelerate handles this now
    accelerator.print("Model loaded!!")

    # init wandb
    # wandb.init(project='multi_gpu_imagenet_kspace')

    # Define loss function and optimizer

    total_batch_size = batch_size_per_gpu * accelerator.num_processes * gradient_accumulation_steps

    optimizer = optim.AdamW(model.parameters(), lr=(lr*total_batch_size/256), weight_decay=args.weight_decay)

    # Define learning rate scheduler
    # Fllowing this for math of total steps: https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification_no_trainer.py
    # Scheduler and math around the number of training steps.

    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    warmup_steps = warmup_epochs * num_update_steps_per_epoch
    # accelerator.print("total training steps: ", max_train_steps)


    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                # num_warmup_steps=warmup_steps*accelerator.num_processes, # apparently it is increasing the warmup steps by num_processes
                                                num_warmup_steps=warmup_steps, 
                                                num_training_steps=max_train_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Training loop
    accelerator.init_trackers(EXPERIMENT_NAME)

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {batch_size_per_gpu}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {max_train_steps}")
    accelerator.print(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")


    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        path = os.path.basename(args.resume_from_checkpoint)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        resume_step = int(training_difference.split("steps_")[-1]) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_loader)
        completed_steps = resume_step // args.gradient_accumulation_steps
        resume_step -= starting_epoch * len(train_loader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)


    # evaluate at the beginning once
    ################################

    if args.do_initial_eval:
        model.eval()
        accelerator.print(f"doing initial evaluation ...")
        epoch_eval_loss = 0.0
        for step, batch in enumerate(val_loader):
            with torch.no_grad():
                inputs, _, raw_kspace = batch
                # inputs, labels = inputs.to(device), labels.to(device) # accelerate handles this now
                loss, _, _ = model(inputs, mask_ratio=args.mask_ratio)
                current_batch_loss = loss.detach().float()
                epoch_eval_loss += current_batch_loss

        epoch_eval_loss = epoch_eval_loss.sum().item() / len(val_loader)
        accelerator.log(
                {
                    "epoch_eval_loss": epoch_eval_loss,
                    "epoch": starting_epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
                )
        accelerator.print(f"Initial Eval loss at epoch {starting_epoch}: loss: {epoch_eval_loss}")

    # initial evaluation done
    ################################

    accelerator.print(f"training starts ...")
    for epoch in range(starting_epoch, num_train_epochs):

        # Training
        model.train()
        total_loss = 0
        correct = 0

        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_loader, resume_step)
        else:
            active_dataloader = train_loader


        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                inputs, _, raw_kspace = batch
                # inputs, labels = inputs.to(device), labels.to(device) # accelerate handles this now
                optimizer.zero_grad()

                loss, _, _ = model(inputs, mask_ratio=args.mask_ratio)

                current_batch_loss = loss.detach().float()
                total_loss += current_batch_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % log_interval == 0:
                
                batch_size = inputs.shape[0]

                current_batch_loss = current_batch_loss.sum().item() / batch_size
                # accelerator.print(f"completed_steps: {completed_steps} | Current train batch loss: {current_batch_loss} | Current train batch accuracy: {batch_train_accuracy}")

                accelerator.log(
                {
                    "train_batch_loss": current_batch_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
                )

            if completed_steps % checkpoint_interval == 0:
                save_path = os.path.join("/gpfs/scratch/rt2741/experiments", EXPERIMENT_NAME, f"checkpoint_steps_{completed_steps}")
                accelerator.save_state(save_path)
                accelerator.print("saved state at: ", save_path)

        # NOTE: there is eval, we just track loss on val set
        model.eval()
        epoch_eval_loss = 0.0
        for step, batch in enumerate(val_loader):
            with torch.no_grad():
                inputs, _, raw_kspace = batch
                # inputs, labels = inputs.to(device), labels.to(device) # accelerate handles this now
                loss, _, _ = model(inputs, mask_ratio=args.mask_ratio)
                current_batch_loss = loss.detach().float()
                epoch_eval_loss += current_batch_loss

        epoch_eval_loss = epoch_eval_loss.sum().item() / len(val_loader)
        accelerator.log(
                {
                    "epoch_eval_loss": epoch_eval_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
                )

    accelerator.end_training()

if __name__ == "__main__":
    main()

    # to run 
    """
    accelerate launch train_radial_kspace_mae.py \
        --norm spokewise \
        --log_scale_amplitude \
        --min_max_normalize \
        --embed_dim=768 \
        --model_depth=12 \
        --num_heads=12  \
        --decoder_embed_dim=512 \
        --decoder_model_depth=4 \
        --decoder_num_heads=4  \
        --num_epochs 300 \
        --mask_ratio=0.70 

    """
