#!/usr/bin/env python
"""
Professional training and validation script for SingleHeadResidualCNN using PyTorch Lightning.

This script supports training and validating your model, and it saves prediction and label
files from the validation set. When training, you can loop over a range of random seeds so that 
multiple independent runs occur in one execution.

Usage:
    Training (multiple seeds):
        python run_model.py --mode train --in_folder /path/to/RNA-seq-preprocessed.hdf5 \
           --batch_id MF_PC_IL4 --use_wandb --wandb_api YOUR_WANDB_API_KEY \
           --random_seed_start 0 --random_seed_end 5 --device 0

    Validation (single checkpoint):
        python run_model.py --mode validate --in_folder /path/to/RNA-seq-preprocessed.hdf5 \
           --checkpoint_path /path/to/your_checkpoint.ckpt --batch_id MF_PC_IL4 --device 0
"""

import os
import argparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Import your project modules (adjust the import paths as needed)
from DeepAllele import data, model


def get_dataloaders(in_folder, batch_id, genome, batch_size=32, split_ratio=0.9, split_by_chrom=True, shuffle=True):
    """
    Load the train and validation DataLoaders from an HDF5 file.
    """
    print(f"\nLoading data from: {in_folder}")
    print(f"Batch ID: {batch_id}")
    print(f"Genome: {genome}")
    print(f"Split ratio: {split_ratio}, Split by chromosome: {split_by_chrom}")
    
    trainloader, valloader, _, _ = data.load_h5_single(
        in_folder,
        split_ratio,
        batch_size,
        batch_id=batch_id,
        Genome=genome,
        split_by_chrom=split_by_chrom,
        shuffle=shuffle,
    )
    print(f"Data loading complete. TrainLoader batches: {len(trainloader)}, ValLoader batches: {len(valloader)}")
    return trainloader, valloader


def train_model(single_model, trainloader, valloader, checkpoint_dir, logger, gpu_index, max_epochs=100):
    """
    Train the given model using PyTorch Lightning.
    """
    print(f"\nInitializing training with the following configuration:")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"GPU index: {gpu_index}")
    print(f"Max epochs: {max_epochs}")
    # Set up callbacks.
    early_stop = EarlyStopping(monitor="val_loss", patience=4, mode="min")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}',
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu_index],
        callbacks=[early_stop, checkpoint_callback, lr_monitor],
        logger=logger,
        max_epochs=max_epochs,
        benchmark=False,
        profiler="simple",
    )
    
    trainer.fit(single_model, train_dataloaders=trainloader, val_dataloaders=valloader)
    
    # Reload the best model from checkpoint.
    best_model = single_model.__class__.load_from_checkpoint(checkpoint_callback.best_model_path)
    print(f"Training complete. Best checkpoint for this seed: {checkpoint_callback.best_model_path}")
    return best_model, checkpoint_callback.best_model_path





def main(args):
    print("\n" + "="*50)
    print("Starting DeepAllele training/validation pipeline")
    print("="*50)
    
    print(f"\nModel configuration:")
    print(f"Mode: {args.mode}")
    print(f"Model type: {args.model_type}")
    print(f"Convolution layers: {args.conv_layers}")
    print(f"Kernel number: {args.kernel_number}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Create genome string for path
    genome_str = "_".join(args.genome)
    print(f"Using genomes: {genome_str}")
    
    # Define the hyper-output directory based on hyperparameters.
    hyper_output_path = os.path.join(
        args.out_folder,
        f"{args.conv_layers}_{args.conv_repeat}_{args.kernel_number}_{args.kernel_length}_"
        f"{args.filter_number}_{args.kernel_size}_{args.pooling_size}_{args.learning_rate}_"
        f"{args.h_layers}_{args.hidden_size}_{args.first_batch_norm}_{genome_str}"  # Add genome string
    )
    print(f"\nOutput directory structure:")
    print(f"Base output path: {hyper_output_path}")
    os.makedirs(hyper_output_path, exist_ok=True)
    
    # Load data (the same for all seeds).
    print("\nInitializing data loading...")
    
    trainloader, valloader = get_dataloaders(
        in_folder=args.in_folder,
        batch_id=args.batch_id,
        genome=args.genome,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        split_by_chrom=args.split_by_chrom,
        shuffle=(args.mode == "train"),
    )

    print(type(args.split_by_chrom), type(((args.mode == "train"))))
    
    # Determine the input length from the first training batch.
    for seqs, _ in trainloader:
        input_length = seqs.shape[1]
        print(f"Determined input sequence length: {input_length}")
        break

    # For single-head, update checkpoint structure: batch/{batch_id}/{Genome_string}/random_seed
    batch_output_path = os.path.join(
        hyper_output_path,
        f"batch_{args.batch_id}",
        "_".join(args.genome)
    )
    os.makedirs(batch_output_path, exist_ok=True)

    if args.mode == "train":
        print(f"\nStarting training loop for {args.random_seed_end - args.random_seed_start} seeds")
        # Loop over seeds.
        for seed in range(args.random_seed_start, args.random_seed_end):
            print(f"\n{'='*20} Seed {seed} {'='*20}")
            pl.seed_everything(seed)
            # Create the seed-specific folder under the batch folder.
            seed_output_path = os.path.join(batch_output_path, f"random_seed_{seed}")
            os.makedirs(seed_output_path, exist_ok=True)
            # Define checkpoint directory under the seed folder.
            checkpoint_dir = os.path.join(seed_output_path, "checkpoint")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Set up logger: choose WandbLogger or TensorBoardLogger.
            if args.use_wandb:
                print(f"Initializing Weights & Biases logging:")
                print(f"Project ID: {args.project_id}")
                print(f"Run name: model_{args.model_type}_seed_{seed}_cell_{args.batch_id}")
                os.environ["WANDB_API_KEY"] = args.wandb_api
                run_name = f"model_{args.model_type}_seed_{seed}_cell_{args.batch_id}"
                logger = WandbLogger(project=args.project_id, name=run_name, log_model="all")
            else:
                logger = TensorBoardLogger(save_dir=seed_output_path)
            
            # Create the model.
            print("\nInitializing model...")
            if args.model_type == "SingleHeadResidualCNN":
                single_model = model.SingleHeadResidualCNN(
                    kernel_number=args.kernel_number,
                    kernel_length=args.kernel_length,
                    kernel_size=args.kernel_size,
                    pooling_size=args.pooling_size,
                    conv_layers=args.conv_layers,
                    conv_repeat=args.conv_repeat,
                    hidden_size=args.hidden_size,
                    dropout=0.2,
                    h_layers=args.h_layers,
                    input_length=input_length * 2,  # Adjust if needed (e.g., for paired inputs)
                    filter_number=args.filter_number,
                    pooling_type="avg",
                    learning_rate=args.learning_rate,
                    scheduler="cycle",
                )
            else:
                raise ValueError("Unsupported model type.")
            
            # Train the model for this seed.
            print("\nStarting training...")
            best_model, _ = train_model(single_model, trainloader, valloader, checkpoint_dir, logger, args.device, max_epochs=args.max_epochs)
            
            # Validate using trainer.validate.
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=[args.device],
                max_epochs=1,
                benchmark=False,
                profiler="simple",
            )
            val_results = trainer.validate(best_model, dataloaders=valloader)
            print(f"Seed {seed} validation results:", val_results)
            
            if args.use_wandb:
                logger.experiment.finish()
            # # Save predictions and labels from the validation set.
            # prediction_output_dir = os.path.join(seed_output_path, "predictions")
            # validate_and_save_predictions(best_model, valloader, prediction_output_dir, args.device)
            # print(f"\nValidation for seed {seed} complete")
            # print(f"Results saved in: {prediction_output_dir}")
    else:
        raise ValueError("Mode must be 'train' or 'validate'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Validate SingleHeadResidualCNN model")
    
    # Mode and file paths.
    parser.add_argument("--mode", type=str, choices=["train", "validate"], default="train",
                        help="Mode: train or validate")
    parser.add_argument("--in_folder", type=str, required=True,
                        help="Path to the HDF5 file with preprocessed RNA-seq data")
    parser.add_argument("--out_folder", type=str, default="./output/",
                        help="Directory to save outputs (checkpoints, predictions, etc.)")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to the model checkpoint (required for validate mode)")
    
    # Data options.
    parser.add_argument("--batch_id", type=str, default="MF_PC_IL4", help="Batch ID")
    parser.add_argument("--genome", nargs="+", default=["B6", "Cast"], help="List of genome identifiers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloaders")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/validation split ratio")
    parser.add_argument("--split_by_chrom", type=bool, default=True, help="Whether to split by chromosome")
    
    # Model hyperparameters.
    parser.add_argument("--model_type", type=str, default="SingleHeadResidualCNN", help="Type of model to use")
    parser.add_argument("--conv_layers", type=int, default=4, help="Number of convolution layers")
    parser.add_argument("--conv_repeat", type=int, default=1, help="Number of repeats for convolution layers")
    parser.add_argument("--kernel_number", type=int, default=512, help="Number of kernels")
    parser.add_argument("--kernel_length", type=int, default=10, help="Length of each kernel")
    parser.add_argument("--filter_number", type=int, default=256, help="Number of filters")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size")
    parser.add_argument("--pooling_size", type=int, default=5, help="Pooling size")
    parser.add_argument("--h_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--first_batch_norm", type=bool, default=True, help="Whether to apply batch norm to the first layer")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    
    # Device.
    parser.add_argument("--device", type=int, default=0, help="GPU device index (e.g. 0 for 'cuda:0')")
    
    # Random seed range.
    parser.add_argument("--random_seed_start", type=int, default=0, help="Start of random seed range (inclusive)")
    parser.add_argument("--random_seed_end", type=int, default=1, help="End of random seed range (exclusive)")
    
    # Wandb logging options.
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_api", type=str, default="", help="Wandb API key")
    parser.add_argument("--project_id", type=str, default="DeepAllele-F1-mouse-RNA", help="Wandb project ID")
    
    args = parser.parse_args()
    
    # pl.seed_everything(args.random_seed_start)  # Default seeding (will be overwritten in each seed loop) 
    
    main(args)
