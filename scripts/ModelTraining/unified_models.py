#!/usr/bin/env python
"""
Unified training script for DeepAllele models supporting RNA-seq, ChIP-seq, and ATAC-seq data.
"""

import os
import argparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from DeepAllele import data, model, tools

def get_dataloaders(args):
    """Load data based on assay type"""
    print(f"\nLoading {args.assay_type} data from: {args.in_folder}")
    
    # Common parameters for all data loading
    common_params = {
        "path": args.in_folder,  # Changed from data_path to path to match data.py
        "split_ratio": args.split_ratio,
        "batch_size": args.batch_size,
        "split_by_chrom": args.split_by_chrom,
        "shuffle": (args.mode == "train"),
        "seed": args.random_seed_start  # Add seed parameter
    }
    
    if args.model_type == "single":
        # Single head model data loading
        return data.load_h5_single(
            **common_params,
            batch_id=args.batch_id,
            Genome=args.genome
        )
    else:
        # Multi head model data loading
        return data.load_h5(
            **common_params,
            batch_id=args.batch_id
        )

def train_model(args, model_instance, trainloader, valloader, checkpoint_dir, logger):
    """Unified training function"""
    print(f"\nInitializing training for {args.assay_type} {args.model_type} model")
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{val_loss:.2f}',
            monitor="val_loss",
            mode="min",
            save_top_k=1
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.device],
        callbacks=callbacks,
        logger=logger,
        max_epochs=args.max_epochs,
        benchmark=False,
        profiler="simple"
    )
    
    trainer.fit(model_instance, trainloader, valloader)
    return trainer, callbacks[1].best_model_path

def get_model(args, input_length):
    """Create model based on type and assay"""
    common_params = {
        "kernel_number": args.kernel_number,
        "kernel_length": args.kernel_length,
        "kernel_size": args.kernel_size,
        "pooling_size": args.pooling_size,
        "conv_layers": args.conv_layers,
        "conv_repeat": args.conv_repeat,
        "hidden_size": args.hidden_size,
        "dropout": 0.2,
        "h_layers": args.h_layers,
        "input_length": input_length * 2,
        "filter_number": args.filter_number,
        "pooling_type": "avg",
        "learning_rate": args.learning_rate,
        "scheduler": "cycle"
    }
    
    if args.model_type == "single":
        return model.SingleHeadResidualCNN(**common_params)
    elif args.model_type == "multi":
        return model.SeparateMultiHeadResidualCNN(**common_params)
    elif args.model_type == "multi_ComputedRatio":
        return model.ComputeRatioSeparateMultiHeadResidualCNN(**common_params)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

def main(args):
    """Main execution function"""
    print("\n" + "="*50)
    print(f"Starting DeepAllele {args.assay_type} {args.model_type} training pipeline")
    print("="*50)

    # Define output directory structure
    hyper_output_path = os.path.join(
        args.out_folder,
        f"{args.conv_layers}_{args.conv_repeat}_{args.kernel_number}_{args.kernel_length}_"
        f"{args.filter_number}_{args.kernel_size}_{args.pooling_size}_{args.learning_rate}_"
        f"{args.h_layers}_{args.hidden_size}_{args.first_batch_norm}"
    )
    os.makedirs(hyper_output_path, exist_ok=True)

    if args.mode == "train":
        # Load data based on assay type
        trainloader, valloader, _, _ = get_dataloaders(args)
        
        # Get input length from first batch
        for seqs, _ in trainloader:
            input_length = seqs.shape[1] if args.assay_type != "ATAC" else 330
            print(f"Input sequence length: {input_length}")
            break

        # Training loop for different genomes (only for single-head models)
        genomes = [["B6"], ["Cast"], ["B6", "Cast"]] if args.model_type == "single" else [args.genome]
        
        for genome in genomes:
            if args.model_type == "single":
                args.genome = genome
                trainloader, valloader, _, _ = get_dataloaders(args)
            
            genome_str = "_".join(genome)
            print(f"\nProcessing genome combination: {genome_str}")
            
            # Loop over seeds
            for seed in range(args.random_seed_start, args.random_seed_end):
                print(f"\n{'='*20} Seed {seed} {'='*20}")
                pl.seed_everything(seed)
                
                # Setup output paths
                seed_output_path = os.path.join(
                    hyper_output_path,
                    f"batch_{args.batch_id}" if args.batch_id else "",
                    genome_str,
                    f"random_seed_{seed}"
                )
                os.makedirs(seed_output_path, exist_ok=True)
                
                # Setup logging with unified run naming
                if args.use_wandb:
                    os.environ["WANDB_API_KEY"] = args.wandb_api
                    run_name = f"{args.assay_type}_{args.model_type}_seed_{seed}"
                    if args.batch_id:
                        run_name += f"_batch_{args.batch_id}"
                    if genome_str:
                        run_name += f"_genome_{genome_str}"
                    logger = WandbLogger(project=args.project_id, name=run_name, log_model="all")
                else:
                    logger = TensorBoardLogger(save_dir=seed_output_path)
                
                # Initialize and train model
                model_instance = get_model(args, input_length)
                trainer, best_model_path = train_model(
                    args, model_instance, trainloader, valloader,
                    os.path.join(seed_output_path, "checkpoint"), 
                    logger
                )
                
                if args.use_wandb:
                    logger.experiment.finish()
    
    else:  # validate mode
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        
        # Load validation data
        _, valloader, _, _ = get_dataloaders(args)
        
        # Load model from checkpoint, 
        if args.model_type == "single":
            model_class = model.SingleHeadResidualCNN
        elif args.model_type == "multi":
            model_class = model.SeparateMultiHeadResidualCNN
        elif args.model_type == "multi_ComputedRatio":
            model_class = model.ComputeRatioSeparateMultiHeadResidualCNN

        model_instance = model_class.load_from_checkpoint(args.checkpoint_path)
        
        # Setup trainer for validation
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[args.device],
            max_epochs=1,
            benchmark=False,
            profiler="simple"
        )
        
        # Run validation
        val_results = trainer.validate(model_instance, dataloaders=valloader)
        print("\nValidation results:", val_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified DeepAllele Training Script")
    
    # Basic parameters
    parser.add_argument("--mode", type=str, choices=["train", "validate"], default="train",
                      help="Operation mode: train or validate")
    parser.add_argument("--assay_type", type=str, choices=["RNA", "CHIP", "ATAC"], required=True,
                      help="Type of assay data")
    parser.add_argument("--model_type", type=str, choices=["single", "multi", 'multi_ComputedRatio'], required=True,
                      help="Model architecture type")
    
    # File paths
    parser.add_argument("--in_folder", type=str, required=True,
                      help="Input data folder path")
    parser.add_argument("--out_folder", type=str, default="./output/",
                      help="Output directory path")
    parser.add_argument("--checkpoint_path", type=str, default="",
                      help="Model checkpoint path (for validate mode)")
    
    # Data parameters
    parser.add_argument("--batch_id", type=str, default="",
                      help="Batch identifier")
    parser.add_argument("--genome", nargs="+", default=["B6", "Cast"],
                      help="List of genome identifiers")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                      help="Train/validation split ratio")
    parser.add_argument("--split_by_chrom", type=bool, default=True,
                      help="Whether to split by chromosome")
    
    # Model hyperparameters
    parser.add_argument("--conv_layers", type=int, default=4,
                      help="Number of convolution layers")
    parser.add_argument("--conv_repeat", type=int, default=1,
                      help="Convolution block repeat count")
    parser.add_argument("--kernel_number", type=int, default=512,
                      help="Number of kernels")
    parser.add_argument("--kernel_length", type=int, default=10,
                      help="Kernel length")
    parser.add_argument("--filter_number", type=int, default=256,
                      help="Number of filters")
    parser.add_argument("--kernel_size", type=int, default=3,
                      help="Kernel size")
    parser.add_argument("--pooling_size", type=int, default=5,
                      help="Pooling size")
    parser.add_argument("--h_layers", type=int, default=1,
                      help="Number of hidden layers")
    parser.add_argument("--hidden_size", type=int, default=512,
                      help="Hidden layer size")
    parser.add_argument("--first_batch_norm", type=bool, default=True,
                      help="Use batch normalization in first layer")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100,
                      help="Maximum training epochs")
    
    # Training control
    parser.add_argument("--device", type=int, default=0,
                      help="GPU device index")
    parser.add_argument("--random_seed_start", type=int, default=0,
                      help="Random seed range start")
    parser.add_argument("--random_seed_end", type=int, default=5,
                      help="Random seed range end")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                      help="Use Weights & Biases logging")
    parser.add_argument("--wandb_api", type=str, default="",
                      help="Weights & Biases API key")
    parser.add_argument("--project_id", type=str, default="DeepAllele-unified",
                      help="Project identifier")
    
    args = parser.parse_args()
    print('--------')
    print(args.batch_id)
    if args.batch_id == None:
        print('****')
    print('--------')
    main(args)
