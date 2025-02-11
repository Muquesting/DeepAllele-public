#!/usr/bin/env python
import os
import argparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from DeepAllele import data, model

def train_model(single_model, trainloader, valloader, checkpoint_dir, logger, device, max_epochs=100):
    # Setup callbacks.
    es = EarlyStopping(monitor="val_loss", patience=4, mode="min")
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
        devices=[device],
        callbacks=[es, checkpoint_callback, lr_monitor],
        logger=logger,
        max_epochs=max_epochs,
        benchmark=False,
        profiler="simple",
    )
    
    trainer.fit(single_model, train_dataloaders=trainloader, val_dataloaders=valloader)
    
    best_model = model.SeparateMultiHeadResidualCNN.load_from_checkpoint(checkpoint_callback.best_model_path)
    print(f"Training complete. Best checkpoint: {checkpoint_callback.best_model_path}")
    return best_model, checkpoint_callback.best_model_path

def validate_and_save_predictions(best_model, valloader, output_dir, device):
    # Run prediction and save output files.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[device],
        max_epochs=1,
        benchmark=False,
        profiler="simple",
        logger=False,
    )
    
    pred_batches = trainer.predict(best_model, dataloaders=valloader)
    predictions = []
    labels = []
    for batch in pred_batches:
        predictions.append(batch["pred"])
        labels.append(batch["label"])
    
    predictions = np.concatenate(
        [p.detach().cpu().numpy() if hasattr(p, "detach") else np.array(p) for p in predictions],
        axis=0
    )
    labels = np.concatenate(
        [l.detach().cpu().numpy() if hasattr(l, "detach") else np.array(l) for l in labels],
        axis=0
    )
    
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "predictions.npy")
    label_path = os.path.join(output_dir, "labels.npy")
    np.save(pred_path, predictions)
    np.save(label_path, labels)
    print(f"Predictions saved to: {pred_path}")
    print(f"Labels saved to: {label_path}")

def main(args):
    print("="*50)
    print("Starting DeepAllele multi-head training/validation pipeline")
    print("="*50)
    
    print("Loading data...")
    trainloader, valloader, _, _ = data.load_h5(
        args.in_folder,
        args.split_ratio,
        args.batch_size,
        batch_id=args.batch_id,
        split_by_chrom=args.split_by_chrom,
        shuffle=(args.mode == "train")
    )
    
    for seqs, _ in trainloader:
        input_length = seqs.shape[1]
        print(f"Determined input sequence length: {input_length}")
        break

    hyper_output_path = os.path.join(
        args.out_folder,
        f"{args.conv_layers}_{args.conv_repeat}_{args.kernel_number}_{args.kernel_length}_{args.filter_number}_{args.kernel_size}_{args.pooling_size}_{args.learning_rate}_{args.h_layers}_{args.hidden_size}_{args.first_batch_norm}"
    )
    os.makedirs(hyper_output_path, exist_ok=True)

    if args.mode == "train":
        # Create a batch-group folder first.
        batch_output_path = os.path.join(
            hyper_output_path, f"batch_{args.batch_id}"
        )
        os.makedirs(batch_output_path, exist_ok=True)
        for seed in range(args.random_seed_start, args.random_seed_end):
            print("\n" + "="*20, f"Seed {seed}", "="*20)
            pl.seed_everything(seed)
            # Create the seed-specific folder under the batch folder.
            seed_output_path = os.path.join(batch_output_path, f"random_seed_{seed}")
            os.makedirs(seed_output_path, exist_ok=True)
            # Define checkpoint directory under the seed folder.
            checkpoint_dir = os.path.join(seed_output_path, "checkpoint")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if args.use_wandb:
                os.environ["WANDB_API_KEY"] = args.wandb_api
                run_name = f"model_SeparateMultiHeadResidualCNN_seed_{seed}_cell_{args.batch_id}"
                logger = WandbLogger(project=args.project_id, name=run_name, log_model="all")
            else:
                logger = TensorBoardLogger(save_dir=seed_output_path)
            
            print("\nInitializing multi-head model...")
            single_model = model.SeparateMultiHeadResidualCNN(
                kernel_number=args.kernel_number,
                kernel_length=args.kernel_length,
                kernel_size=args.kernel_size,
                pooling_size=args.pooling_size,
                conv_layers=args.conv_layers,
                conv_repeat=args.conv_repeat,
                hidden_size=args.hidden_size,
                dropout=0.2,
                h_layers=args.h_layers,
                input_length=input_length * 2,
                filter_number=args.filter_number,
                pooling_type="avg",
                learning_rate=args.learning_rate,
                scheduler="cycle",
            )
            
            best_model, _ = train_model(single_model, trainloader, valloader, checkpoint_dir, logger, args.device, max_epochs=args.max_epochs)
            
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=[args.device],
                max_epochs=1,
                benchmark=False,
                profiler="simple",
            )
            val_results = trainer.validate(best_model, dataloaders=valloader)
            print(f"Seed {seed} validation results:", val_results)
            
            # prediction_output_dir = os.path.join(seed_output_path, "predictions")
            # validate_and_save_predictions(best_model, valloader, prediction_output_dir, args.device)
            # print(f"Validation for seed {seed} complete. Results saved in: {prediction_output_dir}")
    
    elif args.mode == "validate":
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        best_model = model.SeparateMultiHeadResidualCNN.load_from_checkpoint(args.checkpoint_path)
        print(f"Loaded model from {args.checkpoint_path}")
        
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[args.device],
            max_epochs=1,
            benchmark=False,
            profiler="simple",
        )
        val_results = trainer.validate(best_model, dataloaders=valloader)
        print("Validation results:", val_results)
        
        prediction_output_dir = os.path.join(hyper_output_path, "predictions")
        validate_and_save_predictions(best_model, valloader, prediction_output_dir, args.device)
    else:
        raise ValueError("Mode must be 'train' or 'validate'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Validate SeparateMultiHeadResidualCNN model")
    parser.add_argument("--mode", type=str, choices=["train", "validate"], default="train", help="Mode: train or validate")
    parser.add_argument("--in_folder", type=str, required=True, help="Path to the HDF5 file with preprocessed RNA-seq data")
    parser.add_argument("--out_folder", type=str, default="./output/", help="Directory to save outputs")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the model checkpoint (required for validate mode)")
    parser.add_argument("--batch_id", type=str, default="B_Fo_Sp_IL4", help="Batch ID")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/validation split ratio")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloaders")
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
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--random_seed_start", type=int, default=0, help="Start seed (inclusive)")
    parser.add_argument("--random_seed_end", type=int, default=1, help="End seed (exclusive)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_api", type=str, default="", help="Wandb API key")
    parser.add_argument("--project_id", type=str, default="DeepAllele-F1-mouse-RNA", help="Wandb project ID")
    parser.add_argument("--split_by_chrom", type=bool, default=True, help="Whether to split by chromosome")
    args = parser.parse_args()
    
    pl.seed_everything(args.random_seed_start)
    main(args)



