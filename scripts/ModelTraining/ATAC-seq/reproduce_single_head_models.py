import numpy as np
import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from DeepAllele import data, model, tools

def train_model(
    Model,
    trainloader,
    valloader,
    checkpoint_path,
    log_path,
    device,
):
    es = EarlyStopping(monitor="val_ratio_corr", patience=4, mode="max")
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor="val_ratio_corr", mode="max", save_top_k=1
    )
    lr_monitor = LearningRateMonitor()
    logger = TensorBoardLogger(log_path, name="model")
    trainer = pl.Trainer(
        devices=[device],
        accelerator="gpu",
        callbacks=[es, checkpoint_callback, lr_monitor],
        benchmark=False,
        profiler="simple",
        logger=logger,
        max_epochs=100,  # TODO: change this to 100
    )
    single_model = Model
    trainer.fit(single_model, trainloader, valloader)

    single_model = model.SingleHeadResidualCNN.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )


if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")
    parser.add_argument(
        "--model_type",
        type=str,
        default="Separate_Multihead_Residual_CNN",
        help="Model type",
    )
    parser.add_argument("--in_folder", help="the input folder")
    parser.add_argument("--out_folder", help="the output folder")

    parser.add_argument(
        "--conv_layers",
        help="the number of the large convolution layers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--conv_repeat",
        help="the number of the convolution conv block",
        type=int,
        default=1,
    )

    parser.add_argument("--kernel_number", type=int, help="the number of the kernels", default=256)
    parser.add_argument("--kernel_length", type=int, help="the length of the kernels", default=15)
    parser.add_argument("--filter_number", type=int, help="the number of the filters", default=256)
    parser.add_argument("--kernel_size", type=int, help="the size of the kernels", default=5)
    parser.add_argument("--pooling_size", type=int, help="the size of the pooling", default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--random_seed_start", type=int, default=0)
    parser.add_argument("--random_seed_end", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--first_batch_norm", type=bool, default=True)

    parser.add_argument("--h_layers", type=int, help="the number of fc hidden layers", default=2)
    parser.add_argument(
        "--hidden_size", type=int, help="the hidden size of fc layers", default=256
    )
    parser.add_argument("--batch_id", type=str, default="sc")

    args = parser.parse_args()

    model_type = args.model_type
    # Set the hyperparameters
    in_folder = args.in_folder
    out_folder = args.out_folder

    conv_layers = int(args.conv_layers)
    conv_repeat = int(args.conv_repeat)
    kernel_number = int(args.kernel_number)
    kernel_length = int(args.kernel_length)
    filter_number = int(args.filter_number)
    kernel_size = int(args.kernel_size)
    pooling_size = int(args.pooling_size)

    h_layers = int(args.h_layers)
    hidden_size = int(args.hidden_size)

    random_seed_start = int(args.random_seed_start)
    random_seed_end = int(args.random_seed_end)
    batch_id = args.batch_id
    device = int(args.device)

    learning_rate = float(args.learning_rate)
    # convert the first_batch_norm to bool, 'True' -> True, 'False' -> False
    first_batch_norm = args.first_batch_norm
    print(first_batch_norm)
    print(args.first_batch_norm)

    

    hyper_output_path = (
        out_folder
        + str(conv_layers)
        + "_"
        + str(conv_repeat)
        + "_"
        + str(kernel_number)
        + "_"
        + str(kernel_length)
        + "_"
        + str(filter_number)
        + "_"
        + str(kernel_size)
        + "_"
        + str(pooling_size)
        + "_"
        + str(learning_rate)
        + "_"
        + str(h_layers)
        + "_"
        + str(hidden_size)
        + "_"
        + str(first_batch_norm)
    )
    for genome in [["B6"], ["Cast"], ["B6", "Cast"]]:

        trainloader, valloader, _,_ = data.load_h5_single(in_folder, 0.9, 32, batch_id=batch_id, Genome=genome, split_by_chrom=True)  # type: ignore
        for random_seed in range(random_seed_start, random_seed_end):
            genome_string = "_".join(genome)

            # train_input_path = "../../data/seuqence_datasets_new_cast.hdf5"
            output_path = hyper_output_path +'/' + genome_string+ "/random_seed_" + str(random_seed)
            checkpoint_path = output_path
            log_path = hyper_output_path + "/log"

            single_model = model.SingleHeadResidualCNN(
                    kernel_number=kernel_number,
                    kernel_length=kernel_length,
                    kernel_size=kernel_size,
                    pooling_size=pooling_size,
                    conv_layers=conv_layers,
                    conv_repeat=conv_repeat,
                    hidden_size=hidden_size,
                    dropout=0.2,
                    h_layers=h_layers,
                    input_length=330 * 2,
                    filter_number=filter_number,
                    pooling_type="avg",
                    learning_rate=learning_rate,
                    scheduler="cycle",
                )

            train_model(
                single_model,
                trainloader=trainloader,
                valloader=valloader,
                checkpoint_path=checkpoint_path,
                log_path=log_path,
                device=device,
            )
