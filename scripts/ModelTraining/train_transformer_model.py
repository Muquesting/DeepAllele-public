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
from DeepAllele import data, model

if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")
    parser.add_argument(
        "--model_type", type=str, default="Separate_Transformer", help="Model type"
    )
    parser.add_argument("--in_folder", help="the input folder")
    parser.add_argument("--out_folder", help="the output folder")

    
    parser.add_argument(
        "--attention_layers",
        help="the number of the attention layers",
        type=int,
        default=2,
    )
    parser.add_argument("--pos_enc_type", default="relative")
    # num_rel_pos_features is only used in Transformer when pos_enc_type="relative"
    parser.add_argument(
        "--num_rel_pos_features",
        help="the number of relative positional features",
        type=int,
        default=66
    )

    parser.add_argument(
        "--conv_layers",
        help="the number of the large convolutional layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--conv_repeat",
        help="the number of the convolutional conv block",
        type=int,
        default=2,
    )

    parser.add_argument("--kernel_number", help="the number of the kernels", default=512)
    parser.add_argument("--kernel_length", help="the length of the kernels", default=10)
    parser.add_argument("--filter_number", help="the number of the filters", default=256)
    parser.add_argument("--kernel_size", help="the size of the kernels", default=3)
    parser.add_argument("--pooling_size", help="the size of the pooling", default=2)

    parser.add_argument("--h_layers", help="the number of fc hidden layers", default=1)
    parser.add_argument("--hidden_size", help="the hidden size of fc layers", default=256)
    parser.add_argument("--learning_rate", default=1e-4)
    args = parser.parse_args()


    model_type = args.model_type
    # Set the hyperparameters
    in_folder = args.in_folder
    out_folder = args.out_folder
    attention_layers = int(args.attention_layers)
    pos_enc_type = args.pos_enc_type
    num_rel_pos_features = int(args.num_rel_pos_features)

    conv_layers = int(args.conv_layers)
    conv_repeat = int(args.conv_repeat)
    kernel_number = int(args.kernel_number)
    kernel_length = int(args.kernel_length)
    filter_number = int(args.filter_number)
    kernel_size = int(args.kernel_size)
    pooling_size = int(args.pooling_size)

    h_layers = int(args.h_layers)
    hidden_size = int(args.hidden_size)
    learning_rate = float(args.learning_rate)


    # train_input_path = "../../data/seuqence_datasets_new_cast.hdf5"
    output_path = (
        out_folder
        + str(attention_layers)
        + "_"
        + pos_enc_type
        + "_"
        + str(num_rel_pos_features)
        + "_"
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
    )
    checkpoint_path = output_path
    log_path = out_folder + "/log"

    trainloader, valloader = data.load_data(in_folder, 0.9, 32)  # type: ignore
    if model_type == "Separate_Transformer":
        single_model = model.Separate_Transformer(
            kernel_number=kernel_number,
            kernel_length=kernel_length,
            filter_number=filter_number,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            conv_layers=conv_layers,
            conv_repeat=conv_repeat,
            attention_layers=attention_layers,
            hidden_size=hidden_size,
            dropout=0.2,
            h_layers=h_layers,
            input_length=330 * 2,
            pooling_type="avg",
            learning_rate=learning_rate,
            num_rel_pos_features=num_rel_pos_features,
        )
    elif model_type == "Transformer":
        single_model = model.Transformer(
            kernel_number=kernel_number,
            kernel_length=kernel_length,
            filter_number=filter_number,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            conv_layers=conv_layers,
            conv_repeat=conv_repeat,
            attention_layers=attention_layers,
            hidden_size=hidden_size,
            dropout=0.2,
            h_layers=h_layers,
            input_length=330 * 2,
            pooling_type="avg",
            learning_rate=learning_rate,
            num_rel_pos_features=num_rel_pos_features,
            pos_enc_type=pos_enc_type,
        )
    else:
        raise ValueError("model type not supported")

    es = EarlyStopping(monitor="val_loss", patience=7)
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", mode="min", save_top_k=1
    )
    lr_monitor = LearningRateMonitor()
    logger = TensorBoardLogger(log_path, name="model")
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        callbacks=[es, checkpoint_callback, lr_monitor],
        benchmark=False,
        profiler="simple",
        logger=logger,
    )

    trainer.fit(single_model, trainloader, valloader)