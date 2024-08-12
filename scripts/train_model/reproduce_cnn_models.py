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
from zmq import device
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

    trainer.fit(single_model, trainloader, valloader)

    PWM = single_model.get_PWM(3.0)

    motif_list = []
    for i in range(PWM.shape[0]):
        if np.max(PWM[i, :, :]) > 0.5:
            print("The %dth kernel" % i)
            print(PWM[i, :, :].max())
            motif_list.append(i)
    #             print(PWM_all[:,:,i])

    pwm = PWM[motif_list[:], :, :].copy()
    tools.mkdir(checkpoint_path)

    tools.write_meme_file(pwm, checkpoint_path + "/selected_motifs.meme")
    tools.write_meme_file(PWM, checkpoint_path + "/all_motifs.meme")


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
        default=2,
    )
    parser.add_argument(
        "--conv_repeat",
        help="the number of the convolution conv block",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--kernel_number", help="the number of the kernels", default=512
    )
    parser.add_argument("--kernel_length", help="the length of the kernels", default=15)
    parser.add_argument(
        "--filter_number", help="the number of the filters", default=256
    )
    parser.add_argument("--kernel_size", help="the size of the kernels", default=3)
    parser.add_argument("--pooling_size", help="the size of the pooling", default=2)

    parser.add_argument("--h_layers", help="the number of fc hidden layers", default=2)
    parser.add_argument(
        "--hidden_size", help="the hidden size of fc layers", default=256
    )
    parser.add_argument("--learning_rate", default=1e-4)
    parser.add_argument("--random_seed_start", default=0)
    parser.add_argument("--random_seed_end", default=5)
    parser.add_argument("--device", default=0)

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
    device = int(args.device)

    learning_rate = float(args.learning_rate)

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
    )
    trainloader, valloader = data.load_data(in_folder, 0.9, 32)  # type: ignore
    for random_seed in range(random_seed_start, random_seed_end):

        # train_input_path = "../../data/seuqence_datasets_new_cast.hdf5"
        output_path = hyper_output_path + "/random_seed_" + str(random_seed)
        checkpoint_path = output_path
        log_path = hyper_output_path + "/log"

        if model_type == "Separate_Multihead_Residual_CNN":
            single_model = model.SeparateMultiHeadResidualCNN(
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
            )
        elif model_type == "Multihead_Residual_CNN":
            single_model = model.MultiHeadResidualCNN(
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
            )
        else:
            raise ValueError("model type not supported")

        train_model(
            single_model,
            trainloader=trainloader,
            valloader=valloader,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            device=device,
        )
