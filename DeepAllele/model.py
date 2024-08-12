from math import ceil
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from DeepAllele.nn import (
    ConvBlock,
    Residual,
    AttentionPool,
    SoftmaxPool,
    Attention,
    Attention_2,
)

# TODO Review and delete the unnecessary  models


def transfer(weights):
    return torch.exp(weights) / torch.exp(weights).sum(dim=1, keepdim=True)


class Base(pl.LightningModule):
    def __init__(self, mask=None, activation=False, scheduler="plateau") -> None:
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.activation = activation

    def forward(self, x):
        return NotImplementedError

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [lr_scheduler]

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5
        )

        if self.hparams.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    # TODO: add an argument to control the patience
                    optimizer,
                    patience=2,
                ),
                "monitor": "val_loss",
            }
        elif self.hparams.scheduler == "cycle":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.hparams.learning_rate / 2,
                    max_lr=self.hparams.learning_rate * 2,
                    cycle_momentum=False,
                ),
                "interval": "step",
            }
        else:
            print("No scheduler is used")

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        if torch.isnan(loss) or torch.isinf(loss):
            return None
        else:
            self.log("train_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        if torch.isnan(loss) or torch.isinf(loss):
            return None
        else:
            self.log("val_loss", loss)
            result = torch.stack([y_hat.cpu().detach(), y.cpu().detach()], dim=1)
            return result

    def validation_epoch_end(self, validation_step_outputs):
        all_val_result = torch.cat(validation_step_outputs, dim=0)
        predict_index = 0
        label_index = 1

        val_count1_corr = stats.pearsonr(
            all_val_result[:, predict_index, 0], all_val_result[:, label_index, 0]
        ).statistic
        val_count2_corr = stats.pearsonr(
            all_val_result[:, predict_index, 1], all_val_result[:, label_index, 1]
        ).statistic
        val_ratio_corr = stats.pearsonr(
            all_val_result[:, predict_index, 2], all_val_result[:, label_index, 2]
        ).statistic

        self.log("val_count1_corr", val_count1_corr)
        self.log("val_count2_corr", val_count2_corr)
        self.log("val_ratio_corr", val_ratio_corr)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return torch.stack([self(batch[0]), batch[1]])

    def get_PWM(self, amplify=5.0):
        kernel_weights = self.conv0[0].weight
        PWM = transfer(amplify * kernel_weights)
        PWM = np.array(PWM.cpu().detach())
        return PWM


class SeparateMultiHeadResidualCNN(Base):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=10,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=330,
        filter_number=256,
        pooling_type="avg",
        learning_rate=1e-3,
        conv_repeat=1,
        dilation=1,
        first_batch_norm=True,
        all_batch_norm=True,
        scheduler="plateau",
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """

        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
            batch_norm=first_batch_norm,
        )

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
                batch_norm=all_batch_norm,
            )
        )
        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                            dilation=dilation,
                            batch_norm=all_batch_norm,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )

        self.ratio_fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_size, 2 * hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for layer in range(h_layers)
        )

        self.ratio_out = nn.Sequential(nn.Linear(2 * hidden_size, 1))
        self.counts_out = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, x, mask=None, activation=False):
        x = torch.permute(x, (0, 2, 1, 3))

        x_b6 = x[:, :, :, 0]
        x_cast = x[:, :, :, 1]

        if activation:
            x_b6 = self.conv0(x_b6)
            x_cast = self.conv0(x_cast)
            # return the stack of the two activations
            return torch.stack([x_b6, x_cast], dim=-1)

        x_rc = torch.flip(x_b6, [1, 2])
        x_b6 = self.conv0(torch.concat([x_b6, x_rc], dim=-1))

        x_rc = torch.flip(x_cast, [1, 2])
        x_cast = self.conv0(torch.concat([x_cast, x_rc], dim=-1))

        if mask is not None:
            x_b6[:, mask, :] = 0
            x_cast[:, mask, :] = 0

        for layer in self.convlayers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = x_b6.flatten(1)
        x_cast = x_cast.flatten(1)

        x_b6 = self.fc0(x_b6)
        x_cast = self.fc0(x_cast)

        x_ratio = torch.cat([x_b6, x_cast], dim=-1)
        for layer in self.ratio_fclayers:
            x_ratio = layer(x_ratio)

        x_ratio = self.ratio_out(x_ratio)

        # print(x1.shape)
        for layer in self.fclayers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = self.counts_out(x_b6)
        x_cast = self.counts_out(x_cast)

        return torch.cat([x_b6, x_cast, x_ratio], dim=-1)

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        return self(batch[0], mask=self.mask, activation=self.activation)


class RatioSingleHeadResidualCNN(Base):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=10,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=330,
        filter_number=256,
        pooling_type="avg",
        learning_rate=1e-3,
        conv_repeat=1,
        dilation=1,
        first_batch_norm=True,
        all_batch_norm=True,
        scheduler="plateau",
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """

        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
            batch_norm=first_batch_norm,
        )

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
                batch_norm=all_batch_norm,
            )
        )
        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                            dilation=dilation,
                            batch_norm=all_batch_norm,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )

        self.ratio_fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_size, 2 * hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for layer in range(h_layers)
        )

        self.ratio_out = nn.Sequential(nn.Linear(2 * hidden_size, 1))
        self.counts_out = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, x, mask=None, activation=False):
        x = torch.permute(x, (0, 2, 1, 3))

        x_b6 = x[:, :, :, 0]
        x_cast = x[:, :, :, 1]

        if activation:
            x_b6 = self.conv0(x_b6)
            x_cast = self.conv0(x_cast)
            # return the stack of the two activations
            return torch.stack([x_b6, x_cast], dim=-1)

        x_rc = torch.flip(x_b6, [1, 2])
        x_b6 = self.conv0(torch.concat([x_b6, x_rc], dim=-1))

        x_rc = torch.flip(x_cast, [1, 2])
        x_cast = self.conv0(torch.concat([x_cast, x_rc], dim=-1))

        if mask is not None:
            x_b6[:, mask, :] = 0
            x_cast[:, mask, :] = 0

        for layer in self.convlayers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = x_b6.flatten(1)
        x_cast = x_cast.flatten(1)

        x_b6 = self.fc0(x_b6)
        x_cast = self.fc0(x_cast)

        x_ratio = torch.cat([x_b6, x_cast], dim=-1)
        for layer in self.ratio_fclayers:
            x_ratio = layer(x_ratio)

        x_ratio = self.ratio_out(x_ratio)

        # print(x1.shape)
        for layer in self.fclayers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = self.counts_out(x_b6)
        x_cast = self.counts_out(x_cast)

        return torch.cat([x_b6, x_cast, x_ratio], dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # only calculate the loss for the ratio
        loss = F.mse_loss(y_hat[:, -1], y[:, -1])

        if torch.isnan(loss) or torch.isinf(loss):
            return None
        else:
            self.log("train_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = F.mse_loss(y_hat, y)
        loss = F.mse_loss(y_hat[:, -1], y[:, -1])

        if torch.isnan(loss) or torch.isinf(loss):
            return None
        else:
            self.log("val_loss", loss)
            result = torch.stack([y_hat.cpu().detach(), y.cpu().detach()], dim=1)
            return result

    def validation_epoch_end(self, validation_step_outputs):
        all_val_result = torch.cat(validation_step_outputs, dim=0)
        predict_index = 0
        label_index = 1

        val_count1_corr = stats.pearsonr(
            all_val_result[:, predict_index, 0], all_val_result[:, label_index, 0]
        ).statistic
        val_count2_corr = stats.pearsonr(
            all_val_result[:, predict_index, 1], all_val_result[:, label_index, 1]
        ).statistic
        val_ratio_corr = stats.pearsonr(
            all_val_result[:, predict_index, 2], all_val_result[:, label_index, 2]
        ).statistic

        self.log("val_count1_corr", val_count1_corr)
        self.log("val_count2_corr", val_count2_corr)
        self.log("val_ratio_corr", val_ratio_corr)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = F.mse_loss(y_hat, y)
        loss = F.mse_loss(y_hat[:, -1], y[:, -1])
        self.log("test_loss", loss)

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        return self(batch[0], mask=self.mask, activation=self.activation)


class SeparateMultiHeadResidualCNN_new(Base):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=10,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=330,
        filter_number=256,
        pooling_type="avg",
        learning_rate=1e-3,
        conv_repeat=1,
        dilation=1,
        first_batch_norm=True,
        all_batch_norm=True,
        scheduler="plateau",
        first_pooling=False,
        first_pooling_size=None,
        num_celltypes=1,
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """

        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
            batch_norm=first_batch_norm,
        )

        self.convlayers = nn.ModuleList()
        if conv_layers > 0:
            self.convlayers.append(
                ConvBlock(
                    kernel_number,
                    filter_number,
                    kernel_size,
                    padding=padding,
                    batch_norm=all_batch_norm,
                )
            )
        fc_dim = input_length

        if first_pooling:
            fc_dim = ceil(fc_dim / first_pooling_size)
            if pooling_type == "max":
                self.pool0 = nn.MaxPool1d(first_pooling_size, ceil_mode=True)
            elif pooling_type == "avg":
                self.pool0 = nn.AvgPool1d(first_pooling_size, ceil_mode=True)
            elif pooling_type == "softmax":
                self.pool0 = SoftmaxPool(first_pooling_size)
            else:
                raise ValueError("pooling type not supported")

        for layer in range(conv_layers - 1):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                            dilation=dilation,
                            batch_norm=all_batch_norm,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )

        self.ratio_fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_size, 2 * hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for layer in range(h_layers)
        )

        self.ratio_out = nn.Sequential(
            nn.Linear(2 * hidden_size, self.hparams.num_celltypes)
        )
        self.counts_out = nn.Sequential(
            nn.Linear(hidden_size, self.hparams.num_celltypes)
        )

    def forward(self, x, mask=None, activation=False):
        x = torch.permute(x, (0, 2, 1, 3))

        x_b6 = x[:, :, :, 0]
        x_cast = x[:, :, :, 1]

        if activation:
            x_b6 = self.conv0(x_b6)
            x_cast = self.conv0(x_cast)
            # return the stack of the two activations
            return torch.stack([x_b6, x_cast], dim=-1)

        x_rc = torch.flip(x_b6, [1, 2])
        x_b6 = self.conv0(torch.concat([x_b6, x_rc], dim=-1))

        x_rc = torch.flip(x_cast, [1, 2])
        x_cast = self.conv0(torch.concat([x_cast, x_rc], dim=-1))

        if self.hparams.first_pooling:
            x_b6 = self.pool0(x_b6)
            x_cast = self.pool0(x_cast)

        if mask is not None:
            x_b6[:, mask, :] = 0
            x_cast[:, mask, :] = 0

        for layer in self.convlayers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = x_b6.flatten(1)
        x_cast = x_cast.flatten(1)

        x_b6 = self.fc0(x_b6)
        x_cast = self.fc0(x_cast)

        x_ratio = torch.cat([x_b6, x_cast], dim=-1)
        for layer in self.ratio_fclayers:
            x_ratio = layer(x_ratio)

        x_ratio = self.ratio_out(x_ratio)

        # print(x1.shape)
        for layer in self.fclayers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = self.counts_out(x_b6)
        x_cast = self.counts_out(x_cast)

        return torch.cat([x_b6, x_cast, x_ratio], dim=-1)

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        return self(batch[0], mask=self.mask, activation=self.activation)


class SeparateMultiHeadSimpleCNN(Base):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=10,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=330,
        filter_number=256,
        pooling_type="avg",
        learning_rate=1e-3,
        conv_repeat=1,
        dilation=1,
        first_batch_norm=True,
        all_batch_norm=True,
        scheduler="plateau",
        num_celltypes=1,
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """

        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
            batch_norm=first_batch_norm,
        )
        if pooling_type == "max":
            self.GloblePooling = nn.MaxPool1d(kernel_size=input_length)
        elif pooling_type == "avg":
            self.GloblePooling = nn.AvgPool1d(kernel_size=input_length)
        elif pooling_type == "attention":
            self.GloblePooling = Attention(input_length)
        elif pooling_type == "softmax":
            self.GloblePooling = SoftmaxPool(pool_size=input_length)

        fc_dim = 1
        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )

        self.fc_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )

        self.ratio_fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_size, 2 * hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for layer in range(h_layers)
        )

        self.ratio_out = nn.Sequential(
            nn.Linear(2 * hidden_size, self.hparams.num_celltypes)
        )
        self.counts_out = nn.Sequential(
            nn.Linear(hidden_size, self.hparams.num_celltypes)
        )

    def forward(self, x, mask=None, activation=False):
        x = torch.permute(x, (0, 2, 1, 3))

        x_b6 = x[:, :, :, 0]
        x_cast = x[:, :, :, 1]

        if activation:
            x_b6 = self.conv0(x_b6)
            x_cast = self.conv0(x_cast)
            # return the stack of the two activations
            return torch.stack([x_b6, x_cast], dim=-1)

        x_rc = torch.flip(x_b6, [1, 2])
        x_b6 = self.conv0(torch.concat([x_b6, x_rc], dim=-1))

        x_rc = torch.flip(x_cast, [1, 2])
        x_cast = self.conv0(torch.concat([x_cast, x_rc], dim=-1))

        if mask is not None:
            x_b6[:, mask, :] = 0
            x_cast[:, mask, :] = 0
        # print("b6 shape", x_b6.shape)
        x_b6 = self.GloblePooling(x_b6)
        x_cast = self.GloblePooling(x_cast)

        x_b6 = x_b6.flatten(1)
        x_cast = x_cast.flatten(1)

        x_b6 = self.fc0(x_b6)
        x_cast = self.fc0(x_cast)

        x_ratio = torch.cat([x_b6, x_cast], dim=-1)
        for layer in self.ratio_fclayers:
            x_ratio = layer(x_ratio)

        x_ratio = self.ratio_out(x_ratio)

        # print(x1.shape)
        for layer in self.fc_layers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = self.counts_out(x_b6)
        x_cast = self.counts_out(x_cast)

        return torch.cat([x_b6, x_cast, x_ratio], dim=-1)

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        return self(batch[0], mask=self.mask, activation=self.activation)


class SingleHeadResidualCNN(Base):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=10,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=330,
        filter_number=256,
        pooling_type="avg",
        learning_rate=1e-3,
        conv_repeat=1,
        dilation=1,
        scheduler="cycle",
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """

        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
        )

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )
        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                            dilation=dilation,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())

    def forward(self, x, activation=False):
        x = torch.permute(x, (0, 2, 1))

        x_rc = torch.flip(x, [1, 2])
        x1 = self.conv0(torch.concat([x, x_rc], dim=-1))
        # x1 = self.conv0(x)
        if activation:
            return x1

        for layer in self.convlayers:
            x1 = layer(x1)

        x1 = x1.flatten(1)
        x1 = self.fc0(x1)
        # print(x1.shape)
        for layer in self.fclayers:
            x1 = layer(x1)

        x1 = self.out(x1)
        # x1 = x1.flatten()
        return x1

    def validation_epoch_end(self, validation_step_outputs):
        all_val_result = torch.cat(validation_step_outputs, dim=0)
        print("----", all_val_result.shape)
        predict_index = 0
        label_index = 1
        prediction = all_val_result[:, predict_index, 0]
        label = all_val_result[:, label_index, 0]

        # convert to numpy
        prediction = prediction.numpy()
        label = label.numpy()

        print(prediction.shape)
        prediction = np.split(prediction, 2, axis=0)
        label = np.split(label, 2, axis=0)

        prediction = np.stack(prediction, axis=1)
        label = np.stack(label, axis=1)

        val_count1_corr = stats.pearsonr(prediction[:, 0], label[:, 0]).statistic
        val_count2_corr = stats.pearsonr(prediction[:, 1], label[:, 1]).statistic
        val_ratio_corr = stats.pearsonr(
            prediction[:, 0] - prediction[:, 1], label[:, 0] - label[:, 1]
        ).statistic

        self.log("val_count1_corr", val_count1_corr)
        self.log("val_count2_corr", val_count2_corr)
        self.log("val_ratio_corr", val_ratio_corr)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        # result = torch.stack([y_hat.cpu().detach(), y.cpu().detach()], dim=1)
        return torch.stack([y_hat.cpu().detach(), y.cpu().detach()], dim=1)


class MultiHeadResidualCNN(Base):
    def __init__(
        self,
        head=3,
        kernel_number=512,
        kernel_length=10,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=330 * 2,
        filter_number=256,
        pooling_type="avg",
        learning_rate=1e-3,
        conv_repeat=1,
        dilation=1,
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """

        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
        )

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )
        fc_dim = input_length * 2
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        nn.Sequential(
                            ConvBlock(
                                filter_number,
                                filter_number,
                                kernel_size,
                                padding=padding,
                                dilation=dilation,
                            ),
                            ConvBlock(
                                filter_number,
                                filter_number,
                                kernel_size,
                                padding=padding,
                                dilation=dilation,
                            ),
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Linear(hidden_size, head)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1, 3))

        x_b6 = x[:, :, :, 0]
        x_cast = x[:, :, :, 1]

        x_rc = torch.flip(x_b6, [1, 2])
        x_b6 = self.conv0(torch.concat([x_b6, x_rc], dim=-1))

        x_rc = torch.flip(x_cast, [1, 2])
        x_cast = self.conv0(torch.concat([x_cast, x_rc], dim=-1))

        x1 = torch.cat([x_b6, x_cast], dim=-1)

        # x1 = self.conv0(x)

        for layer in self.convlayers:
            x1 = layer(x1)

        x1 = x1.flatten(1)
        x1 = self.fc0(x1)
        # print(x1.shape)
        for layer in self.fclayers:
            x1 = layer(x1)

        x1 = self.out(x1)

        return x1


# TODO Check the implementation of the transformer
# If pos_enc_type="relative", use relative positional encoding. (borrow code from enformer-torch, mimic architecture in Vaishnav et. al, 2022)
# If pos_enc_type="sin_cos", use sin-cos positional encoding. (Use sin-cos positional encoding from Attention is All You Need.)
# If pos_enc_type="lookup_table", use embedding lookup table.
class Transformer(Base):
    def __init__(
        self,
        head=3,
        kernel_number=512,
        kernel_length=7,
        filter_number=256,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        conv_repeat=1,
        hidden_size=256,
        dropout=0.2,
        h_layers=2,
        input_length=330 * 2,
        pooling_type="avg",
        padding="same",
        attention_layers=2,
        learning_rate=1e-3,
        num_rel_pos_features=66,  # this hyperparam is specifically for relative positional encoding
        pos_enc_type="relative",  # "sin_cos", "lookup_table"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pos_enc_type = pos_enc_type

        # This conv layer is appied on both forward and RC strands
        self.conv0 = ConvBlock(4, kernel_number, kernel_length, padding=padding)

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )

        fc_dim = input_length * 2
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        if pos_enc_type == "sin_cos":
            self.position_enc = PositionalEncoding(filter_number, n_position=fc_dim)
            self.layer_norm = nn.LayerNorm(filter_number)
        elif pos_enc_type == "lookup_table":
            self.embedding_table = nn.Embedding(
                num_embeddings=fc_dim, embedding_dim=filter_number
            )
            self.layer_norm = nn.LayerNorm(filter_number)

        self.attentionlayers = nn.ModuleList()

        if pos_enc_type == "relative":
            for layer in range(attention_layers):
                self.attentionlayers.append(
                    nn.Sequential(
                        Residual(
                            Attention(
                                dim=filter_number,  # dimension of the last out channel
                                num_rel_pos_features=num_rel_pos_features,
                            ),
                        ),
                        nn.LayerNorm(filter_number),
                        Residual(
                            nn.Sequential(
                                nn.Linear(filter_number, filter_number * 2),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                nn.Linear(filter_number * 2, filter_number),
                                nn.Dropout(dropout),
                            )
                        ),
                        nn.LayerNorm(filter_number),
                    )
                )
        else:
            for layer in range(attention_layers):
                self.attentionlayers.append(
                    nn.Sequential(
                        Residual(
                            Attention_2(
                                dim=filter_number,  # dimension of the last out channel
                            ),
                        ),
                        nn.LayerNorm(filter_number),
                        Residual(
                            nn.Sequential(
                                nn.Linear(filter_number, filter_number * 2),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                nn.Linear(filter_number * 2, filter_number),
                                nn.Dropout(dropout),
                            )
                        ),
                        nn.LayerNorm(filter_number),
                    )
                )

        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Linear(hidden_size, head)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1, 3))

        x_b6 = x[:, :, :, 0]
        x_cast = x[:, :, :, 1]

        x_rc = torch.flip(x_b6, [1, 2])
        x_b6 = self.conv0(torch.concat([x_b6, x_rc], dim=-1))

        x_rc = torch.flip(x_cast, [1, 2])
        x_cast = self.conv0(torch.concat([x_cast, x_rc], dim=-1))

        x1 = torch.cat([x_b6, x_cast], dim=-1)

        for layer in self.convlayers:
            x1 = layer(x1)

        x1 = torch.permute(x1, (0, 2, 1))
        if self.pos_enc_type == "sin_cos":
            # Add positional encoding
            x1 = self.position_enc(x1)
            x1 = self.layer_norm(x1)
        elif self.pos_enc_type == "lookup_table":
            # Add positional encoding
            idx = torch.arange(x1.shape[1]).to(x1.device)
            x1 = x1 + self.embedding_table(idx)
            x1 = self.layer_norm(x1)

        # Attention layer
        for layer in self.attentionlayers:
            x1 = layer(x1)

        # flatten
        x1 = x1.flatten(1)
        x1 = self.fc0(x1)
        for layer in self.fclayers:
            x1 = layer(x1)

        x1 = self.out(x1)
        return x1


# Use sin-cos positional encoding from Attention is All You Need.
# Positional Encoding is only added once.
class PositionalEncoding(nn.Module):
    def __init__(self, encoding_dim, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.pos_table = self._get_sinusoid_encoding_table(n_position, encoding_dim)

    def _get_sinusoid_encoding_table(self, n_position, encoding_dim):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / encoding_dim)
                for hid_j in range(encoding_dim)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


# Transformer_split_end
# Use relative positional encoding.
# This transformer follows the new idea that we need to have different fc layers for different output tasks.
class Separate_Transformer(Base):
    def __init__(
        self,
        head=3,
        kernel_number=512,
        kernel_length=7,
        filter_number=256,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        conv_repeat=1,
        hidden_size=256,
        dropout=0.2,
        h_layers=2,
        input_length=330 * 2,
        pooling_type="avg",
        padding="same",
        attention_layers=2,
        learning_rate=1e-3,
        num_rel_pos_features=66,
    ):
        super().__init__()
        self.save_hyperparameters()

        # This conv layer is appied on both forward and RC strands
        self.conv0 = ConvBlock(4, kernel_number, kernel_length, padding=padding)

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )

        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.attentionlayers = nn.ModuleList()

        for layer in range(attention_layers):
            self.attentionlayers.append(
                nn.Sequential(
                    Residual(
                        Attention(
                            dim=filter_number,  # dimension of the last out channel
                            num_rel_pos_features=num_rel_pos_features,
                        ),
                    ),
                    nn.LayerNorm(filter_number),
                    Residual(
                        nn.Sequential(
                            nn.Linear(filter_number, filter_number * 2),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                            nn.Linear(filter_number * 2, filter_number),
                            nn.Dropout(dropout),
                        )
                    ),
                    nn.LayerNorm(filter_number),
                )
            )

        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )

        self.ratio_fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_size, 2 * hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for layer in range(h_layers)
        )

        self.ratio_out = nn.Sequential(nn.Linear(2 * hidden_size, 1))
        self.counts_out = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1, 3))

        x_b6 = x[:, :, :, 0]
        x_cast = x[:, :, :, 1]

        x_rc = torch.flip(x_b6, [1, 2])
        x_b6 = self.conv0(torch.concat([x_b6, x_rc], dim=-1))

        x_rc = torch.flip(x_cast, [1, 2])
        x_cast = self.conv0(torch.concat([x_cast, x_rc], dim=-1))

        for layer in self.convlayers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = torch.permute(x_b6, (0, 2, 1))
        x_cast = torch.permute(x_cast, (0, 2, 1))

        # Attention layer
        for layer in self.attentionlayers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = x_b6.flatten(1)
        x_cast = x_cast.flatten(1)

        x_b6 = self.fc0(x_b6)
        x_cast = self.fc0(x_cast)

        x_ratio = torch.cat([x_b6, x_cast], dim=-1)  ####
        for layer in self.ratio_fclayers:
            x_ratio = layer(x_ratio)

        x_ratio = self.ratio_out(x_ratio)

        for layer in self.fclayers:
            x_b6 = layer(x_b6)
            x_cast = layer(x_cast)

        x_b6 = self.counts_out(x_b6)
        x_cast = self.counts_out(x_cast)

        return torch.cat([x_b6, x_cast, x_ratio], dim=-1)
