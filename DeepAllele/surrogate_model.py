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


class SeparateMultiHeadResidualCNN_DeepliftSurrogate(Base):
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

        self.conv0_b6 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
            batch_norm=first_batch_norm,
        )
        
        self.conv0_cast = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
            batch_norm=first_batch_norm,
        )

        self.convlayers_b6 = nn.ModuleList()
        self.convlayers_cast = nn.ModuleList()

        self.convlayers_b6.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
                batch_norm=all_batch_norm,
            )
        )
        self.convlayers_cast.append(
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
                self.convlayers_b6.append(
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
                
                self.convlayers_cast.append(
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
                self.convlayers_b6.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers_b6.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers_b6.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers_b6.append(SoftmaxPool(pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )
                
            if pooling_type == "max":
                self.convlayers_cast.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers_cast.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers_cast.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers_cast.append(SoftmaxPool(pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.fc0_b6 = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )
        self.fc0_cast = nn.Sequential(
            nn.Linear(fc_dim * filter_number, hidden_size), nn.ReLU()
        )

        self.fclayers_b6 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        
        self.fclayers_cast = nn.ModuleList(
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
        self.counts_out_b6 = nn.Sequential(nn.Linear(hidden_size, 1))
        self.counts_out_cast = nn.Sequential(nn.Linear(hidden_size, 1))


    def forward(self, x, mask=None, activation=False):
        x = torch.permute(x, (0, 2, 1, 3))

        x_b6 = x[:, :, :, 0]
        x_cast = x[:, :, :, 1]

        if activation:
            x_b6 = self.conv0_b6(x_b6)
            x_cast = self.conv0_cast(x_cast)
            # return the stack of the two activations
            return torch.stack([x_b6, x_cast], dim=-1)

        x_rc = torch.flip(x_b6, [1, 2])
        x_b6 = self.conv0_b6(torch.concat([x_b6, x_rc], dim=-1))

        x_rc = torch.flip(x_cast, [1, 2])
        x_cast = self.conv0_cast(torch.concat([x_cast, x_rc], dim=-1))

        if mask is not None:
            x_b6[:, mask, :] = 0
            x_cast[:, mask, :] = 0

        for layer in self.convlayers_b6:
            x_b6 = layer(x_b6)
            
        for layer in self.convlayers_cast:
            x_cast = layer(x_cast)

        x_b6 = x_b6.flatten(1)
        x_cast = x_cast.flatten(1)

        x_b6 = self.fc0_b6(x_b6)
        x_cast = self.fc0_cast(x_cast)

        x_ratio = torch.cat([x_b6, x_cast], dim=-1)
        for layer in self.ratio_fclayers:
            x_ratio = layer(x_ratio)

        x_ratio = self.ratio_out(x_ratio)

        for layer in self.fclayers_b6:
            x_b6 = layer(x_b6)
            
        for layer in self.fclayers_cast:
            x_cast = layer(x_cast)

        x_b6 = self.counts_out_b6(x_b6)
        x_cast = self.counts_out_cast(x_cast)

        return torch.cat([x_b6, x_cast, x_ratio], dim=-1)

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        return self(batch[0], mask=self.mask, activation=self.activation)