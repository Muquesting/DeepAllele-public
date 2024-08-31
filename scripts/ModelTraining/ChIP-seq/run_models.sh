#!/bin/bash

data_path="/data/tuxm/project/F1-ASCA/data/input/Chip-seq/processed_data/sequence_datasets_chip_SPRET_B6_20230126.hdf5"

python reproduce_multi_head_models.py --in_folder="$data_path" --out_folder=./Chip-seq/multi-head/SPRET/ --conv_layers=6 --conv_repeat=1 --kernel_length=15 --pooling_size=2 --kernel_size=5 --random_seed_start=0 --random_seed_end=10 --device=0

python reproduce_multi_head_models.py --in_folder="$data_path" --out_folder=./Chip-seq/multi-head/PWK/ --conv_layers=6 --conv_repeat=1 --kernel_length=15 --pooling_size=2 --kernel_size=5 --random_seed_start=0 --random_seed_end=10 --device=1

python reproduce_single_head_models.py --in_folder="$data_path" --out_folder=./Chip-seq/single-head/SPRET/ --conv_layers=6 --conv_repeat=1 --kernel_length=15 --pooling_size=2 --kernel_size=5 --random_seed_start=0 --random_seed_end=10 --device=2

python reproduce_single_head_models.py --in_folder="$data_path" --out_folder=./Chip-seq/single-head/PWK/ --conv_layers=6 --conv_repeat=1 --kernel_length=15 --pooling_size=2 --kernel_size=5 --random_seed_start=0 --random_seed_end=10 --device=3
