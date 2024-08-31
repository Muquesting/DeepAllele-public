#!/bin/bash

python reproduce_cnn_models.py --in_folder=/data/tuxm/project/F1-ASCA/data/input/bulk_seq_ATAC_preprocessed_new_20230126.hdf5 --out_folder=./ATAC/sum/multi-head/ --conv_layers=4 --random_seed_start=0 --random_seed_end=10 --device=2 --batch_id=sum

python reproduce_cnn_models.py --in_folder=/data/tuxm/project/F1-ASCA/data/input/bulk_seq_ATAC_preprocessed_new_20230126.hdf5 --out_folder=./ATAC/sc/multi-head/ --conv_layers=4 --random_seed_start=0 --random_seed_end=10 --device=3 --batch_id=sc

python reproduce_single_head_models.py --in_folder=/data/tuxm/project/F1-ASCA/data/input/bulk_seq_ATAC_preprocessed_new_20230126.hdf5 --out_folder=./ATAC/sum/single-head/ --conv_layers=4 --random_seed_start=5 --random_seed_end=10 --device=0 --batch_id=sum

python reproduce_single_head_models.py --in_folder=/data/tuxm/project/F1-ASCA/data/input/bulk_seq_ATAC_preprocessed_new_20230126.hdf5 --out_folder=./ATAC/sc/single-head/ --conv_layers=4 --random_seed_start=5 --random_seed_end=10 --device=1 --batch_id=sc

