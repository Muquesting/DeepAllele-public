#!/bin/bash

set -e
set -u
set -x

# All the data for this project can be downloaded from Figshare:
# https://figshare.com/articles/dataset/Genome_files/28694384
# Download ckpt files for trained models
model_links='https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint'
# Download processed data as hdf5 files
process_data='https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Processed_data'

# Setup directories
datadir=data/


mkdir -p "${datadir}atac/data/"

wget -O ${datadir}atac/data/ATAC-seq-preprocessed.hdf5 https://figshare.com/ndownloader/files/53316875
mkdir -p "${datadir}atac/sc/results/mh/init0/"
mkdir -p "${datadir}atac/sc/results/sh/init0/"
mkdir -p "${datadir}atac/sc/models/mh/init0/"
mkdir -p "${datadir}atac/sc/models/sh/init0/"
wget -O ${datadir}atac/sc/models/sh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FATAC%2Fsc%2Fsh%2F4_1_256_15_256_5_4_0.0001_2_256_True
wget -O ${datadir}atac/sc/models/mh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FATAC%2Fsc%2Fmh%2F4_1_256_15_256_5_4_0.0001_2_256_True
mkdir -p "${datadir}atac/bulk/results/mh/init0/"
mkdir -p "${datadir}atac/bulk/results/sh/init0/"
mkdir -p "${datadir}atac/bulk/models/mh/init0/"
mkdir -p "${datadir}atac/bulk/models/sh/init0/"
wget -O ${datadir}atac/bulk/models/sh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FATAC%2Fsum%2Fsh%2F4_1_256_15_256_5_4_0.0001_2_256_True
wget -O ${datadir}atac/bulk/models/mh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FATAC%2Fsum%2Fmh%2F4_1_256_15_256_5_4_0.0001_2_256_True


mkdir -p "${datadir}rna/data/"
wget -O ${datadir}rna/data/RNA-seq-preprocessed.hdf5 https://figshare.com/ndownloader/files/53316854

mkdir -p "${datadir}rna/Bcell/models/mh/init0/"
mkdir -p "${datadir}rna/Bcell/models/sh/init0/"
mkdir -p "${datadir}rna/Bcell/results/mh/init0/"
mkdir -p "${datadir}rna/Bcell/results/sh/init0/"
wget -O ${datadir}rna/Bcell/models/sh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FRNA%2Fsingle%2FB_Fo_Sp_IL4%2F4_1_512_10_256_3_5_0.0001_2_512_True
wget -O ${datadir}rna/Bcell/models/mh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FRNA%2Fmulti%2FB_Fo_Sp_IL4%2F4_1_512_10_256_3_5_0.0001_2_512_True

mkdir -p "${datadir}rna/MFcell/models/mh/init0/"
mkdir -p "${datadir}rna/MFcell/models/sh/init0/"
mkdir -p "${datadir}rna/MFcell/results/mh/init0/"
mkdir -p "${datadir}rna/MFcell/results/sh/init0/"
wget -O ${datadir}rna/MFcell/models/sh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FRNA%2Fsingle%2FMF_PC_IL4%2F4_1_512_10_256_3_5_0.0001_2_512_True
wget -O ${datadir}rna/MFcell/models/mh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FRNA%2Fmulti%2FMF_PC_IL4%2F4_1_512_10_256_3_5_0.0001_2_512_True



mkdir -p "${datadir}chip/data/SPRET/"
wget -O ${datadir}chip/data/SPRET/sequence_datasets_chip_SPRET_B6.hdf5 https://figshare.com/ndownloader/files/53316869
mkdir -p "${datadir}chip/data/PWK/"
wget -O ${datadir}chip/data/PWK/sequence_datasets_chip_PWK_B6.hdf5 https://figshare.com/ndownloader/files/53316863

mkdir -p "${datadir}chip/SPRET/models/mh/init0/"
mkdir -p "${datadir}chip/SPRET/models/sh/init0/"
mkdir -p "${datadir}chip/SPRET/results/mh/init0/"
mkdir -p "${datadir}chip/SPRET/results/sh/init0/"
wget -O ${datadir}chip/SPRET/models/sh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FChip%2FSPRET%2Fsh%2F6_1_256_15_256_5_2_0.0001_2_256_True
wget -O ${datadir}chip/SPRET/models/mh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FChip%2FSPRET%2Fmh%2F6_1_256_15_256_5_2_0.0001_2_256_True

mkdir -p "${datadir}chip/PWK/models/mh/init0/"
mkdir -p "${datadir}chip/PWK/models/sh/init0/"
mkdir -p "${datadir}chip/PWK/results/mh/init0/"
mkdir -p "${datadir}chip/PWK/results/sh/init0/"
wget -O ${datadir}chip/PWK/models/sh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FChip%2FPWK%2Fsh%2F6_1_256_15_256_5_2_0.0001_2_256_True
wget -O ${datadir}chip/PWK/models/mh/init0/model.ckpt https://figshare.com/ndownloader/articles/28694384/versions/3?folder_path=Model_checkpoint%2FChip%2FPWK%2Fmh%2F6_1_256_15_256_5_2_0.0001_2_256_True

