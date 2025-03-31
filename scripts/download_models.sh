#!/bin/bash

set -e
set -u
set -x

# Download ckpt files for trained models
model_links=''
process_data=''

# Setup directories
datadir=data/
mkdir -p "${datadir}atac/models/mh/init0/"
mkdir -p "${datadir}chip/models/mh/init0/"
mkdir -p "${datadir}rna/models/mh/init0/"
mkdir -p "${datadir}atac/models/sh/init0/"
mkdir -p "${datadir}chip/models/sh/init0/"
mkdir -p "${datadir}rna/models/sh/init0/"

mkdir -p "${datadir}atac/data/"
mkdir -p "${datadir}chip/data/"
mkdir -p "${datadir}rna/data/"

mkdir -p "${datadir}atac/results/mh/init0/"
mkdir -p "${datadir}chip/results/mh/init0/"
mkdir -p "${datadir}rna/results/mh/init0/"
mkdir -p "${datadir}atac/results/sh/init0/"
mkdir -p "${datadir}chip/results/sh/init0/"
mkdir -p "${datadir}rna/results/sh/init0/"


for link in $model_links
do
	cd modeldir
	wget $link
done

for link in $process_data
do
        cd modeldir
        wget $link
done

