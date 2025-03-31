# DeepAllele Scripts

This directory contains scripts for data preprocessing and model training for the DeepAllele project, a deep learning framework for analyzing allele-specific gene expression and chromatin accessibility.

## Directory Structure

- `Preprocess/`: Scripts for preprocessing different types of sequencing data
- `ModelTraining/`: Scripts for training various models
- `Process/`: Additional processing utilities

## Environment Setup

Before running any script, set up the appropriate environment variables to configure paths for your environment:

```bash
# Set main data and output directories
export DATA_DIR="/path/to/your/data"
export OUTPUT_DIR="/path/to/output"

# For model training with Weights & Biases
export WANDB_API_KEY="your-wandb-key"
```

## Preprocessing Scripts

### RNA-seq Preprocessing

Process RNA-seq data with a specified window size:

```bash
cd Preprocess
./run_rna_preprocess.sh [window_size]
```

Default window size is 5000 bp if not specified.

**Key Parameters:**
- `VCF_FILE`: VCF file with variants
- `GENE_META`: Gene metadata file
- `GENOME_FILE`: Reference genome FASTA
- `CAST_EXPR`: CAST expression data file
- `B6_EXPR`: B6 expression data file

### ChIP-seq Preprocessing

Process ChIP-seq data with a specified sequence length:

```bash
cd Preprocess
./run_chip_preprocess.sh [seq_length]
```

Default sequence length is 551 bp if not specified.

**Key Parameters:**
- `SEQUENCE_PATH`: Base path for sequence FASTA files
- `C57_PWK_CHIP`: Path to C57-PWK ChIP data file
- `C57_SPRET_CHIP`: Path to C57-SPRET ChIP data file

### ATAC-seq Preprocessing

Process ATAC-seq data:

```bash
cd Preprocess
./run_atac_preprocess.sh
```

**Key Parameters:**
- `INPUT_CSV`: Path to input CSV file with ATAC-seq data
- `RAW_DATA_PATH`: Base path to raw data directory
- `B6_SEQ_PATH`: Path to B6 sequences FASTA file
- `CAST_SEQ_PATH`: Path to CAST sequences FASTA file
- `PEAKS_INFO_PATH`: Path to peak information file

## Model Training Scripts

### RNA-seq Model Training

Train models on RNA-seq data:

```bash
cd ModelTraining
./run_rna_models.sh
```

This will train single, multi, and multi-computed ratio models for specified batch IDs.

**Key Parameters:**
- `RNA_DATA`: Path to preprocessed RNA-seq HDF5 file
- `RNA_BATCHES`: Array of batch IDs to process
- `OUT_ROOT`: Root directory for output models
- `DEVICE`: GPU device ID to use (default: 0)

### ChIP-seq Model Training

Train models on ChIP-seq data for different strains:

```bash
cd ModelTraining
./run_chip_models.sh
```

This will train single and multi models for SPRET and PWK strains.

**Key Parameters:**
- `SPRET_DATA`: Path to SPRET ChIP-seq data
- `PWK_DATA`: Path to PWK ChIP-seq data
- `DEVICE`: GPU device ID to use (default: 0)

### ATAC-seq Model Training

Train models on ATAC-seq data:

```bash
cd ModelTraining
./run_atac_models.sh
```

This will train single, multi, and multi-computed ratio models for specified batch IDs.

**Key Parameters:**
- `ATAC_DATA`: Path to preprocessed ATAC-seq HDF5 file
- `ATAC_BATCHES`: Array of batch IDs to process
- `DEVICE`: GPU device ID to use (default: 0)

## Processing Utilities

The `Process/` directory contains additional utilities such as:

- `insert_variants.py`: Insert variants into reference sequences

Example usage:

```bash
cd Process
python insert_variants.py -f [fasta_file] -s [snp_file] -d [indel_file] -o [output_fasta_path] [--filter_pass]
```

## Model Configuration

All model training scripts support various hyperparameters:

- `--conv_layers`: Number of convolutional layers
- `--kernel_number`: Number of kernels in convolutional layers
- `--kernel_length`: Length of kernels
- `--filter_number`: Number of filters
- `--kernel_size`: Size of kernels
- `--pooling_size`: Size of pooling windows
- `--h_layers`: Number of hidden layers
- `--hidden_size`: Size of hidden layers
- `--learning_rate`: Learning rate for optimization
- `--random_seed_start`/`--random_seed_end`: Range of random seeds for multiple runs

## Tracking Experiments

The training scripts use Weights & Biases (W&B) for experiment tracking. Set your W&B API key in the `WANDB_API_KEY` environment variable or directly in the script.
