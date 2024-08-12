## MTX counts preprocess

```bash
python mtx_counts_preprocess.py --in_dir='/data/tuxm/project/F1-ASCA/data/raw_data/'
```

## OCR preprocessing

Extract the sequences from `maternal.fa` and `paternal.fa` and combine the sequence information with the counts information from `maternal.h5ad` and `paternal.h5ad` to create a file `.hdf5` file.

```bash
python OCR_preprocess.py --in_dir='/data/tuxm/project/F1-ASCA/data/raw_data/' --in_fa='/homes/gws/tuxm/Project/ASCC/data/sequences/' --maternal_fa='final_peaks_cast_unc_2021_11_08_B6SEQS.fa' --paternal_fa='final_peaks_cast_unc_2021_11_08_shifted_to_CASTEIJ.fa' --out_dir='/data/tuxm/project/F1-ASCA/data/input/extendpeaks_processed/' --out_file='extendpeaks2x_500_processed.hdf5'
```

```bash
python OCR_preprocess.py --in_dir='/data/tuxm/project/F1-ASCA/data/raw_data/' --in_fa='/data/tuxm/project/F1-ASCA/data/raw_data/extendedpeaks/' --maternal_fa='peaks2x_500_B6SEQS.fa' --paternal_fa='peaks2x_500_CASTSEQS.fa' --out_dir='/data/tuxm/project/F1-ASCA/data/input/extendpeaks_processed/' --out_file='extendpeaks2x_500_processed.hdf5'
```

```bash
python OCR_preprocess.py --in_dir='/data/tuxm/project/F1-ASCA/data/raw_data/' --in_fa='/data/tuxm/project/F1-ASCA/data/raw_data/extendedpeaks/' --maternal_fa='peaks4x_1000_B6SEQS.fa' --paternal_fa='peaks4x_1000_CASTSEQS.fa' --out_dir='/data/tuxm/project/F1-ASCA/data/input/extendpeaks_processed/' --out_file='extendpeaks4x_1000_processed.hdf5'
```

```bash
python OCR_preprocess.py --in_dir='/data/tuxm/project/F1-ASCA/data/raw_data/' --in_fa='/data/tuxm/project/F1-ASCA/data/raw_data/extendedpeaks/' --maternal_fa='peaks8x_2000_B6SEQS.fa' --paternal_fa='peaks8x_2000_CASTSEQS.fa' --out_dir='/data/tuxm/project/F1-ASCA/data/input/extendpeaks_processed/' --out_file='extendpeaks8x_2000_processed.hdf5'
```

## Models and Hyperparameters

### MultiHead_Residual_CNN

2 convolutions inside the residual block.

```bash
python ./train_model/train_cnn_model.py --in_folder='/data/tuxm/project/F1-ASCA/data/seuqence_datasets_new_cast.hdf5' --out_folder='../output/multiheadResidualCNN/' --model_type=Multihead_Residual_CNN --conv_layers=6 --conv_repeat=1 --kernel_number=1024 --kernel_length=7 --filter_number=512 --kernel_size=5 --pooling_size=2
```

### Separate_Multihead_Residual_CNN

This CNN follows the new idea that we need to have different fc layers for different output tasks.

```bash
python ./train_model/train_cnn_model.py --in_folder='/data/tuxm/project/F1-ASCA/data/seuqence_datasets_new_cast.hdf5' --out_folder='../output/separate_multiheadResidualCNN/' --model_type=Separate_Multihead_Residual_CNN --conv_layers=6 --conv_repeat=2 --kernel_number=1024 --kernel_length=7 --filter_number=512 --kernel_size=5 --pooling_size=2
```

### Transformer (Use relative positional encoding)

```bash
python ./train_model/train_transformer_model.py --in_folder='/data/tuxm/project/F1-ASCA/data/seuqence_datasets_new_cast.hdf5' --out_folder='../output/tf_relative/' --model_type=Transformer --pos_enc_type=relative --attention_layers=2 --num_rel_pos_features=66 --conv_layers=6 --conv_repeat=1 --kernel_number=1024 --kernel_length=7 --filter_number=512 --kernel_size=5 --pooling_size=2
```

### Transformer (Use sin-cos positional encoding)

Use sin-cos positional encoding from Attention is All You Need. Positional Encoding is only added once.

```bash
python ./train_model/train_transformer_model.py --in_folder='/data/tuxm/project/F1-ASCA/data/seuqence_datasets_new_cast.hdf5' --out_folder='../output/tf_sin_cos/' --model_type=Transformer --pos_enc_type=sin_cos --attention_layers=2 --conv_layers=6 --conv_repeat=1 --kernel_number=1024 --kernel_length=7 --filter_number=512 --kernel_size=5 --pooling_size=2
```

### Transformer (Use lookup table)

Use embedding lookup table as positional encoding.

```bash
python ./train_model/train_transformer_model.py --in_folder='/data/tuxm/project/F1-ASCA/data/seuqence_datasets_new_cast.hdf5' --out_folder='../output/tf_lookup_table/' --model_type=Transformer --pos_enc_type=lookup_table --attention_layers=2 --conv_layers=6 --conv_repeat=1 --kernel_number=1024 --kernel_length=7 --filter_number=512 --kernel_size=5 --pooling_size=2
```

### Separate_Transformer

Use relative positional encoding. This transformer follows the new idea that we need to have different fc layers for different output tasks.

```bash
python ./train_model/train_transformer_model.py --in_folder='/data/tuxm/project/F1-ASCA/data/seuqence_datasets_new_cast.hdf5' --out_folder='../output/separate_tf/' --model_type=Separate_Transformer --attention_layers=2 --conv_layers=6 --conv_repeat=1 --kernel_number=1024 --kernel_length=7 --filter_number=512 --kernel_size=5 --pooling_size=2
```
