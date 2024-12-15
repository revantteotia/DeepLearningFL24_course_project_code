Code repository for Deep Learning Fall 2024 course

Project title: K-space representation learning

Student: Revant Teotia (rt2741@nyu.edu)


**To setup environment**
```
# create new env kt
$ conda create -n kt python=3.8.5

# activate kt
$ conda activate kt

# install dependencies
$ pip install -r requirements.txt
```

**Code structure**
- ```dataloaders/imagenet_radial_kspace_dataset_hdf5.py``` has pytorch dataset class. It requires the radial spokes k-space data to be precomputed on disk in a .hdf5 file.
- ```models/radial_kspace_transformer.py``` has code for k-space MAE model. It uses transformer model defined in ```models/radial_kspace_transformer.py```.
-```train_radial_kspace_mae.py``` has the training script to train MAE model.
- ```ftune_mae_encoder.py``` has the training script to finetune the pre-trained MAE encoder model.

**To run training sscripts**
(Note that it would require all the imagenet radial k-space data pre-computed on the disk)

To train MAE: 


```
accelerate launch train_radial_kspace_mae.py \
        --log_scale_amplitude \
        --min_max_normalize \
        --embed_dim=768 \
        --model_depth=12 \
        --num_heads=12  \
        --decoder_embed_dim=512 \
        --decoder_model_depth=4 \
        --decoder_num_heads=4  \
        --num_epochs 300 \
        --mask_ratio=0.70 
```

To finetune pre-trained encoder: 
```
accelerate launch ftune_mae_encoder.py \
        --embed_dim=768 \
        --model_depth=12 \
        --num_heads=12 \
        --spokes_to_use 256 \
        --num_epochs 300 \
        --do_initial_eval \
        --log_scale_amplitude \
        --min_max_normalize \
        --lr 0.003 \
        --pre_trained_checkpoint path_to_checkpoint_on_disk
```
