## Pre-training GCMAE

To pre-train ViT-Base (recommended default)
```
python main_pretrain.py \
    --data_path path/to/data \
    --data_val_path path/to/data \
    --output_dir path/to/ouput/dir \
    --log_dir path/to/log/dir \
    --batch_size 128 \
    --model gcmae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.5 \
    --epochs 80 \
    --warmup_epochs 40 \
    --blr 1e-3 --weight_decay 0.05 \
    --low_dim 768 \
    --nce_k 8192 \
    --nce_t 0.07 \
    --nce_m 0.5 \
```