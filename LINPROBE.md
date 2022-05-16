## Linear probe GCMAE 

```
python main_linprobe.py \
    --data_path_train path/to/train/data \
    --data_path_val path/to/val/data \
    --nb_classes 2 \
    --output_dir path/to/ouput/dir \
    --log_dir path/to/log/dir \
    --batch_size 512 \
    --model vit_base_patch16 \
    --epochs 90 \
    --finetune path/to/pth/path
```

