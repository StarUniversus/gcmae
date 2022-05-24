## Fine tune GCMAE 

```
python main_finetune.py \
    --data_path path/to/data \
    --nb_classes 9 \
    --output_dir path/to/ouput/dir \
    --log_dir path/to/log/dir \
    --batch_size 128 \
    --model vit_base_patch16 \
    --epochs 50 \
    --finetune path/to/pth/path \
```
