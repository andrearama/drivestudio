segformer_path=/home/dense/daniel/drivestudio/drivestudio/SegFormer

python datasets/tools/extract_masks.py \
    --data_root /external/10g/carlnas/fs1/datasets/waymo/data/waymo/processed/training \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --scene_ids 339 350 124 \
    --process_dynamic_mask