segformer_path=/home/dense/daniel/drivestudio/drivestudio/SegFormer

split=trainval

CUDA_VISIBLE_DEVICES=6 python datasets/tools/extract_masks.py \
    --data_root data/nuscenes/processed_10Hz/$split \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --start_idx 832 \
    --num_scenes 10 \
    --process_dynamic_mask