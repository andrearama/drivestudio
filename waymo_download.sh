export PYTHONPATH=$(pwd)
INPUT_PATH="/external/10g/carlnas/fs1/datasets/waymo/data/waymo/raw"

CUDA_VISIBLE_DEVICES=7 python datasets/waymo/waymo_download.py \
    --target_dir $INPUT_PATH \
    --scene_ids 351 352 353 354 355 356 357 358 359 360 \