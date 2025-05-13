export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=7 python datasets/preprocess.py \
    --data_root /external/10g/carlnas/fs1/datasets/waymo/data/waymo/raw \
    --target_dir /external/10g/carlnas/fs1/datasets/waymo/data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_ids 351 352 353 354 355 356 357 358 359 \
    --process_keys images lidar calib pose dynamic_masks objects \
    --vsdebug False \