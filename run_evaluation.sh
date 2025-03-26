export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=2 python tools/eval.py \
    --resume_from "/home/dense/daniel/drivestudio/drivestudio/outputs/drivestudio/drivestudio_night_scenes/depth_map_front_back/checkpoint_final.pth" \
