export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=5 python tools/eval.py --resume_from "/home/dense/daniel/drivestudio/drivestudio/outputs/drivestudio/drivestudio_night_scenes/test_displacement_scale_0_05/checkpoint_final.pth"