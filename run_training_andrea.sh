export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

output_root="outputs/drivestudio/"
project="drivestudio_night_scenes"
expname="test1_1cam_retry_allosses_normals_3"
scene_idx=814
steps=$(seq 0 5 190)

test_timesteps="[$(echo $steps | tr ' ' ',')]"

CUDA_VISIBLE_DEVICES=1,2 python tools/train.py \
    --config_file configs/omnire_extended_cam.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=nuscenes/1cams \
    data.data_root="/home/dense/daniel/drivestudio/drivestudio/data/nuscenes/processed_10Hz/trainval" \
    data.pixel_source.load_smpl="false" \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    trainer.render.avg_renderings="True"\
    trainer.learn_fixednoise="True"\
    trainer.model_flare="True"\
    trainer.use_emitted="True"\
    trainer.use_normals="True"\
    data.pixel_source.use_depth_map_front="False"\
    data.pixel_source.use_depth_map_back="False"\
    logging.vis_freq=10000\
    data.pixel_source.test_timesteps="$test_timesteps"\
    trainer.losses.clip_rgb_loss_depth="False"\
    #trainer.losses.inverse_depth_smoothness.w=0.05\
    #trainer.losses.depth.w=0.02 \
    #trainer.losses.opacity_entropy.w=0.5 \
    #trainer.losses.inverse_depth_smoothness.w=0.01\



