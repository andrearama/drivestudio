export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

output_root="outputs/drivestudio/"
project="drivestudio_night_scenes"
expname="waymo350_no_optimization"
scene_idx=350
steps=$(seq 0 5 195)

test_timesteps="[$(echo $steps | tr ' ' ',')]"

CUDA_VISIBLE_DEVICES=6 python tools/train.py \
    --config_file configs/omnire_extended_cam.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/5cams \
    data.data_root="/external/10g/carlnas/fs1/datasets/waymo/data/waymo/processed/training" \
    data.pixel_source.load_smpl="false" \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    trainer.render.avg_renderings="True"\
    data.pixel_source.use_depth_map_front="False"\
    data.pixel_source.use_depth_map_back="False"\
    logging.vis_freq=10000\
    data.pixel_source.test_timesteps="$test_timesteps"\
    model.RigidNodes.position_adjustments="False" \
    #trainer.losses.inverse_depth_smoothness.w=1\
    #trainer.losses.opacity_entropy.w=0.5 \
    
    #trainer.losses.opacity_entropy.w=0.5 \
    #trainer.losses.inverse_depth_smoothness.w=0.05\
    #trainer.losses.depth.w=0.02 \
    
    
