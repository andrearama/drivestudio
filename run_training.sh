export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

output_root="outputs/drivestudio/"
project="drivestudio_night_scenes"
expname="vehicle_interpolation"
scene_idx=814
steps=$(seq 0 5 190)

test_timesteps="[$(echo $steps | tr ' ' ',')]"

CUDA_VISIBLE_DEVICES=1 python tools/train.py \
    --config_file configs/omnire_extended_cam.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    --vsdebug \
    dataset=nuscenes/1cams \
    data.data_root="data/nuscenes/processed_10Hz/trainval" \
    data.pixel_source.load_smpl="false" \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    trainer.render.avg_renderings="True"\
    logging.vis_freq=10000\
    data.pixel_source.test_timesteps="$test_timesteps"\
    #trainer.optim.num_iters=10000\