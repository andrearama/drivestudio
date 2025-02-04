export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

output_root="outputs/drivestudio/"
project="drivestudio_night_scenes"
expname="test_split3"
scene_idx=814
test_timesteps="[163, 194]"

CUDA_VISIBLE_DEVICES=3 python tools/train.py \
    --config_file configs/omnire_extended_cam.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=nuscenes/6cams \
    data.data_root="data/nuscenes/processed_10Hz/trainval" \
    data.pixel_source.load_smpl="false" \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    data.pixel_source.test_timesteps="$test_timesteps"\
    trainer.render.avg_renderings="True"\