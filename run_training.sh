export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

output_root="outputs/drivestudio/"
project="drivestudio_night_scenes"
expname="learnable_scale_initial_0_05_lr_0_00005"
scene_idx=814

CUDA_VISIBLE_DEVICES=5 python tools/train.py \
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
    trainer.render.avg_renderings="True"\
    #trainer.render.avg_renderings_scale="0.15" \
    #trainer.optim.num_iters=3\