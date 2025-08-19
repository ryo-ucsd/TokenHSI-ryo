python ./tokenhsi/run.py --task HumanoidTrajSitCarryClimbPush \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
    --cfg_env tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb_push.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --checkpoint  output/server/Humanoid.pth   \
    --test \
    --num_envs 16