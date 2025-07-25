python ./tokenhsi/run.py --task HumanoidPush \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_push.yaml \
    --motion_file tokenhsi/data/dataset_push/dataset_push.yaml \
    --checkpoint output/Humanoid_25-18-41-24/nn/Humanoid_00002000.pth \
    --test \
    --num_envs 16