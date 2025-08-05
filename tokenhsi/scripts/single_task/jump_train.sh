python ./tokenhsi/run.py --task HumanoidJump \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_jump.yaml \
    --motion_file tokenhsi/data/dataset_jump/dataset_jump.yaml \
    --num_envs 4096 \
    --headless
