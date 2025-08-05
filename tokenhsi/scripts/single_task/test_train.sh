python ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_test.yaml \
    --motion_file tokenhsi/data/dataset_test/dataset_test.yaml \
    --num_envs 4096 \
    --headless
