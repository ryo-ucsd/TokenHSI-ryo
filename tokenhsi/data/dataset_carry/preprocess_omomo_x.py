import sys
sys.path.append("./")

import os
import os.path as osp
import yaml
import joblib
import numpy as np
import argparse

FPS_DEFAULT = 30.0
FINGER_MODE = "flat"  # or "rest_curl" for a tiny constant flex

# Finger names that match your MJCF rig
FINGERS_PER_HAND = [
    ("Thumb1","Thumb2","Thumb3"),
    ("Index1","Index2","Index3"),
    ("Middle1","Middle2","Middle3"),
    ("Ring1","Ring2","Ring3"),
    ("Pinky1","Pinky2","Pinky3"),
]

def build_finger_joint_names():
    names = []
    for side in ("L_","R_"):
        for chain in FINGERS_PER_HAND:
            for seg in chain:
                names.append(f"{side}{seg}")
    return names  # 30

def make_finger_axis_angle(T, mode="flat"):
    aa = np.zeros((T, 30*3), dtype=np.float32)
    if mode == "rest_curl":
        # small constant curl (assumes flex about local +x)
        mcp, pip, dip = np.deg2rad([10.0, 8.0, 6.0])
        block = []
        for _ in (0,1):  # L, R
            for _chain in FINGERS_PER_HAND:
                block.extend([mcp, 0.0, 0.0])
                block.extend([pip, 0.0, 0.0])
                block.extend([dip, 0.0, 0.0])
        aa[:] = np.array(block, dtype=np.float32)[None, :]
    return aa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_cfg", type=str, default=osp.join(osp.dirname(__file__), "../dataset_cfg.yaml"))
    args = parser.parse_args()

    with open(args.dataset_cfg, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    omomo_dir  = cfg["omomo_dir"]
    output_dir = osp.join(osp.dirname(__file__), "motions")
    os.makedirs(output_dir, exist_ok=True)

    candidates = {"omomo": cfg["motions"]["omomo"]}

    train = joblib.load(osp.join(omomo_dir, "train_diffusion_manip_seq_joints24.p"))
    test  = joblib.load(osp.join(omomo_dir, "test_diffusion_manip_seq_joints24.p"))

    seq_to_id = {}
    for k, v in train.items():
        seq_to_id[v["seq_name"]] = {"set": "train", "id": k}
    for k, v in test.items():
        seq_to_id[v["seq_name"]] = {"set": "test", "id": k}

    BODY_J = 24
    finger_joint_names = build_finger_joint_names()
    NF = len(finger_joint_names)  # 30

    for skill, seq_list in candidates.items():
        out_skill = osp.join(output_dir, skill)
        os.makedirs(out_skill, exist_ok=True)

        for seq_name in seq_list:
            entry = train[seq_to_id[seq_name]["id"]] if seq_to_id[seq_name]["set"] == "train" else test[seq_to_id[seq_name]["id"]]
            T = entry["root_orient"].shape[0]

            # (T,72) = root_orient(3)+pose_body(63)+zeros(6)
            poses72 = np.concatenate(
                [entry["root_orient"], entry["pose_body"], np.zeros((T,6), dtype=np.float32)],
                axis=-1
            ).astype(np.float32)
            trans = entry["trans"].astype(np.float32)
            fps = float(FPS_DEFAULT)

            base_dir = osp.join(out_skill, f"OMOMO+__+{seq_name}")
            os.makedirs(base_dir, exist_ok=True)

            # A) Save original 24-joint file
            np.save(osp.join(base_dir, "smpl_params.npy"), {"poses": poses72, "trans": trans, "fps": fps})

            # B) Save finger-ready SMPL-X style (still axis-angle, but 24+30=54 joints)
            finger_aa = make_finger_axis_angle(T, mode=FINGER_MODE)  # (T,90)
            poses162 = np.concatenate([poses72, finger_aa], axis=-1).astype(np.float32)  # (T,162)
            np.save(osp.join(base_dir, "smplx_params.npy"), {"poses": poses162, "trans": trans, "fps": fps})

            # C) Sidecar meta
            joint_names_24 = [
                "Pelvis","L_Hip","R_Hip","Torso","L_Knee","R_Knee","Spine","L_Ankle","R_Ankle",
                "Chest","L_Toe","R_Toe","Neck","L_Thorax","R_Thorax","Head",
                "L_Shoulder","R_Shoulder","L_Elbow","R_Elbow","L_Wrist","R_Wrist","L_Hand","R_Hand"
            ]
            meta = {
                "joint_names": joint_names_24 + finger_joint_names,
                "num_body_joints": BODY_J,
                "num_finger_joints": NF,
                "finger_joint_names": finger_joint_names,
                "driven_mask": np.array([True]*BODY_J + [False]*NF, dtype=bool),
                "fps": fps,
            }
            np.save(osp.join(base_dir, "ref_motion_meta.npy"), meta, allow_pickle=True)

            print(f"Saved: {osp.join(base_dir, 'smpl_params.npy')}")
            print(f"Saved: {osp.join(base_dir, 'smplx_params.npy')}")
            print(f"Saved: {osp.join(base_dir, 'ref_motion_meta.npy')}")
