import sys
sys.path.append("./")

import os
import os.path as osp
import glob
import numpy as np
import torch
import torchgeometry as tgm
from tqdm import tqdm

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_identity
from lpanlib.isaacgym_utils.vis.api import vis_motion_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

# --------- configurable bits ----------
TARGET_XML = osp.join(osp.dirname(__file__), "../assets/mjcf/smplx_humanoid_fingers.xml")
WORLD_ALIGN_Q = torch.tensor([-0.5, -0.5, -0.5, 0.5])  # set to quat_identity() if not needed
ROOT_HEIGHT_OFFSET = 0.07
# --------------------------------------

# 24 body names (original order used in your exporter)
BODY_24 = [
    "Pelvis","L_Hip","R_Hip","Torso","L_Knee","R_Knee","Spine","L_Ankle","R_Ankle",
    "Chest","L_Toe","R_Toe","Neck","L_Thorax","R_Thorax","Head",
    "L_Shoulder","R_Shoulder","L_Elbow","R_Elbow","L_Wrist","R_Wrist","L_Hand","R_Hand"
]

# parameters for motion editing
candidates = {
        "OMOMO+__+sub3_largetable_006": [85, 155],
        "OMOMO+__+sub3_largetable_007": [85, 165],
        "OMOMO+__+sub3_largetable_008": [0, 130],
        "OMOMO+__+sub3_largetable_009": [100, 175],
        "OMOMO+__+sub3_largetable_010": [80, 160],
        "OMOMO+__+sub3_largetable_011": [75, 145],
        "OMOMO+__+sub3_largetable_042": [0, 170],
        "OMOMO+__+sub3_largetable_043": [0, 180],
        "OMOMO+__+sub3_largetable_044": [0, 160],
        "OMOMO+__+sub3_largetable_045": [0, 190],
        "OMOMO+__+sub3_largetable_046": [0, 155],
        "OMOMO+__+sub3_largetable_047": [0, 165],
    }

# Option B: drop L_Hand/R_Hand (target rig parents fingers at wrists)
BODY_22 = [n for n in BODY_24 if n not in ("L_Hand","R_Hand")]

# 30 finger names (must match exporter & MJCF)
FINGERS = []
for side in ("L_","R_"):
    for base in ("Thumb","Index","Middle","Ring","Pinky"):
        FINGERS += [f"{side}{base}1", f"{side}{base}2", f"{side}{base}3"]

# Source joint order = 22 body + 30 fingers (total 52)
SRC_NAMES_52 = BODY_22 + FINGERS


def angle_axis_to_xyzw_quat(aa):  # (T, J, 3) -> (T, J, 4) in XYZW
    q_wxyz = tgm.angle_axis_to_quaternion(aa.reshape(-1, 3)).reshape(aa.shape[0], aa.shape[1], 4)
    return q_wxyz[:, :, [1, 2, 3, 0]]  # wxyz -> xyzw


if __name__ == "__main__":
    all_files = glob.glob(osp.join(osp.dirname(__file__), "motions/*/*/smplx_params.npy"))
    if not all_files:
        print("No smplx_params.npy files found.")
        sys.exit(0)

    

    # Load target skeleton (with fingers)
    target_skel = SkeletonTree.from_mjcf(TARGET_XML)
    target_tpose = SkeletonState.zero_pose(target_skel)

    # Name -> index on target
    tgt_index = {n: i for i, n in enumerate(target_skel.node_names)}

    # Sanity: every source joint must exist on target
    missing = [n for n in SRC_NAMES_52 if n not in tgt_index]
    if missing:
        raise ValueError(f"Missing joints in target skeleton: {missing}")

    # Map joint name -> parent name on target
    tgt_parent_name = {}
    for name, idx in tgt_index.items():
        pidx = target_skel.parent_indices[idx]
        #joint name : parent name
        tgt_parent_name[name] = None if pidx < 0 else target_skel.node_names[pidx]

    # Build source "virtual skeleton" (52 joints) with parents remapped by name into SRC order
    name_to_src_idx = {n: i for i, n in enumerate(SRC_NAMES_52)}
    parent_indices = []
    local_trans = []
    for name in SRC_NAMES_52:
        tidx = tgt_index[name]
        pname = tgt_parent_name[name]
        if pname is None:
            parent_indices.append(-1)  # root
        else:
            if pname not in name_to_src_idx:
                raise ValueError(f"Parent '{pname}' of '{name}' not present in SRC_NAMES_52")
            parent_indices.append(name_to_src_idx[pname])
        local_trans.append(target_skel.local_translation[tidx])

    skel_dict = {
        "node_names": SRC_NAMES_52,
        "parent_indices": {"arr": np.array(parent_indices), 'context': {'dtype': 'int64'}},
        "local_translation": {"arr": np.stack(local_trans),'context': {'dtype': 'float32'}},
    }
    src_skel = SkeletonTree.from_dict(skel_dict)
    src_tpose = SkeletonState.zero_pose(src_skel)

    # Precompute indices to slice (T,54,3) -> (T,52,3): keep 22 body (drop L_Hand/R_Hand) + 30 fingers
    body_keep_idx = [BODY_24.index(n) for n in BODY_22]  # 22 from first 24
    finger_idx = list(range(24, 54))                     # last 30 are fingers
    keep_all = body_keep_idx + finger_idx                # len 52
    #keep all has all indices except for the hands

    pbar = tqdm(all_files, desc="Generating")
    for f in pbar:
        raw = np.load(f, allow_pickle=True).item()

        seq_name = f.split("/")[-2]

        if seq_name in list(candidates.keys()):
            f_start = candidates[seq_name][0]
            f_end = candidates[seq_name][1]
            poses = torch.tensor(raw["poses"][f_start:f_end], dtype=torch.float32)
            trans = torch.tensor(raw["trans"][f_start:f_end], dtype=torch.float32)
        else:
            poses = torch.tensor(raw["poses"], dtype=torch.float32)
            trans = torch.tensor(raw["trans"], dtype=torch.float32)
        

        # Robust conversion (handles numpy arrays or torch tensors saved inside .npy)
        poses_np = np.asarray(poses, dtype=np.float32)   # (T, 162) for 54 joints
        trans_np = np.asarray(trans, dtype=np.float32)   # (T, 3)
        fps = float(raw.get("fps", 30.0))

        #so T is the number of poses
        T = poses_np.shape[0]
        if poses_np.shape[-1] != 54 * 3:
            raise ValueError(f"Expected poses with 162 dims (54*3). Got {poses_np.shape[-1]} in {f}")

        # Reshape to (T,54,3) then drop L_Hand/R_Hand -> (T,52,3)
        poses_aa_full = torch.from_numpy(poses_np).reshape(T, 54, 3)
        poses_aa = poses_aa_full[:, keep_all, :].contiguous()
        root_trans = torch.from_numpy(trans_np)

        # Axis-angle -> quats (xyzw)
        src_quat_xyzw = angle_axis_to_xyzw_quat(poses_aa)  # (T,52,4)

        # Build GLOBAL pose on source skeleton from LOCAL rotations + root translation
        src_state_local = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=src_skel,
            r=src_quat_xyzw,
            t=root_trans,
            is_local=True,
        )
        src_motion = SkeletonMotion.from_skeleton_state(src_state_local, fps=fps)

        # Align target T-pose to source/world (optional frame fix)
        target_origin_global = target_tpose.global_rotation.clone()            # (J,4)
        world_align_q = WORLD_ALIGN_Q.to(dtype=target_origin_global.dtype)     # (4,)
        aligned_per_joint = quat_mul_norm(world_align_q, target_origin_global) # (J,4)

        # Broadcast across time -> (T, J, 4)
        target_aligned_global = aligned_per_joint.unsqueeze(0).expand(
            src_motion.global_rotation.shape[0],  # T
            -1, -1
        ).contiguous()
        
        # Start from aligned T-pose for the target
        tgt_final_global = target_aligned_global.clone()
        tgt_final_root_t = src_motion.root_translation.clone()  # copy pelvis world pos

        # Overwrite target joints by name (now 1:1)
        mapped = []
        src_rots_global = src_motion.global_rotation  # (T,52,4)
        for i_src, name in enumerate(SRC_NAMES_52):
            i_tgt = tgt_index[name]
            # Compose: R_target = R_src * R_alignedTpose
            tgt_final_global[:, i_tgt, :] = quat_mul_norm(
                src_rots_global[:, i_src, :],
                target_aligned_global[:, i_tgt, :],
            )
            mapped.append(name)

        # Convert to local on target skeleton
        new_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=target_skel,
            r=tgt_final_global,
            t=tgt_final_root_t,
            is_local=False,
        ).local_repr()
        new_motion = SkeletonMotion.from_skeleton_state(new_state, fps=fps)

        # Ground fix
        min_h = torch.min(new_motion.global_translation[:, :, 2], dim=-1)[0].mean()
        new_root_t = new_motion.root_translation.clone()
        new_root_t[:, 2] += (-min_h + ROOT_HEIGHT_OFFSET)
        new_state2 = SkeletonState.from_rotation_and_root_translation(
            target_skel, new_motion.local_rotation, new_root_t, is_local=True
        )
        new_motion2 = SkeletonMotion.from_skeleton_state(new_state2, fps=fps)

        # Save next to the source file (parallel folder)
        save_dir = osp.join(osp.dirname(f), "smplx_humanoid_fingers")
        os.makedirs(save_dir, exist_ok=True)
        new_motion2.to_file(osp.join(save_dir, "ref_motion.npy"))

        # Quick HTML preview
        vis_motion_use_scenepic_animation(
            asset_filename=TARGET_XML,
            rigidbody_global_pos=new_motion2.global_translation,
            rigidbody_global_rot=new_motion2.global_rotation,
            fps=fps,
            up_axis="z",
            color=name_to_rgb['AliceBlue'] * 255,
            output_path=osp.join(save_dir, "ref_motion_render.html"),
        )

        seq_name = osp.basename(osp.dirname(f))
        pbar.set_postfix_str(f"OK: {seq_name} (mapped {len(mapped)} joints)")
