import sys
sys.path.append("./")

import os
import os.path as osp
import torch
import trimesh
import numpy as np
import torchgeometry as tgm
import glob

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

from body_models.model_loader import get_body_model

from lpanlib.isaacgym_utils.vis.api import vis_motion_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

joints_to_use = {
    "from_smpl_original_to_smpl_humanoid": np.array([0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]),
    "from_smpl_original_to_phys_humanoid_v3": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
}

def project_joints_simple(motion):
    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"]
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"]
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

    return new_motion

###### SMPL Model Joints
# 0 Pelvis
# 1 L_Hip
# 2 R_Hip
# 3 Torso
# 4 L_Knee
# 5 R_Knee
# 6 Spine
# 7 L_Ankle
# 8 R_Ankle
# 9 Chest
# 10 L_Toe
# 11 R_Toe
# 12 Neck
# 13 L_Thorax
# 14 R_Thorax
# 15 Head
# 16 L_Shoulder
# 17 R_Shoulder
# 18 L_Elbow
# 19 R_Elbow
# 20 L_Wrist
# 21 R_Wrist
# 22 L_Hand
# 23 R_Hand

if __name__ == '__main__':

    all_files = glob.glob(osp.join(osp.dirname(__file__), "motions/*/*/smpl_params.npy"))

    # parameters for motion editing
    candidates = {
        "ACCAD+__+Female1Walking_c3d+__+B19_-_walk_to_pick_up_box_stageii": [105, 205],
        "ACCAD+__+Female1Walking_c3d+__+B21_-_put_down_box_to_walk_stageii": [0, 135],
        "OMOMO+__+sub7_largebox_006": [85, 155],
        "OMOMO+__+sub7_largebox_007": [85, 165],
        "OMOMO+__+sub7_largebox_008": [0, 130],
        "OMOMO+__+sub7_largebox_009": [100, 175],
        "OMOMO+__+sub7_largebox_010": [80, 160],
        "OMOMO+__+sub7_largebox_011": [75, 145],
        "OMOMO+__+sub7_largebox_042": [0, 170],
        "OMOMO+__+sub7_largebox_043": [0, 180],
        "OMOMO+__+sub7_largebox_044": [0, 160],
        "OMOMO+__+sub7_largebox_045": [0, 190],
        "OMOMO+__+sub7_largebox_046": [0, 155],
        "OMOMO+__+sub7_largebox_047": [0, 165],
        "OMOMO+__+sub7_smallbox_046": [0, 205],
        "OMOMO+__+sub7_smallbox_047": [0, 190],
        "OMOMO+__+sub7_smallbox_048": [0, 175],
        "OMOMO+__+sub7_smallbox_049": [0, 190],
        "OMOMO+__+sub7_smallbox_051": [0, 160],
    }

    # load skeleton of smpl_humanoid
    smpl_humanoid_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/smpl_humanoid.xml")
    smpl_humanoid_skeleton = SkeletonTree.from_mjcf(smpl_humanoid_xml_path)

    # load skeleton of phys_humanoid_v3
    phys_humanoid_v3_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/phys_humanoid_v3.xml")
    phys_humanoid_v3_skeleton = SkeletonTree.from_mjcf(phys_humanoid_v3_xml_path)

    # load skeleton of smpl_original
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=1, debug=False)
    jts_global_trans = bm().joints[0, :24, :].cpu().detach().numpy()
    jts_local_trans = np.zeros_like(jts_global_trans)
    for i in range(jts_local_trans.shape[0]):
        parent = bm.parents[i]
        if parent == -1:
            jts_local_trans[i] = jts_global_trans[i]
        else:
            jts_local_trans[i] = jts_global_trans[i] - jts_global_trans[parent]

    skel_dict = smpl_humanoid_skeleton.to_dict()
    skel_dict["node_names"] = [
        "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine", "L_Ankle", "R_Ankle",
        "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", 
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
    ]
    skel_dict["parent_indices"]["arr"] = bm.parents.numpy()
    skel_dict["local_translation"]["arr"] = jts_local_trans
    smpl_original_skeleton = SkeletonTree.from_dict(skel_dict)

    # create tposes
    smpl_humanoid_tpose = SkeletonState.zero_pose(smpl_humanoid_skeleton)
    smpl_original_tpose = SkeletonState.zero_pose(smpl_original_skeleton)

    phys_humanoid_v3_tpose = SkeletonState.zero_pose(phys_humanoid_v3_skeleton)
    local_rotation = phys_humanoid_v3_tpose.local_rotation
    local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")]
    )
    local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")]
    )

    # print("phys_humanoid_v3_tpose")
    # plot_skeleton_state(phys_humanoid_v3_tpose)
    # print("smpl_humanoid_tpose")
    # plot_skeleton_state(smpl_humanoid_tpose)
    # print("smpl_original_tpose")
    # plot_skeleton_state(smpl_original_tpose)

    body_model = get_body_model("SMPL", "NEUTRAL", 1, debug=False)

    for f in all_files:
        skill = f.split("/")[-3]
        seq_name = f.split("/")[-2]

        print("processing [skill: {}] [seq_name: {}]".format(skill, seq_name))
        
        raw_params = np.load(f, allow_pickle=True).item()

        if seq_name in list(candidates.keys()):
            f_start = candidates[seq_name][0]
            f_end = candidates[seq_name][1]
            poses = torch.tensor(raw_params["poses"][f_start:f_end], dtype=torch.float32)
            trans = torch.tensor(raw_params["trans"][f_start:f_end], dtype=torch.float32)
        else:
            poses = torch.tensor(raw_params["poses"], dtype=torch.float32)
            trans = torch.tensor(raw_params["trans"], dtype=torch.float32)
        
        fps = raw_params["fps"]

        # compute world absolute position of root joint
        trans = body_model(
            global_orient=poses[:, 0:3], 
            body_pose=poses[:, 3:72],
            transl=trans[:, :],
        ).joints[:, 0, :].cpu().detach()

        poses = poses.reshape(-1, 24, 3)

        # angle axis ---> quaternion
        poses_quat = tgm.angle_axis_to_quaternion(poses.reshape(-1, 3)).reshape(poses.shape[0], -1, 4)

        # switch quaternion order
        # wxyz -> xyzw
        poses_quat = poses_quat[:, :, [1, 2, 3, 0]]

        # generate motion
        skeleton_state = SkeletonState.from_rotation_and_root_translation(smpl_original_skeleton, poses_quat, trans, is_local=True)
        motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=fps)

        # plot_skeleton_motion_interactive(motion)

        configs = {
            "smpl_humanoid": {
                "skeleton": smpl_humanoid_skeleton,
                "xml_path": smpl_humanoid_xml_path,
                "tpose": smpl_humanoid_tpose,
                "joints_to_use": joints_to_use["from_smpl_original_to_smpl_humanoid"],
                "root_height_offset": 0.015,
            },
            "phys_humanoid_v3": {
                "skeleton": phys_humanoid_v3_skeleton,
                "xml_path": phys_humanoid_v3_xml_path,
                "tpose": phys_humanoid_v3_tpose,
                "joints_to_use": joints_to_use["from_smpl_original_to_phys_humanoid_v3"],
                "root_height_offset": 0.07,
            },
        }

        ###### retargeting ######
        for k, v in configs.items():

            target_origin_global_rotation = v["tpose"].global_rotation.clone()
            
            # 用一个相对于静止的世界坐标系进行旋转 对齐两个初始Tpose
            target_aligned_global_rotation = quat_mul_norm( 
                torch.tensor([-0.5, -0.5, -0.5, 0.5]), target_origin_global_rotation
            )

            # viz_pose = SkeletonState.from_rotation_and_root_translation(
            #     skeleton_tree=v["skeleton"],
            #     r=target_aligned_global_rotation,
            #     t=v["tpose"].root_translation,
            #     is_local=False,
            # )
            # plot_skeleton_state(viz_pose)

            # retargeting... 太TMD简单了 居然搞了两天
            target_final_global_rotation = quat_mul_norm(
                skeleton_state.global_rotation.clone()[..., v["joints_to_use"], :], target_aligned_global_rotation.clone()
            )
            target_final_root_translation = skeleton_state.root_translation.clone()

            new_skeleton_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree=v["skeleton"],
                r=target_final_global_rotation,
                t=target_final_root_translation,
                is_local=False,
            ).local_repr()
            new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state, fps=fps)

            new_motion_params_root_trans = new_motion.root_translation.clone()
            new_motion_params_local_rots = new_motion.local_rotation.clone()

            # check foot-ground penetration
            min_h = torch.min(new_motion.global_translation[:, :, 2], dim=-1)[0].mean()
            # min_h = torch.min(new_motion.global_translation[:, :, 2])
            for i in range(new_motion.global_translation.shape[0]):
                new_motion_params_root_trans[i, 2] += -min_h
            
            # adjust the height of the root to avoid ground penetration
            root_height_offset = v["root_height_offset"]
            new_motion_params_root_trans[:, 2] += root_height_offset

            # update new_motion
            new_skeleton_state = SkeletonState.from_rotation_and_root_translation(v["skeleton"], new_motion_params_local_rots, new_motion_params_root_trans, is_local=True)
            new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state, fps=fps)

            # save retargeted motion
            save_dir = osp.join(osp.dirname(f), k)
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, "ref_motion.npy")
            new_motion.to_file(save_path)

            # plot_skeleton_motion_interactive(new_motion)

            # scenepic animation
            vis_motion_use_scenepic_animation(
                asset_filename=v["xml_path"],
                rigidbody_global_pos=new_motion.global_translation,
                rigidbody_global_rot=new_motion.global_rotation,
                fps=fps,
                up_axis="z",
                color=name_to_rgb['AliceBlue'] * 255,
                output_path=osp.join(save_dir, "ref_motion_render.html"),
            )
