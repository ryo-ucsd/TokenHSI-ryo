import sys
sys.path.append("./")

import os
import os.path as osp
import torch
import numpy as np
import torchgeometry as tgm
from tqdm import tqdm
import glob

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

from body_models.model_loader import get_body_model

from lpanlib.isaacgym_utils.vis.api import vis_motion_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

joints_to_use = {
    "from_smpl_original_to_phys_humanoid_v3": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
}

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

    body_model = get_body_model("SMPL", "NEUTRAL", 1, debug=False)

    pbar = tqdm(all_files)
    for f in pbar:
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

            # retargeting...
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
