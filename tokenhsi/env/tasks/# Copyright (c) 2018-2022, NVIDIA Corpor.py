# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import yaml
from enum import Enum
import numpy as np
import torch
import json
import trimesh
import pickle

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

HEAD2ROOT_OFFSET = 0.730045

class HumanoidAdaptCarry2Push(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # manage multi task obs
        self._num_tasks = 2
        task_obs_size_carry = 3 + 3 + 6 + 3 + 3 + 3 * 8 # bps
        self._each_subtask_obs_size = [
            task_obs_size_carry, # new carry
            task_obs_size_carry, # old carry
        ]
        self._multiple_task_names = ["new_carry", "old_carry"]
        self._enable_task_mask_obs = False

        # configs for task
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        self._mode = cfg["env"]["mode"] # determine which set of objects to use (train or test)
        assert self._mode in ["train", "test"]

        if cfg["args"].eval:
            self._mode = "test"
 
        # configs for box
        box_cfg = cfg["env"]["box"]
        self._build_train_sizes = box_cfg["build"]["trainSizes"]
        self._build_test_sizes = box_cfg["build"]["testSizes"]

        self._enable_bbox_obs = box_cfg["obs"]["enableBboxObs"]

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAdaptCarry2Push.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {} # to enable multi-skill reference init, use dict instead of list
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]
        self._power_dof_ids = [
            0, 1, 2, # abdomen
            3, 4, 5, # neck
            6, 7, 8, # right_shoulder
            9, # right_elbow
            10, 11, 12, # left_shoulder
            13, # left_elbow
            14, 15, 16, # right_hip
            17, 18, 19, # right_knee
            20, 21, 22, # right_ankle
            23, 24, 25, # left_hip
            26, 27, 28, # left_knee
            29, 30, 31, # left_ankle
        ]
        self._print_power_reward = cfg["env"]["print_power_reward"]
        if self._print_power_reward:

            self._print_env_id = 0

            plt.ion()

            N = cfg["env"]["episodeLength"]
            
            fig, self._power_vis_ax = plt.subplots()
            self._power_vis_ax.set_xlim([0, N])
            self._power_vis_ax.set_ylim([-1.0, 0])
            self._power_vis_ax.set_autoscale_on(False)
            self._power_vis_ax.set_xticks(np.arange(0, N, 100))
            self._power_vis_ax.set_yticks(np.arange(-1.0, 0, 0.2))
            self._power_vis_ax.grid(True)

            self._power_curve_x = []
            self._power_curve_y = []
            self._power_vis_line, = self._power_vis_ax.plot(self._power_curve_x, self._power_curve_y, label='Power Reward Curve of Env {}'.format(self._print_env_id), color='cornflowerblue')
            self._power_vis_ax.legend(loc='lower center', ncol=4, prop=font_manager.FontProperties(size=10))

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._skill = cfg["env"]["skill"]
        self._skill_init_prob = torch.tensor(cfg["env"]["skillInitProb"], device=self.device, dtype=torch.float) # probs for state init
        self._skill_disc_prob = torch.tensor(cfg["env"]["skillDiscProb"], device=self.device, dtype=torch.float) # probs for amp obs demo fetch

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        # tensors for task
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_root_rot = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)
        self._prev_box_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        self._box_init_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        self._tar_facing_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_facing_dir[..., 0] = 1.0 # x
        assert cfg["env"]["pushDir"] == "x"

        self._box_default_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._box_default_dir[..., 0] = 1.0 # x

        self._power_dof_ids = to_torch(self._power_dof_ids, device=self.device, dtype=torch.long)

        if (not self.headless):
            self._build_marker_state_tensors()

        # tensors for box
        self._build_box_tensors()

        # tensors for enableTrackInitState
        self._every_env_init_dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        # tensors for fixing obs bug
        self._kinematic_humanoid_rigid_body_states = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device, dtype=torch.float)

        ###### evaluation!!!
        self._is_eval = cfg["args"].eval
        if self._is_eval:

            self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

            self._success_threshold = cfg["env"]["eval"]["successThreshold"]

            self._skill = cfg["env"]["eval"]["skill"]
            self._skill_init_prob = torch.tensor(cfg["env"]["eval"]["skillInitProb"], device=self.device, dtype=torch.float) # probs for state init

        return
    
    def get_multi_task_info(self):

        num_subtasks = self._num_tasks
        each_subtask_obs_size = self._each_subtask_obs_size

        each_subtask_obs_mask = torch.zeros(num_subtasks, sum(each_subtask_obs_size), dtype=torch.bool, device=self.device)

        index = torch.cumsum(torch.tensor([0] + each_subtask_obs_size), dim=0).to(self.device)
        for i in range(num_subtasks):
            each_subtask_obs_mask[i, index[i]:index[i + 1]] = True
    
        info = {
            "onehot_size": num_subtasks,
            "tota_subtask_obs_size": sum(each_subtask_obs_size),
            "each_subtask_obs_size": each_subtask_obs_size,
            "each_subtask_obs_mask": each_subtask_obs_mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": self._enable_task_mask_obs,

            "each_subtask_name": self._multiple_task_names,
            "major_task_name": "carry",
            "has_extra": False,
        }

        return info
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        self._box_handles = []
        self._box_masses = []
        self._load_box_asset()

        super()._create_envs(num_envs, spacing, num_per_row)

        self._box_masses = to_torch(self._box_masses, device=self.device, dtype=torch.float32)
        return
    
    def _load_marker_asset(self):
        asset_root = "mcp_on_box/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return
    
    def _load_box_asset(self):

        num_boxes = len(self._build_train_sizes)
        sizes = self._build_train_sizes
        if self._mode == "test":
            num_boxes = len(self._build_test_sizes)
            sizes = self._build_test_sizes

        sampled_ids = torch.multinomial(torch.ones(num_boxes) * (1.0 / num_boxes), num_samples=self.num_envs, replacement=True)
        
        self._box_size = torch.tensor(sizes, device=self.device, dtype=torch.float32)[sampled_ids]
        
        # randomize mass
        self._box_density = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
        self._box_density[:] = 10

        # create asset
        self._box_assets = []
        for i in range(self.num_envs):
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = self._box_density[i]
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(self.gym.create_box(self.sim, self._box_size[i, 0], self._box_size[i, 1], self._box_size[i, 2], asset_options))
        
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_box(env_id, env_ptr)

        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return

    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = self._box_size[env_id, 0] / 2 + 3.0
        default_pose.p.y = 0
        default_pose.p.z = self._box_size[env_id, 2] / 2 # ensure no penetration between box and ground plane
    
        box_handle = self.gym.create_actor(env_ptr, self._box_assets[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
        self._box_handles.append(box_handle)

        mass = self.gym.get_actor_rigid_body_properties(env_ptr, box_handle)[0].mass
        self._box_masses.append(mass)

        return
    
    def _build_marker(self, env_id, env_ptr):
        col_group = self.num_envs + 1
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.5)
        self._marker_handles.append(marker_handle)

        return

    def _build_box_tensors(self):
        num_actors = self.get_num_actors_per_env()

        idx = self._box_handles[0]
        self._box_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        
        self._box_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + idx

        self._initial_box_states = self._box_states.clone()
        self._initial_box_states[:, 7:13] = 0

        self._build_box_bps()

        return

    def _build_box_bps(self):
        bps_0 = torch.vstack([     self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_1 = torch.vstack([-1 * self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_2 = torch.vstack([-1 * self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_3 = torch.vstack([     self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_4 = torch.vstack([     self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_5 = torch.vstack([-1 * self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_6 = torch.vstack([-1 * self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_7 = torch.vstack([     self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        self._box_bps = torch.cat([bps_0, bps_1, bps_2, bps_3, bps_4, bps_5, bps_6, bps_7], dim=1).to(self.device) # (num_envs, 8, 3)

        return
    
    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        idx = self._marker_handles[0]
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
        self._marker_pos = self._marker_states[..., :3]
        
        self._marker_actor_ids = self._humanoid_actor_ids + idx

        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size += sum(self._each_subtask_obs_size[:-1]) # exclude redundant one
        return obs_size

    def _update_task(self):
        return

    def _reset_task(self, env_ids):
        return
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return
    
    def _update_marker(self):

        self._marker_pos[:, :] = self._humanoid_root_states[..., 0:3]
        self._marker_pos[:, 2] = 2.0

        env_ids_int32 = torch.cat([self._marker_actor_ids, self._box_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def _draw_task(self):
        self._update_marker()

        vel_scale = 2.0
        heading_cols = np.array([[0.0, 1.0, 0.0], # 物体的目标线速度
                                [1.0, 0.0, 0.0]], dtype=np.float32) # 物体当前线速度

        self.gym.clear_lines(self.viewer)

        if self._show_lines_flag:

            root_pos = self._box_states[..., 0:3]
            # prev_root_pos = self._prev_box_pos
            # sim_vel = (root_pos - prev_root_pos) / self.dt
            # sim_vel[..., -1] = 0

            # starts = root_pos
            # starts[..., -1] = 2.0 # 高点，好能看见

            # tar_ends = torch.clone(starts)
            # tar_ends[..., 0:2] += vel_scale * self._tar_speed.unsqueeze(-1) * self._tar_facing_dir
            # sim_ends = starts + vel_scale * sim_vel

            # verts = torch.cat([starts, tar_ends, starts, sim_ends], dim=-1).cpu().numpy()

            # for i, env_ptr in enumerate(self.envs):
            #     curr_verts = verts[i:i+1]
            #     curr_verts = curr_verts.reshape([2, 6])
            #     self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, heading_cols)

        return
    
    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return
    
    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            box_states = self._box_states
            box_bps = self._box_bps
            tar_pos = self._box_init_pos

        else:
            root_states = self._humanoid_root_states[env_ids]
            box_states = self._box_states[env_ids]
            box_bps = self._box_bps[env_ids]
            tar_pos = self._box_init_pos[env_ids]
        
        obs = compute_location_observations(root_states, box_states, box_bps, tar_pos,
                                            self._enable_bbox_obs)

        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        rigid_body_pos = self._rigid_body_pos
        box_pos = self._box_states[..., 0:3]
        box_rot = self._box_states[..., 3:7]
        hands_ids = self._key_body_ids[[0, 1]]

        head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "head")
        head_pos = rigid_body_pos[:, head_id]

        pushing_r = compute_pushing_reward(root_pos, box_pos, self._prev_box_pos, box_rot, self._box_init_pos, self._box_default_dir, self._tar_facing_dir, self.dt)

        power = torch.abs(torch.multiply(self.dof_force_tensor[:, self._power_dof_ids], self._dof_vel[:, self._power_dof_ids])).sum(dim = -1)
        power_reward = -self._power_coefficient * power
        if self._print_power_reward:
            self._power_curve_x.append(self.progress_buf[self._print_env_id].item())
            self._power_curve_y.append(power_reward[self._print_env_id].item())
            self._power_vis_line.set_data(self._power_curve_x, self._power_curve_y)
            plt.pause(0.000001)

        if self._power_reward:
            self.rew_buf[:] = pushing_r + power_reward
        else:
            self.rew_buf[:] = pushing_r

        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_root_rot[:] = self._humanoid_root_states[..., 3:7]
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        return
    
    def _reset_boxes(self, env_ids):

        # for skill is omomo, carryWith, the initial location of the box is from the reference box motion
        for sk_name in ["push"]:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):

                    curr_env_ids = self._reset_ref_env_ids[sk_name]

                    root_pos = self._box_size[curr_env_ids].clone() / 2 # on the ground
                    root_pos[:, 1] = self._humanoid_root_states[curr_env_ids, 1] # y

                    max_x_coord = self._kinematic_humanoid_rigid_body_states[curr_env_ids, :, 0].max(dim=-1)[0]
                    root_pos[:, 0] += max_x_coord + 2 # 先走过去，再击打 # Walk over first, then hit

                    self._box_states[curr_env_ids, 0:3] = root_pos
                    self._box_states[curr_env_ids, 3:7] = 0.0
                    self._box_states[curr_env_ids, 6] = 1.0 # quaternion
                    self._box_states[curr_env_ids, 7:10] = 0.0
                    self._box_states[curr_env_ids, 10:13] = 0.0

                    self._box_init_pos[curr_env_ids, :] = root_pos
        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        if self._is_eval:
            self._success_buf[env_ids] = 0
            self._precision_buf[env_ids] = float('Inf')

        env_ids_int32 = self._box_actor_ids[env_ids].view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def save_imgs(self):
        if self.cfg["args"].record:
            # render frame if we're saving video
            if self.save_video and self.viewer is not None:
                num = str(self.save_img_count)
                num = '0' * (6 - len(num)) + num
                new_save_dir = os.path.join(self.save_video_dir, "imgs")
                os.makedirs(new_save_dir, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer, f"{new_save_dir}/frame_{num}.png")

                parameters = {
                    "hu_rigid_body_pos_rot": self._fetch_humanoid_rigid_body_pos_rot_states().cpu(),
                    "box_states": self._box_states.clone().cpu(),
                    "box_sizes": self._box_size.clone().cpu(),
                    # "box_tar_pos": self._tar_pos.clone().cpu(),
                    "dt": self.dt,
                    "progress": self.progress_buf.clone().cpu()
                }

                new_save_dir = os.path.join(self.save_video_dir, "parameters")
                os.makedirs(new_save_dir, exist_ok=True)
                with open(f"{new_save_dir}/frame_{num}.pkl", 'wb') as f:
                    pickle.dump(parameters, f)

                self.save_img_count += 1

        else:
            super().save_imgs()

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        self.extras["policy_obs"] = self.obs_buf.clone()

        if self._is_eval:
            self._compute_metrics_evaluation()
            self.extras["success"] = self._success_buf
            self.extras["precision"] = self._precision_buf

        return
    
    def _compute_metrics_evaluation(self):

        box_falled = self._box_states[..., 2] < self._box_size[..., 2] / 2.0

        self._success_buf[box_falled] += 1

        self._precision_buf[box_falled] = 0.0 # no erros

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):

        # random sample a skill
        sk_id = torch.multinomial(self._skill_disc_prob, num_samples=1, replacement=True)
        sk_name = self._skill[sk_id]
        curr_motion_lib = self._motion_lib[sk_name]

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = curr_motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = curr_motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0, curr_motion_lib)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0, motion_lib):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/phys_humanoid.xml") or (asset_file == "mjcf/phys_humanoid_v2.xml") or (asset_file == "mjcf/phys_humanoid_v3.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 2 * 2 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)

        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            self._skill_categories = list(motion_config['motions'].keys()) # all skill names stored in the yaml file
            self._motion_lib = {}
            for skill in self._skill_categories:
                self._motion_lib[skill] = MotionLib(motion_file=motion_file,
                                                    skill=skill,
                                                    dof_body_ids=self._dof_body_ids,
                                                    dof_offsets=self._dof_offsets,
                                                    key_body_ids=self._key_body_ids.cpu().numpy(), 
                                                    device=self.device)
        else:
            raise NotImplementedError

        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_boxes(env_ids)
            self._reset_task(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._init_amp_obs(env_ids)

            if self._print_power_reward and (self._print_env_id in env_ids):
                self._power_curve_x = []
                self._power_curve_y = []
            
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAdaptCarry2Push.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAdaptCarry2Push.StateInit.Start
              or self._state_init == HumanoidAdaptCarry2Push.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAdaptCarry2Push.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

        self._kinematic_humanoid_rigid_body_states[env_ids] = self._initial_humanoid_rigid_body_states[env_ids]

        self._every_env_init_dof_pos[env_ids] = self._initial_dof_pos[env_ids] # for "enableTrackInitState"

        return

    def _reset_ref_state_init(self, env_ids):
        sk_ids = torch.multinomial(self._skill_init_prob, num_samples=env_ids.shape[0], replacement=True)

        for uid, sk_name in enumerate(self._skill):
            curr_motion_lib = self._motion_lib[sk_name]
            curr_env_ids = env_ids[(sk_ids == uid).nonzero().squeeze(-1)] # be careful!!!

            if len(curr_env_ids) > 0:

                num_envs = curr_env_ids.shape[0]
                motion_ids = curr_motion_lib.sample_motions(num_envs)

                if (self._state_init == HumanoidAdaptCarry2Push.StateInit.Random
                    or self._state_init == HumanoidAdaptCarry2Push.StateInit.Hybrid):
                    motion_times = curr_motion_lib.sample_time_rsi(motion_ids) # avoid times with serious self-penetration
                elif (self._state_init == HumanoidAdaptCarry2Push.StateInit.Start):
                    motion_times = torch.zeros(num_envs, device=self.device)
                else:
                    assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

                root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                    = curr_motion_lib.get_motion_state(motion_ids, motion_times)

                self._set_env_state(env_ids=curr_env_ids, 
                                    root_pos=root_pos, 
                                    root_rot=root_rot, 
                                    dof_pos=dof_pos, 
                                    root_vel=root_vel, 
                                    root_ang_vel=root_ang_vel, 
                                    dof_vel=dof_vel)

                self._reset_ref_env_ids[sk_name] = curr_env_ids
                self._reset_ref_motion_ids[sk_name] = motion_ids
                self._reset_ref_motion_times[sk_name] = motion_times

                # update buffer for kinematic humanoid state
                body_pos, body_rot, body_vel, body_ang_vel \
                    = curr_motion_lib.get_motion_state_max(motion_ids, motion_times)
                self._kinematic_humanoid_rigid_body_states[curr_env_ids] = torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)

                self._every_env_init_dof_pos[curr_env_ids] = dof_pos # for "enableTrackInitState"

        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        for i, sk_name in enumerate(self._skill):
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):
                    self._init_amp_obs_ref(self._reset_ref_env_ids[sk_name], self._reset_ref_motion_ids[sk_name],
                                           self._reset_ref_motion_times[sk_name], sk_name)

        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times, skill_name):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib[skill_name].get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        if (env_ids is None):
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
        else:
            kinematic_rigid_body_pos = self._kinematic_humanoid_rigid_body_states[:, :, 0:3]
            key_body_pos = kinematic_rigid_body_pos[:, self._key_body_ids, :]
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._kinematic_humanoid_rigid_body_states[env_ids, 0, 0:3],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 3:7],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 7:10],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 10:13],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights,):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor,) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(torch.ones_like(fall_height), fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs


@torch.jit.script
def compute_location_observations(root_states, box_states, box_bps, tar_pos, enableBboxObs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot) # (num_envs, 4)

    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    box_vel = box_states[:, 7:10]
    box_ang_vel = box_states[:, 10:13]
    
    local_box_pos = box_pos - root_pos
    local_box_pos = quat_rotate(heading_rot, local_box_pos)

    local_box_rot = quat_mul(heading_rot, box_rot)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(local_box_rot)

    local_box_vel = quat_rotate(heading_rot, box_vel)
    local_box_ang_vel = quat_rotate(heading_rot, box_ang_vel)

    # compute observations for bounding points of the box
    ## transform from object local space to world space
    box_pos_exp = torch.broadcast_to(box_pos.unsqueeze(-2), (box_pos.shape[0], box_bps.shape[1], box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
    box_rot_exp = torch.broadcast_to(box_rot.unsqueeze(-2), (box_rot.shape[0], box_bps.shape[1], box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
    box_bps_world_space = quat_rotate(box_rot_exp.reshape(-1, 4), box_bps.reshape(-1, 3)) + box_pos_exp.reshape(-1, 3) # (num_envs*8, 3)

    ## transform from world space to humanoid local space
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(-2), (heading_rot.shape[0], box_bps.shape[1], heading_rot.shape[1]))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(-2), (root_pos.shape[0], box_bps.shape[1], root_pos.shape[1]))
    box_bps_local_space = quat_rotate(heading_rot_exp.reshape(-1, 4), box_bps_world_space - root_pos_exp.reshape(-1, 3)) # (num_envs*8, 3)

    # task obs
    local_tar_pos = quat_rotate(heading_rot, tar_pos - root_pos) # 3d xyz

    if enableBboxObs:
        obs = torch.cat([local_box_vel, local_box_ang_vel, local_box_pos, local_box_rot_obs, box_bps_local_space.reshape(root_pos.shape[0], -1), local_tar_pos], dim=-1)
    else:
        obs = torch.cat([local_box_vel, local_box_ang_vel, local_box_pos, local_box_rot_obs, local_tar_pos], dim=-1)

    return obs

@torch.jit.script
def compute_pushing_reward(root_pos, box_pos, prev_box_pos, box_rot, box_init_pos, box_default_dir, box_tar_dir, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor

    # 人距离箱子越近越好
    human2box_2d = torch.sum((root_pos[..., 0:2] - box_pos[..., 0:2]) ** 2, dim=-1)
    box2its_init_pos_2d = torch.sum((box_init_pos[..., 0:2] - box_pos[..., 0:2]) ** 2, dim=-1).sqrt()

    r1 = torch.exp(-0.5 * human2box_2d)
    r2 = box2its_init_pos_2d ** 2
    r2 = torch.clamp_max(r2, 1.0) # 打出去1m已经够可以的了

    reward = 0.3 * r1 + 0.7 * r2

    # vel_err_scale = 10.0
    # tangent_err_w = 0.1

    # dir_reward_w = 0.7
    # facing_reward_w = 0.3

    # # 约束箱子在其目标朝向上的线速度大小等于目标速度，同时约束目标朝向的切线方向上的线速度大小为0
    # delta_root_pos = box_pos - prev_box_pos
    # root_vel = delta_root_pos / dt
    # tar_dir_speed = torch.sum(box_tar_dir * root_vel[..., :2], dim=-1)

    # tar_dir_vel = tar_dir_speed.unsqueeze(-1) * box_tar_dir
    # tangent_vel = root_vel[..., :2] - tar_dir_vel

    # tangent_speed = torch.sum(tangent_vel, dim=-1)

    # tar_vel_err = tar_speed - tar_dir_speed
    # tangent_vel_err = tangent_speed
    # dir_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err + 
    #                     tangent_err_w * tangent_vel_err * tangent_vel_err))

    # speed_mask = tar_dir_speed <= 0
    # dir_reward[speed_mask] = 0

    # # 约束箱子的当前朝向和其目标朝向一致
    # box_heading_rot = torch_utils.calc_heading_quat(box_rot)
    # box_default_dir = torch.cat([box_default_dir, torch.zeros_like(box_default_dir[..., 0:1])], dim=-1)
    # rotated_box_default_dir = quat_rotate(box_heading_rot, box_default_dir)
    # facing_err = torch.sum(box_tar_dir * rotated_box_default_dir[..., 0:2], dim=-1)
    # facing_reward = torch.clamp_min(facing_err, 0.0)

    # reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    return reward

