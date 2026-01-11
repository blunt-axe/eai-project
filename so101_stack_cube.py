import grasp_cube.agents.robots.so101.so_101_ee

import numpy as np
import sapien
import torch
import torch.nn.functional as F
import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.agents import register_agent
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array
from mani_skill.agents import MultiAgent

import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent

from utils.distort import get_distorted_image_tensor

from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_configs,
    update_camera_configs_from_dict,
)

from transforms3d.euler import euler2quat

from so101_lift_cube import PickCubeSO101Env


from utils.builder import (
    build_cube_w_friction,
    build_box_w_friction,
)

@register_env("StackCubeSO101-v0", max_episode_steps=200)
class StackCubeSO101Env(PickCubeSO101Env):
    def __init__(self, *args, **kwargs):
        # 1. Define constants FIRST
        self.RED_SIZE = 0.03
        self.GREEN_SIZE = 0.03
        self.MIN_SEPARATION = 0.05
        self.CENTER_X, self.CENTER_Y = 0.472, 0.26
        self.RAND_RANGE = 0.06 

        # 2. Then call super().__init__
        # This will trigger _load_scene, which can now find GREEN_SIZE
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        # Build table and red cube from Lift class
        super()._load_scene(options)

        # Add the Green Cube
        self.green_half = self.GREEN_SIZE / 2
        self.green_cube = build_cube_w_friction(
            self.scene, 
            half_size=self.green_half, 
            color=[0, 1, 0, 1], 
            name="target_green_cube", 
            body_type="dynamic",
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.1
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Initialize agent and red cube from Lift class
        super()._initialize_episode(env_idx, options)
        
        b = len(env_idx)
        with torch.device(self.device):
            # Randomize Green Cube
            green_xy = (torch.rand((b, 2)) * 2 - 1) * self.RAND_RANGE + torch.tensor([self.CENTER_X, self.CENTER_Y])
            
            self.green_cube.set_pose(Pose.create_from_pq(
                p=torch.cat([green_xy, torch.ones((b, 1)) * self.green_half], dim=-1), 
                q=torch.tensor([1, 0, 0, 0]).repeat(b, 1)
            ))

            # Ensure Red Cube is not overlapping Green Cube
            red_xyz = torch.zeros((b, 3))
            red_xyz[:, 2] = torch.ones((b,)) * self.RED_SIZE / 2
            for i in range(b):
                while True:
                    cand_xy = (torch.rand(2) * 2 - 1) * self.RAND_RANGE + torch.tensor([self.CENTER_X, self.CENTER_Y])
                    if torch.linalg.norm(cand_xy - green_xy[i]) > self.MIN_SEPARATION:
                        red_xyz[i, :2] = cand_xy
                        break

            yaw = (torch.rand(b) * 2 - 1) * torch.pi
            half_yaw = yaw * 0.5
            qs = torch.zeros((b, 4), device=self.device)
            qs[:, 0] = torch.cos(half_yaw)  # w
            qs[:, 3] = torch.sin(half_yaw)  # z
            
            self.red_cube.set_pose(Pose.create_from_pq(p=red_xyz, q=qs))

    def evaluate(self):
        # Success criteria: Red on top of Green
        # red, green = self.red_cube.pose, self.green_cube.pose
        
        # xy_dist = torch.linalg.norm(red.p[:, :2] - green.p[:, :2], dim=1)
        # aligned_xy = xy_dist < 0.005

        # # Red Z should be roughly Green Z + Green_Size
        # height_ok = torch.abs(red.p[:, 2] - (green.p[:, 2] + self.GREEN_SIZE)) < 0.003

        # lin_vel = torch.linalg.norm(self.red_cube.linear_velocity, dim=1)
        # stable = (lin_vel < 0.1)

        # success = aligned_xy & height_ok & stable
        # return {"success": success, "aligned_xy": aligned_xy, "height_ok": height_ok}
        

        # For evaluation, need to release        
        red, green = self.red_cube.pose, self.green_cube.pose

        height_ok = torch.abs(red.p[:, 2] - (green.p[:, 2] + self.GREEN_SIZE)) < 0.003

        contacting_force = self.scene.get_pairwise_contact_forces(self.red_cube, self.green_cube)
        contacting = torch.linalg.norm(contacting_force, dim=1) > 0.01

        lin_vel = torch.linalg.norm(self.red_cube.linear_velocity, dim=1)
        stable = (lin_vel < 0.1)

        qpos = self.agent.robot.get_qpos()
        qpos_gripper = qpos[..., -1]
        not_grasping = qpos_gripper >= 0.6
        
        success = height_ok & contacting & stable & not_grasping

        return {"success": success, "contacting": contacting, "not_grasping": not_grasping}

    def compute_dense_reward(self, obs: any, action: Array, info: dict):
        # --- Poses ---
        tcp_pose = self.agent.tcp_pose
        red_pose = self.red_cube.pose
        green_pose = self.green_cube.pose

        is_grasping = self.agent.is_grasping(self.red_cube).float()

        # ======================================================
        # 1. Reach reward (ONLY before grasp)
        # ======================================================
        tcp_dist = torch.linalg.norm(tcp_pose.p - red_pose.p, dim=1)
        reach_reward = (1 - is_grasping) * (1 - torch.tanh(5.0 * tcp_dist)) * (red_pose.p[:, 2] < 0.02).float()

        # ======================================================
        # 2. Grasp reward (simple & stable)
        # ======================================================
        grasp_reward = is_grasping * 1.0

        # ======================================================
        # 3. Lift height reward (after grasp)
        # ======================================================
        h_low = 0.03
        h_high = 0.05

        lift_height = red_pose.p[:, 2] - self.RED_SIZE / 2  # height above table

        low_penalty  = torch.tanh( 20 * torch.clamp(h_low - lift_height, min=0.0) )
        high_penalty = torch.tanh( 20 * torch.clamp(lift_height - h_high, min=0.0) )

        lift_reward = is_grasping * (1.0 - low_penalty - high_penalty)
        lift_reward = torch.clamp(lift_reward, min=0.0)

        # ======================================================
        # 4. Align XY with green cube (after grasp & lifted)
        # ======================================================
        xy_dist = torch.linalg.norm(
            red_pose.p[:, :2] - green_pose.p[:, :2], dim=1
        )
        align_reward = 0.2 + (
            is_grasping
            * (lift_height > self.GREEN_SIZE).float()
            * (1 - torch.tanh(5.0 * xy_dist))
        ) * 0.8
        #if(is_grasping.mean().item() > 0.2):
        #    print("lift_height:", lift_height.mean().item())
        #    print("lift_reward:", lift_reward.mean().item())
        #    print("xy_dist:", xy_dist.mean().item())
        #    print("align_reward:", align_reward.mean().item())
        # ======================================================
        # 5. Place height reward (approach green top)
        # ======================================================
        target_z = green_pose.p[:, 2] + self.GREEN_SIZE
        z_dist = torch.abs(red_pose.p[:, 2] - target_z)
        place_reward = (
            (xy_dist < 0.02).float()
            * (1 - torch.tanh(10.0 * z_dist))
            * is_grasping  # only when grasping
        )

        # place_offhand_reward = (
        #     (1.75+(1 - torch.tanh(10.0 * z_dist)))
        #     * (1 - is_grasping)  # only when not grasping
        #     * (xy_dist < 0.02).float()
        #     * (torch.abs(red_pose.p[:, 2] - (green_pose.p[:, 2] + self.GREEN_SIZE)) < 0.01).float()
        # )

        
        # ======================================================
        # 6. Slow motion reward (after grasp)
        # ======================================================
        #qvel = self.agent.robot.get_qvel()[..., :-1]
        #slow_reward = is_grasping * (1 - torch.tanh(4.0 * torch.linalg.norm(qvel, dim=1)))

        # ======================================================
        # Total reward
        # ======================================================
        total_reward = (
            1.0 * reach_reward
            + 1.5 * grasp_reward
            + 4.0 * lift_reward
            * align_reward
            + 4.0 * place_reward
            # + 4.0 * place_offhand_reward
        )

        # Success bonus
        total_reward[info["success"]] = 15.0

        return total_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        reward = torch.clamp(self.compute_dense_reward(obs, action, info) / 15.0, 0.0, 1.0)
        return reward